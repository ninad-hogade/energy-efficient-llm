# src/train.py (modified to expose metrics)

# Add to imports
import json
import os
import sys
import time
from datetime import datetime
import torch
import torch.distributed as dist
import deepspeed
import logging
import argparse
from omegaconf import OmegaConf
from energy_monitor import EnergyMonitor
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_scheduler
)
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("training")

# Ensure we have proper error handling
def setup_distributed(rank, world_size, master_addr, master_port):
    """Initialize the distributed environment with proper error handling."""
    logger.info(f"Setting up distributed: rank={rank}, world_size={world_size}")
    
    # Set the environment variables for distributed
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = "0"  # we handle one GPU per pod
    
    # Initialize the distributed backend
    try:
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank
        )
        logger.info(f"Successfully initialized process group for rank {rank}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize distributed: {e}")
        return False

class AdaptiveTrainer(Trainer):
    def __init__(self, energy_monitor=None, config=None, **kwargs):
        super().__init__(**kwargs)
        self.energy_monitor = energy_monitor
        self.config = config
        self.last_adaptation_time = time.time()
        self.frozen_layers = set()
        self.precision_mode = "fp16"  # Start with fp16
        self.energy_consumed_kwh = 0
        self.initial_power_usage = None
        self.eval_metrics = {}
    
    def training_step(self, model, inputs):
        """Override training step to implement energy-efficient adaptations"""
        current_time = time.time()
        
        # Apply adaptive techniques based on interval
        if hasattr(self.config, 'deepspeed') and hasattr(self.config.deepspeed, 'adaptive_training') and \
           hasattr(self.config.deepspeed.adaptive_training, 'adaptation_cooldown') and \
           (current_time - self.last_adaptation_time > self.config.deepspeed.adaptive_training.adaptation_cooldown):
            
            # 1. Adapt precision if enabled
            if hasattr(self.config.training, 'adaptive_precision') and self.config.training.adaptive_precision:
                self._adapt_precision(model)
            
            # 2. Apply dynamic layer freezing if enabled
            if hasattr(self.config.training, 'dynamic_layer_freezing') and self.config.training.dynamic_layer_freezing:
                self._adapt_layer_freezing(model)
            
            # Log energy metrics
            if self.state.global_step % 100 == 0:
                self.log_energy_metrics(model)
            
            self.last_adaptation_time = current_time
        
        # Perform regular training step
        return super().training_step(model, inputs)
    
    def _adapt_precision(self, model):
        """Switch between FP16 and BF16 based on energy metrics"""
        if not hasattr(self.energy_monitor, 'stats_history') or not self.energy_monitor.stats_history:
            return
        
        # Get latest energy stats
        latest_stats = self.energy_monitor.stats_history[-1]
        avg_power = sum(stat.power_usage for stat in latest_stats.values()) / len(latest_stats)
        
        # Record initial power usage if not set
        if self.initial_power_usage is None:
            self.initial_power_usage = avg_power
            return
        
        # Get energy thresholds
        if hasattr(self.config, 'deepspeed') and hasattr(self.config.deepspeed, 'adaptive_training'):
            high_threshold = self.config.deepspeed.adaptive_training.energy_threshold_high
        else:
            high_threshold = 250  # default
        
        # If power usage is too high, switch to BF16 for better energy efficiency
        if avg_power > high_threshold and self.precision_mode == "fp16":
            logger.info(f"Power usage too high ({avg_power:.2f}W), switching to BF16 precision")
            self.args.bf16 = True
            self.args.fp16 = False
            self.precision_mode = "bf16"
            
        # If power usage is manageable, switch back to FP16 for potential speedup
        elif avg_power < high_threshold * 0.8 and self.precision_mode == "bf16":
            logger.info(f"Power usage acceptable ({avg_power:.2f}W), switching to FP16 precision")
            self.args.fp16 = True
            self.args.bf16 = False
            self.precision_mode = "fp16"
    
    def _adapt_layer_freezing(self, model):
        """Dynamically freeze/unfreeze layers based on gradient activity"""
        if not hasattr(model, "named_parameters"):
            return
        
        # Track gradient magnitudes for layers
        layer_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Extract layer name (assuming transformer architecture)
                if "layer" in name:
                    layer_name = name.split(".")[1]  # Get layer number
                    if layer_name not in layer_grads:
                        layer_grads[layer_name] = []
                    layer_grads[layer_name].append(param.grad.abs().mean().item())
        
        # Calculate average gradient magnitude per layer
        layer_avg_grads = {}
        for layer_name, grads in layer_grads.items():
            if grads:
                layer_avg_grads[layer_name] = sum(grads) / len(grads)
        
        # Sort layers by gradient magnitude
        sorted_layers = sorted(layer_avg_grads.items(), key=lambda x: x[1])
        
        # Keep at least min_active_layers unfrozen
        min_active = getattr(self.config.training, 'min_active_layers', 3)
        threshold = getattr(self.config.training, 'layer_activation_threshold', 0.01)
        
        # Identify layers to freeze/unfreeze
        to_freeze = set()
        to_unfreeze = set()
        
        # Unfreeze layers with high gradient activity
        active_count = 0
        for layer_name, grad_mag in reversed(sorted_layers):
            if grad_mag > threshold:
                to_unfreeze.add(layer_name)
                active_count += 1
            if active_count >= min_active:
                break
        
        # Make sure we have minimum active layers
        if active_count < min_active:
            # Unfreeze additional layers with highest gradients
            for layer_name, _ in reversed(sorted_layers):
                if layer_name not in to_unfreeze:
                    to_unfreeze.add(layer_name)
                    active_count += 1
                    if active_count >= min_active:
                        break
        
        # Determine layers to freeze (low gradient activity)
        for layer_name, grad_mag in sorted_layers:
            if grad_mag < threshold and layer_name not in to_unfreeze:
                to_freeze.add(layer_name)
        
        # Apply freezing/unfreezing
        layers_frozen = 0
        layers_unfrozen = 0
        
        for name, param in model.named_parameters():
            if "layer" in name:
                layer_name = name.split(".")[1]
                if layer_name in to_freeze and layer_name not in self.frozen_layers:
                    param.requires_grad = False
                    self.frozen_layers.add(layer_name)
                    layers_frozen += 1
                elif layer_name in to_unfreeze and layer_name in self.frozen_layers:
                    param.requires_grad = True
                    self.frozen_layers.remove(layer_name)
                    layers_unfrozen += 1
        
        if layers_frozen > 0 or layers_unfrozen > 0:
            logger.info(f"Adapted layer freezing: {layers_frozen} layers frozen, {layers_unfrozen} layers unfrozen")
            logger.info(f"Currently frozen layers: {sorted(self.frozen_layers)}")
    
    def log_energy_metrics(self, model, step=None, phase="training"):
        """Log energy metrics for monitoring"""
        if not hasattr(self.energy_monitor, 'stats_history') or not self.energy_monitor.stats_history:
            return
        
        # Create metrics directory if it doesn't exist
        os.makedirs("/app/output/metrics", exist_ok=True)
        
        # Get latest energy stats
        latest_stats = self.energy_monitor.stats_history[-1]
        total_power = sum(stat.power_usage for stat in latest_stats.values())
        
        # Calculate energy per step (approximation)
        # Assuming each step takes about 1 second, adjust as needed
        energy_per_step = total_power  # Joules if power is in Watts and time is 1 second
        
        # Calculate energy efficiency (if we have evaluation metrics)
        energy_efficiency = 0
        if hasattr(self, 'eval_metrics') and 'perplexity' in self.eval_metrics:
            # Initial perplexity (from first evaluation)
            if not hasattr(self, 'initial_perplexity'):
                self.initial_perplexity = self.eval_metrics['perplexity']
                self.energy_consumed_kwh = 0
            
            # Current improvement
            perplexity_improvement = max(0, self.initial_perplexity - self.eval_metrics['perplexity'])
            
            # Update energy consumed (convert from W to kWh assuming 1 second per step)
            self.energy_consumed_kwh += (total_power / 3600000)  # convert W to kWh
            
            # Calculate efficiency
            if self.energy_consumed_kwh > 0:
                energy_efficiency = perplexity_improvement / self.energy_consumed_kwh
        
        # Count active layers
        active_layers = 0
        if hasattr(model, "named_parameters"):
            for name, param in model.named_parameters():
                if "layer" in name and param.requires_grad:
                    layer_name = name.split(".")[1]
                    if layer_name not in self.frozen_layers:
                        active_layers += 1
        
        # Estimate power savings
        power_saved = {}
        if hasattr(self, 'initial_power_usage') and self.initial_power_usage:
            current_power = total_power
            baseline_power = self.initial_power_usage
            
            # Estimate savings from different techniques
            if self.precision_mode == "bf16":
                # BF16 typically saves about 10-20% power compared to FP16
                power_saved["precision"] = baseline_power * 0.15
            
            # Layer freezing savings (rough estimate)
            if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
                total_layers = model.config.num_hidden_layers
                frozen_ratio = len(self.frozen_layers) / max(1, total_layers)
                power_saved["layer_freezing"] = baseline_power * 0.3 * frozen_ratio
        
        # Create metrics object
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "step": step if step is not None else self.state.global_step,
            "model_name": model.config._name_or_path if hasattr(model, "config") else "unknown",
            "phase": phase,
            "energy_per_step": energy_per_step,
            "energy_efficiency": energy_efficiency,
            "total_energy_kwh": self.energy_consumed_kwh,
            "active_layers": active_layers,
            "precision_mode": self.precision_mode,
            "power_saved": power_saved
        }
        
        # Save metrics to file
        metrics_file = f"/app/output/metrics/metrics_{int(datetime.now().timestamp())}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Log summary
        logger.info(f"Energy metrics - Power: {total_power:.1f}W, Mode: {self.precision_mode}, Active layers: {active_layers}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/app/configs/training_config.yaml")
    parser.add_argument("--deepspeed_config", type=str, default="/app/configs/ds_config.json")
    parser.add_argument("--max_wait", type=int, default=600, 
                       help="Maximum time to wait for other nodes in seconds")
    args = parser.parse_args()
    
    # Get rank and world size from environment
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    logger.info(f"Starting up node with rank={rank} in world_size={world_size}")
    logger.info(f"Master: {master_addr}:{master_port}")
    
    # Load configuration
    try:
        config = OmegaConf.load(args.config)
        logger.info(f"Loaded config from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Initialize distributed environment with retry logic
    if world_size > 1:
        logger.info("Setting up distributed training environment")
        success = False
        retry_count = 0
        max_retries = 10
        while not success and retry_count < max_retries:
            success = setup_distributed(rank, world_size, master_addr, master_port)
            if not success:
                logger.warning(f"Distributed setup failed, retrying ({retry_count+1}/{max_retries})")
                time.sleep(10)  # Wait before retrying
                retry_count += 1
        
        if not success:
            logger.error("Failed to initialize distributed training after multiple attempts")
            # Keep the process alive for debugging
            logger.error("Entering idle state for debugging")
            while True:
                time.sleep(60)
    
    # Initialize energy monitor 
    energy_monitor = EnergyMonitor(
        args.deepspeed_config,
        monitoring_interval=config.training.energy_monitoring_interval
    )
    energy_monitor.start()
    
    # Load tokenizer and model
    logger.info(f"Loading model: {config.training.model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.training.model_name_or_path)
        
        # Add padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model configuration
        model_config = AutoConfig.from_pretrained(config.training.model_name_or_path)
        
        # Create model
        model = AutoModelForCausalLM.from_pretrained(
            config.training.model_name_or_path,
            config=model_config
        )
        
        logger.info(f"Model loaded successfully: {type(model)}")
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        if world_size > 1:
            dist.destroy_process_group()
        sys.exit(1)
    
    try:
        # Load dataset
        logger.info(f"Loading dataset: {config.training.dataset_name}")
        dataset = load_dataset(
            config.training.dataset_name,
            config.training.dataset_config_name
        )
        
        # Preprocessing function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=config.training.max_seq_length
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir="/app/output",
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            num_train_epochs=config.training.num_train_epochs,
            warmup_steps=config.training.warmup_steps,
            logging_steps=config.training.logging_steps,
            evaluation_strategy=config.training.evaluation_strategy,
            eval_steps=config.training.eval_steps,
            save_steps=config.training.save_steps,
            fp16=config.training.fp16,
            deepspeed=args.deepspeed_config,
            load_best_model_at_end=True,
            # Add more distributed settings
            local_rank=0,
            dataloader_num_workers=4,
            ddp_find_unused_parameters=False,
        )
        
        # Create trainer
        trainer = AdaptiveTrainer(
            energy_monitor=energy_monitor,
            config=config,
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
        )
        
        logger.info("Starting training")
        trainer.train()
        
        # Save final model
        logger.info("Training completed, saving model")
        trainer.save_model("/app/output/final_model")
        tokenizer.save_pretrained("/app/output/final_model")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
    finally:
        # Clean up
        energy_monitor.stop()
        if world_size > 1:
            dist.destroy_process_group()
        logger.info("Training process exiting")

if __name__ == "__main__":
    main()