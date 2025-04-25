import time
import os
import threading
import json
import pynvml
import logging
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("energy_monitor")

@dataclass
class GPUStats:
    index: int
    power_usage: float  # Watts
    temperature: int    # Celsius
    utilization: int    # Percentage
    memory_used: int    # MB
    memory_total: int   # MB
    energy_efficiency: float  # GFLOPS/Watt (estimated)

class EnergyMonitor:
    def __init__(self, config_path: str, monitoring_interval: int = 60):
        self.monitoring_interval = monitoring_interval
        self.should_run = False
        self.monitor_thread = None
        self.stats_history: List[Dict[int, GPUStats]] = []
        self.config = self._load_config(config_path)
        self._init_nvml()
        
        # Create stats directory
        os.makedirs("/app/energy_stats", exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        try:
            # Try to load as JSON first
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    if config_path.endswith('.json'):
                        config = json.load(f)
                    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        config = yaml.safe_load(f)
                    
                    # Return with all required fields
                    return self._ensure_config_defaults(config)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
        
        # Return default config if loading fails
        logger.warning(f"Using default config, could not load from {config_path}")
        return self._ensure_config_defaults({})
    
    def _ensure_config_defaults(self, config: dict) -> dict:
        """Ensure all required configuration fields exist with defaults"""
        if 'adaptive_training' not in config:
            config['adaptive_training'] = {}
            
        adaptive_training = config['adaptive_training']
        
        # Set defaults for missing fields
        if 'enabled' not in adaptive_training:
            adaptive_training['enabled'] = True
        if 'energy_threshold_high' not in adaptive_training:
            adaptive_training['energy_threshold_high'] = 250
        if 'energy_threshold_low' not in adaptive_training:
            adaptive_training['energy_threshold_low'] = 150
        if 'min_thread_percentage' not in adaptive_training:
            adaptive_training['min_thread_percentage'] = 30
        if 'max_thread_percentage' not in adaptive_training:
            adaptive_training['max_thread_percentage'] = 80
            
        if 'mps' not in config:
            config['mps'] = {}
            
        if 'enabled' not in config['mps']:
            config['mps']['enabled'] = True
        if 'initial_thread_percentage' not in config['mps']:
            config['mps']['initial_thread_percentage'] = 50
            
        return config
    
    def _init_nvml(self):
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        logger.info(f"Found {self.device_count} GPU devices")
        
    def _get_gpu_stats(self) -> Dict[int, GPUStats]:
        stats = {}
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Estimate energy efficiency (GFLOPS/Watt) - simplified calculation
            # In a real implementation, you'd use a more accurate method
            energy_efficiency = (util.gpu * 0.1) / max(power_usage, 1)  
            
            stats[i] = GPUStats(
                index=i,
                power_usage=power_usage,
                temperature=temp,
                utilization=util.gpu,
                memory_used=mem_info.used // (1024 * 1024),  # Convert to MB
                memory_total=mem_info.total // (1024 * 1024),  # Convert to MB
                energy_efficiency=energy_efficiency
            )
        return stats
    
    def _adjust_mps_settings(self, stats: Dict[int, GPUStats]):
        """Adjust MPS thread allocation based on energy efficiency metrics"""
        for gpu_idx, gpu_stat in stats.items():
            current_efficiency = gpu_stat.energy_efficiency
            high_threshold = self.config['adaptive_training']['energy_threshold_high']
            low_threshold = self.config['adaptive_training']['energy_threshold_low']
            
            # Current MPS thread percentage
            try:
                with open(f"/tmp/nvidia-mps/gpu{gpu_idx}_thread_percentage", "r") as f:
                    current_percentage = int(f.read().strip())
            except:
                current_percentage = self.config['mps']['initial_thread_percentage']
            
            new_percentage = current_percentage
            
            # Power too high - reduce threads
            if gpu_stat.power_usage > high_threshold:
                new_percentage = max(
                    self.config['adaptive_training']['min_thread_percentage'],
                    current_percentage - 5
                )
                logger.info(f"GPU {gpu_idx} power too high ({gpu_stat.power_usage}W), reducing threads to {new_percentage}%")
            
            # Power too low - increase threads for better utilization
            elif gpu_stat.power_usage < low_threshold and gpu_stat.utilization < 70:
                new_percentage = min(
                    self.config['adaptive_training']['max_thread_percentage'],
                    current_percentage + 5
                )
                logger.info(f"GPU {gpu_idx} power too low ({gpu_stat.power_usage}W), increasing threads to {new_percentage}%")
            
            # Update MPS thread percentage if changed
            if new_percentage != current_percentage:
                try:
                    with open(f"/tmp/nvidia-mps/gpu{gpu_idx}_thread_percentage", "w") as f:
                        f.write(str(new_percentage))
                except Exception as e:
                    logger.error(f"Failed to update MPS thread percentage: {e}")
    
    def _monitoring_loop(self):
        while self.should_run:
            try:
                stats = self._get_gpu_stats()
                self.stats_history.append(stats)
                
                # Save current stats to file
                timestamp = int(time.time())
                with open(f"energy_stats/gpu_stats_{timestamp}.json", 'w') as f:
                    # Convert dataclass objects to dict for JSON serialization
                    serializable_stats = {
                        idx: {k: v for k, v in stat.__dict__.items()}
                        for idx, stat in stats.items()
                    }
                    json.dump(serializable_stats, f, indent=2)
                
                # Adjust MPS settings based on energy metrics
                if self.config['adaptive_training']['enabled']:
                    self._adjust_mps_settings(stats)
                
                # Print summary
                total_power = sum(stat.power_usage for stat in stats.values())
                avg_util = sum(stat.utilization for stat in stats.values()) / len(stats)
                logger.info(f"Total power: {total_power:.2f}W, Avg GPU util: {avg_util:.1f}%")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def start(self):
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            logger.warning("Monitor is already running")
            return
        
        self.should_run = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Energy monitoring started")
    
    def stop(self):
        self.should_run = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Energy monitoring stopped")
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    # Example usage
    monitor = EnergyMonitor("/app/configs/training_config.json", monitoring_interval=30)
    monitor.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()