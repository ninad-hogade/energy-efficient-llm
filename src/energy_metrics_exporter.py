# src/energy_metrics_exporter.py
import time
import threading
from prometheus_client import start_http_server, Gauge, Counter
import json
import glob
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("metrics_exporter")

class EnergyMetricsExporter:
    def __init__(self, port=8000, monitoring_interval=10):
        self.port = port
        self.monitoring_interval = monitoring_interval
        self.should_run = False
        self.monitor_thread = None
        
        # Define Prometheus metrics
        self.energy_per_step = Gauge('training_energy_per_step', 
                                     'Energy consumption per training step (Joules)',
                                     ['model', 'phase'])
        
        self.energy_efficiency = Gauge('training_energy_efficiency',
                                      'Training energy efficiency (perplexity improvement/kWh)',
                                      ['model', 'phase'])
        
        self.total_energy = Counter('training_total_energy',
                                   'Total energy consumed during training (kWh)',
                                   ['model'])
        
        self.power_saved = Gauge('training_power_saved',
                                'Estimated power saved compared to baseline (Watts)',
                                ['technique'])
        
        self.active_layers = Gauge('training_active_layers',
                                  'Number of active (unfrozen) layers',
                                  ['model'])
        
        self.precision_mode = Gauge('training_precision_mode',
                                   'Current precision mode (0=FP32, 1=FP16, 2=BF16)',
                                   ['model'])
        
        self.mps_thread_percentage = Gauge('training_mps_thread_percentage',
                                          'CUDA MPS thread percentage',
                                          ['gpu'])
    
    def read_training_metrics(self):
        """Read the latest training metrics from files"""
        try:
            # Read training metrics if they exist
            metrics_files = sorted(glob.glob("/app/output/metrics/*.json"))
            if metrics_files:
                latest_file = metrics_files[-1]
                with open(latest_file, 'r') as f:
                    metrics = json.load(f)
                
                # Update Prometheus metrics
                model_name = metrics.get('model_name', 'unknown')
                phase = metrics.get('phase', 'training')
                
                # Energy metrics
                if 'energy_per_step' in metrics:
                    self.energy_per_step.labels(model=model_name, phase=phase).set(
                        metrics['energy_per_step'])
                
                if 'energy_efficiency' in metrics:
                    self.energy_efficiency.labels(model=model_name, phase=phase).set(
                        metrics['energy_efficiency'])
                
                if 'total_energy_kwh' in metrics:
                    # Set counter to absolute value
                    self.total_energy.labels(model=model_name)._value.set(
                        metrics['total_energy_kwh'])
                
                # Training configuration metrics
                if 'active_layers' in metrics:
                    self.active_layers.labels(model=model_name).set(
                        metrics['active_layers'])
                
                if 'precision_mode' in metrics:
                    mode_value = {'fp32': 0, 'fp16': 1, 'bf16': 2}.get(
                        metrics['precision_mode'].lower(), 0)
                    self.precision_mode.labels(model=model_name).set(mode_value)
                
                # Power saving estimates
                for technique, value in metrics.get('power_saved', {}).items():
                    self.power_saved.labels(technique=technique).set(value)
        
        except Exception as e:
            logger.error(f"Error reading training metrics: {e}")
    
    def read_mps_thread_percentages(self):
        """Read MPS thread percentages for each GPU"""
        try:
            mps_dir = "/tmp/nvidia-mps"
            if os.path.exists(mps_dir):
                for filename in os.listdir(mps_dir):
                    if filename.startswith("gpu") and "_thread_percentage" in filename:
                        gpu_idx = filename.replace("gpu", "").split("_")[0]
                        with open(os.path.join(mps_dir, filename), "r") as f:
                            try:
                                percentage = int(f.read().strip())
                                self.mps_thread_percentage.labels(gpu=gpu_idx).set(percentage)
                            except ValueError:
                                logger.warning(f"Invalid thread percentage in {filename}")
        except Exception as e:
            logger.error(f"Error reading MPS thread percentages: {e}")
    
    def _monitoring_loop(self):
        while self.should_run:
            try:
                # Read and update all metrics
                self.read_training_metrics()
                self.read_mps_thread_percentages()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def start(self):
        """Start the Prometheus HTTP server and monitoring thread"""
        # Start Prometheus HTTP server
        start_http_server(self.port)
        logger.info(f"Prometheus metrics server started on port {self.port}")
        
        # Start monitoring thread
        self.should_run = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Metrics monitoring started")
    
    def stop(self):
        """Stop the monitoring thread"""
        self.should_run = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Metrics monitoring stopped")

if __name__ == "__main__":
    exporter = EnergyMetricsExporter(port=8000, monitoring_interval=10)
    exporter.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        exporter.stop()