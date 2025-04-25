#!/bin/bash

# Uninstall the current NVIDIA device plugin with MPS configuration
echo "Removing existing NVIDIA device plugin with MPS configuration..."
microk8s kubectl delete -n nvidia-device-plugin daemonset nvdp-nvidia-device-plugin
microk8s helm3 uninstall nvdp -n nvidia-device-plugin

# Get all node names
NODES=$(microk8s kubectl get nodes -o jsonpath='{.items[*].metadata.name}')

# Remove MPS labels from all nodes
echo "Removing MPS-related labels from nodes..."
for NODE in $NODES; do
  echo "Cleaning labels on node $NODE"
  microk8s kubectl label node $NODE nvidia.com/device-plugin.config- nvidia.com/gpu.replicas- nvidia.com/mps.capable- --overwrite
done

# Add NVIDIA device plugin repository (in case it's needed)
echo "Adding NVIDIA device plugin repository..."
microk8s helm3 repo add nvdp https://nvidia.github.io/k8s-device-plugin
microk8s helm3 repo update

# Install NVIDIA device plugin with STANDARD configuration (no MPS)
echo "Installing standard NVIDIA device plugin (without MPS)..."
microk8s helm3 install nvdp nvdp/nvidia-device-plugin \
  --version=0.17.1 \
  --namespace nvidia-device-plugin \
  --create-namespace \
  --set gfd.enabled=true \
  --set config.default=default  # Explicitly specify default config

echo "Waiting for the device plugin to restart and update node labels..."
sleep 30

# Verify the configuration
echo "Checking GPU resources after disabling MPS:"
microk8s kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.capacity.nvidia\\.com/gpu

# Verify that MPS-related labels are removed
echo "Checking for MPS labels:"
microk8s kubectl get node --output=json | jq '.items[].metadata.labels' | grep -E "mps|SHARED|replicas" || echo "No MPS labels found - successfully disabled"

echo ""
echo "If you still see MPS labels, you may need to restart the microk8s service with:"
echo "sudo systemctl restart snap.microk8s.daemon-containerd"
echo "sudo systemctl restart snap.microk8s.daemon-kubelet"
echo ""
echo "And then wait 1-2 minutes for the changes to propagate."
