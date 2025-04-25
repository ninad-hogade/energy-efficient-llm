#!/bin/bash
# scripts/deploy-monitoring.sh

# Create monitoring namespace
microk8s kubectl create namespace monitoring

# Apply DCGM Exporter
microk8s kubectl apply -f kubernetes/dcgm-exporter.yaml

# Apply Prometheus
microk8s kubectl apply -f kubernetes/prometheus.yaml

# Apply Grafana and dashboards
microk8s kubectl apply -f kubernetes/grafana.yaml
microk8s kubectl apply -f kubernetes/energy-dashboard.yaml

# Wait for Grafana to start
echo "Waiting for Grafana to start..."
sleep 30

# Get Grafana service IP
GRAFANA_IP=$(microk8s kubectl get svc grafana -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "Monitoring stack deployed!"
echo "Access Grafana at http://$GRAFANA_IP:3000"
echo "Default login: admin/admin"