import os
import json
import glob
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Energy-Efficient LLM Training Dashboard"),
    
    html.Div([
        html.Div([
            html.H3("Energy Metrics"),
            dcc.Graph(id='power-graph'),
            dcc.Graph(id='efficiency-graph'),
        ], className="six columns"),
        
        html.Div([
            html.H3("GPU Utilization"),
            dcc.Graph(id='utilization-graph'),
            dcc.Graph(id='memory-graph'),
        ], className="six columns"),
    ], className="row"),
    
    dcc.Interval(
        id='interval-component',
        interval=10*1000,  # in milliseconds (10 seconds)
        n_intervals=0
    )
])

@app.callback(
    [Output('power-graph', 'figure'),
     Output('efficiency-graph', 'figure'),
     Output('utilization-graph', 'figure'),
     Output('memory-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n):
    # Get list of all stats files
    stats_files = sorted(glob.glob("energy_stats/gpu_stats_*.json"))
    
    # Limit to last 100 files for performance
    stats_files = stats_files[-100:]
    
    if not stats_files:
        # Create empty figures if no data
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available")
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    # Process all files
    timestamps = []
    power_data = []
    efficiency_data = []
    utilization_data = []
    memory_data = []
    
    for file_path in stats_files:
        # Extract timestamp from filename
        file_name = os.path.basename(file_path)
        timestamp = int(file_name.replace("gpu_stats_", "").replace(".json", ""))
        datetime_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        timestamps.append(datetime_str)
        
        # Load data
        with open(file_path, 'r') as f:
            stats = json.load(f)
        
        # Process each GPU's data
        gpu_power = {}
        gpu_efficiency = {}
        gpu_utilization = {}
        gpu_memory = {}
        
        for gpu_idx, gpu_data in stats.items():
            gpu_idx_str = f"GPU {gpu_idx}"
            
            # Collect metrics
            gpu_power[gpu_idx_str] = gpu_data["power_usage"]
            gpu_efficiency[gpu_idx_str] = gpu_data["energy_efficiency"]
            gpu_utilization[gpu_idx_str] = gpu_data["utilization"]
            
            memory_used = gpu_data["memory_used"]
            memory_total = gpu_data["memory_total"]
            memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
            gpu_memory[gpu_idx_str] = memory_percent
        
        power_data.append(gpu_power)
        efficiency_data.append(gpu_efficiency)
        utilization_data.append(gpu_utilization)
        memory_data.append(gpu_memory)
    
    # Create DataFrames
    power_df = pd.DataFrame(power_data, index=timestamps)
    efficiency_df = pd.DataFrame(efficiency_data, index=timestamps)
    utilization_df = pd.DataFrame(utilization_data, index=timestamps)
    memory_df = pd.DataFrame(memory_data, index=timestamps)
    
    # Create figures
    power_fig = go.Figure()
    efficiency_fig = go.Figure()
    utilization_fig = go.Figure()
    memory_fig = go.Figure()
    
    # Add traces for each GPU
    for gpu in power_df.columns:
        power_fig.add_trace(go.Scatter(x=power_df.index, y=power_df[gpu], mode='lines', name=f"{gpu} Power (W)"))
        efficiency_fig.add_trace(go.Scatter(x=efficiency_df.index, y=efficiency_df[gpu], mode='lines', name=f"{gpu} Efficiency"))
        utilization_fig.add_trace(go.Scatter(x=utilization_df.index, y=utilization_df[gpu], mode='lines', name=f"{gpu} Util (%)"))
        memory_fig.add_trace(go.Scatter(x=memory_df.index, y=memory_df[gpu], mode='lines', name=f"{gpu} Mem (%)"))
    
    # Update layouts
    power_fig.update_layout(
        title="GPU Power Usage Over Time",
        xaxis_title="Time",
        yaxis_title="Power (Watts)",
        legend_title="GPUs"
    )
    
    efficiency_fig.update_layout(
        title="GPU Energy Efficiency Over Time",
        xaxis_title="Time",
        yaxis_title="Efficiency (Higher is better)",
        legend_title="GPUs"
    )
    
    utilization_fig.update_layout(
        title="GPU Utilization Over Time",
        xaxis_title="Time",
        yaxis_title="Utilization (%)",
        legend_title="GPUs"
    )
    
    memory_fig.update_layout(
        title="GPU Memory Usage Over Time",
        xaxis_title="Time",
        yaxis_title="Memory Usage (%)",
        legend_title="GPUs"
    )
    
    return power_fig, efficiency_fig, utilization_fig, memory_fig

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)