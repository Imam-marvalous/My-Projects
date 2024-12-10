import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def simulate_congestion(bandwidth, rtt=100, packet_loss=0.01, simulation_time=100):
    """
    Simulate congestion based on bandwidth, RTT, and packet loss.
    Args:
        bandwidth (float): Bandwidth in Mbps.
        rtt (float): Round-trip time in ms (default: 100ms).
        packet_loss (float): Packet loss rate (default: 1%).
        simulation_time (int): Simulation time in seconds (default: 100).
    
    Returns:
        pandas.DataFrame: DataFrame containing time and congestion values.
    """
    # Generate time steps for the simulation
    time = np.linspace(0, simulation_time, num=20)
    
    # Enhanced congestion model
    base_congestion = bandwidth / (rtt * (1 - packet_loss) + 1)
    random_variation = np.random.normal(1, 0.1, len(time))
    congestion = base_congestion * np.exp(-0.05 * time) * random_variation
    
    # Calculate throughput (as percentage of bandwidth)
    throughput = (congestion / bandwidth) * 100
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Time (s)': time.round(1),
        'Congestion Level': congestion.round(2),
        'Throughput (%)': throughput.round(1),
        'Effective Bandwidth (Mbps)': (bandwidth * throughput / 100).round(2)
    })
    return df

# Set page config
st.set_page_config(layout="wide")

# App title and description
st.title("Network Congestion Simulator")
st.markdown("""
This simulation visualizes network congestion patterns and their impact on effective bandwidth. 
Adjust the parameters below to see how they affect network performance.
""")

# Input parameters
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    bandwidth = st.slider("Bandwidth (Mbps)", 1, 100, 10)
with col2:
    rtt = st.slider("RTT (ms)", 10, 500, 100)
with col3:
    packet_loss = st.slider("Packet Loss Rate (%)", 0, 20, 1) / 100

# Simulate the network congestion
df = simulate_congestion(bandwidth, rtt, packet_loss)

# Create 3D grid for surface plot
time_steps = np.linspace(0, 100, num=20)
packet_loss_range = np.linspace(0, 20, 10) / 100  # Example range for visualization
time_grid, loss_grid = np.meshgrid(time_steps, packet_loss_range)

# Calculate congestion for the grid
congestion_grid = bandwidth / (rtt * (1 - loss_grid) + 1) * np.exp(-0.05 * time_grid)

# Visualization section
viz_col1, viz_col2 = st.columns([2, 1])
with viz_col1:
    # Create the 3D surface plot
    fig = go.Figure(data=[go.Surface(z=congestion_grid, x=time_grid, y=loss_grid, 
                                     colorscale='Viridis')])
    # Customize layout
    fig.update_layout(
        title="3D Surface Plot of Network Congestion",
        scene=dict(
            xaxis_title="Time (s)",
            yaxis_title="Packet Loss Rate (%)",
            zaxis_title="Congestion Level"
        )
    )
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

with viz_col2:
    st.subheader("Performance Metrics")
    avg_throughput = df['Throughput (%)'].mean()
    avg_bandwidth = df['Effective Bandwidth (Mbps)'].mean()
    st.metric("Average Throughput", f"{avg_throughput:.1f}%")
    st.metric("Average Effective Bandwidth", f"{avg_bandwidth:.1f} Mbps")
    st.metric("Packet Loss", f"{packet_loss*100:.1f}%")

# Display the data table
st.subheader("Detailed Network Performance Data")
st.dataframe(
    df.style.format({
        'Time (s)': '{:.1f}',
        'Congestion Level': '{:.2f}',
        'Throughput (%)': '{:.1f}',
        'Effective Bandwidth (Mbps)': '{:.2f}'
    }).background_gradient(
        subset=['Throughput (%)'], 
        cmap='YlOrRd'
    ),
    height=400
)

# Explanatory notes
st.markdown("""
### Understanding the Metrics:
- **Throughput (%)**: Percentage of maximum bandwidth currently being utilized
- **Effective Bandwidth**: Actual data transfer rate accounting for congestion
- **Congestion Level**: Measure of network congestion (lower is better)

The graph shows the congestion level as a 3D surface plot. Adjusting the parameters will dynamically update the visualization.
""")

