import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    time = np.linspace(0, simulation_time, num=20)  # Reduced number of points for clearer table view
    
    # Enhanced congestion model with added randomness for more realistic behavior
    base_congestion = bandwidth / (rtt * (1 - packet_loss) + 1)
    random_variation = np.random.normal(1, 0.1, len(time))  # Add some random variation
    congestion = base_congestion * np.exp(-0.05 * time) * random_variation
    
    # Calculate throughput (as percentage of bandwidth)
    throughput = (congestion / bandwidth) * 100
    
    # Create a DataFrame with rounded values for better display
    df = pd.DataFrame({
        'Time (s)': time.round(1),
        'Congestion Level': congestion.round(2),
        'Throughput (%)': throughput.round(1),
        'Effective Bandwidth (Mbps)': (bandwidth * throughput/100).round(2)
    })
    return df

# Set page config
st.set_page_config(layout="wide")

# Streamlit app title and description
st.title("Network Congestion Simulator")
st.markdown("""
This enhanced simulation visualizes network congestion patterns and their impact on effective bandwidth. 
Adjust the parameters below to see how they affect network performance.
""")

# Create two columns for controls
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    bandwidth = st.slider("Bandwidth (Mbps)", 1, 100, 10)
with col2:
    rtt = st.slider("RTT (ms)", 10, 500, 100)
with col3:
    packet_loss = st.slider("Packet Loss Rate (%)", 0, 20, 1) / 100

# Simulate the network congestion
df = simulate_congestion(bandwidth, rtt, packet_loss)

# Create two columns for visualization
viz_col1, viz_col2 = st.columns([2, 1])

with viz_col1:
    # Create and customize the plot
    plt.style.use('fivethirtyeight')  # Using a built-in style
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot multiple metrics
    ax.plot(df['Time (s)'], df['Throughput (%)'], 
            label='Throughput %', 
            color='#2E86C1',  # Nice blue color
            linewidth=2)
    ax.plot(df['Time (s)'], df['Effective Bandwidth (Mbps)'], 
            label='Effective Bandwidth', 
            color='#28B463',  # Nice green color
            linewidth=2, 
            linestyle='--')
    
    # Add confidence interval
    ax.fill_between(df['Time (s)'], 
                   df['Throughput (%)'] * 0.9, 
                   df['Throughput (%)'] * 1.1, 
                   color='#2E86C1', 
                   alpha=0.1)
    
    # Customize the plot appearance
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Network Metrics", fontsize=10)
    ax.set_title("Network Performance Over Time", fontsize=12, pad=15)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()
    
    # Display the graph
    st.pyplot(fig)

with viz_col2:
    st.subheader("Performance Metrics")
    
    # Calculate summary statistics
    avg_throughput = df['Throughput (%)'].mean()
    avg_bandwidth = df['Effective Bandwidth (Mbps)'].mean()
    
    # Display metrics
    st.metric("Average Throughput", f"{avg_throughput:.1f}%")
    st.metric("Average Effective Bandwidth", f"{avg_bandwidth:.1f} Mbps")
    st.metric("Packet Loss", f"{packet_loss*100:.1f}%")

# Display the data table with formatting
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

# Add explanatory notes
st.markdown("""
### Understanding the Metrics:
- *Throughput (%)*: Percentage of maximum bandwidth currently being utilized
- *Effective Bandwidth*: Actual data transfer rate accounting for congestion
- *Congestion Level*: Measure of network congestion (lower is better)

The graph shows both the throughput percentage (solid blue line) and effective bandwidth (dashed green line) over time. 
The light blue shaded area represents the uncertainty range in the measurements.
""")
