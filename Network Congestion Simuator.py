import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to simulate network congestion
def simulate_congestion(bandwidth, packet_size, delay, duration):
    """
    Simulate network congestion and return congestion metrics over time.

    Args:
        bandwidth (int): Network bandwidth in Mbps.
        packet_size (int): Size of packets in bytes.
        delay (float): Network delay in milliseconds.
        duration (int): Duration of simulation in seconds.

    Returns:
        pd.DataFrame: A dataframe with simulated congestion data over time.
    """
    time = np.arange(0, duration, 0.1)  # Simulating in 0.1s intervals
    num_packets_sent = (bandwidth * 1e6 * time) / (packet_size * 8)  # Mbps to bps conversion
    congestion = (num_packets_sent * packet_size) / bandwidth  # A simple congestion calculation

    # Simulate delay effect
    congestion += (delay / 1000) * np.sin(0.1 * time)  # Sinusoidal fluctuation to simulate delays
    congestion = np.clip(congestion, 0, 1)  # Normalize between 0 and 1

    # Create a DataFrame to hold the simulated data
    df = pd.DataFrame({
        'Time (s)': time,
        'Congestion Level': congestion
    })
    return df

# Streamlit UI components
def display_ui():
    st.title('Network Congestion Simulator')

    # User Inputs
    bandwidth = st.slider('Network Bandwidth (Mbps)', min_value=1, max_value=100, value=10, step=1)
    packet_size = st.slider('Packet Size (bytes)', min_value=100, max_value=1500, value=512, step=10)
    delay = st.slider('Network Delay (ms)', min_value=1, max_value=500, value=50, step=1)
    duration = st.slider('Simulation Duration (s)', min_value=10, max_value=300, value=60, step=10)

    st.write(f"Simulating with: Bandwidth = {bandwidth} Mbps, Packet Size = {packet_size} bytes, "
             f"Delay = {delay} ms, Duration = {duration} seconds")

    # Simulate congestion
    df = simulate_congestion(bandwidth, packet_size, delay, duration)

    # Display Results
    st.subheader('Congestion Simulation Data')
    st.write(df)

    # Plot the congestion over time
    st.subheader('Congestion Level Over Time')
    fig, ax = plt.subplots()
    ax.plot(df['Time (s)'], df['Congestion Level'], label='Congestion Level')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Congestion Level')
    ax.set_title('Network Congestion Over Time')
    ax.legend()

    st.pyplot(fig)

# Run the Streamlit app
if __name__ == '__main__':
    display_ui()
