import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def data_generator(num_points):
    '''
    Generates random data in a sine wave pattern with random noise and anomalies.
    Args:
      num_points (int): The number of data points to generate.
    Returns:
      data_stream (numpy array): Generated data in a sine wave pattern with noise and anomalies.
    '''
    time = np.arange(num_points)
    sine_wave_data = 10 * np.sin(2 * np.pi * time / 100)
    random_noise = np.random.normal(0, 1, num_points)
    data_stream = sine_wave_data + random_noise

    for i in range(num_points):
        if random.random() < 0.01:
            data_stream[i] += np.random.choice([-1, 1]) * random.uniform(50, 100)
    
    return data_stream

def anomaly_detection(data, alpha, z_value):
    '''
    Detects anomalies based on the moving average.
    Args:
      data (numpy array): Input data stream/points.
      alpha (float): Smoothing factor for the weighted average.
      z_value (float): Value for the sensitivity in anomaly detection.
    Returns:
      data_stream (list): List of weighted values based on alpha.
      anomalies (list): List of detected anomalies.
    '''
    data_stream = [data[0]]
    anomalies = []
    
    for i in range(1, len(data)):
        moving_average = alpha * data[i] + (1 - alpha) * data_stream[-1]
        data_stream.append(moving_average)
        deviation = data[i] - moving_average

        if np.abs(deviation) > z_value:
            anomalies.append((i, data[i]))
            
    return data_stream, anomalies

def plot(data, moving_average, anomalies):
    '''
    Plots the data stream, moving average and detected anomalies.
    Args:
        data (numpy array): Input data stream/points to be plotted.
        moving_average (list): List of weight values based on the specified alpha.
        anomalies (list): List of detected anomalies.
    Returns:
        None: Displays the plot with the data, moving average and anomalies (not in realtime).
    '''
    fig, ax = plt.subplots()
    
    ax.plot(data, 'b-', label='Data Stream')
    ax.plot(moving_average, 'y-', label='Moving Average')
    
    if anomalies:
        x_anomalies, y_anomalies = zip(*anomalies)
        ax.scatter(x_anomalies, y_anomalies, color='red', label='Anomalies')
    
    ax.set_xlim(0, len(data))
    ax.set_ylim(-120, 120)
    ax.legend()
    plt.show()

def realtime_visualization(data, moving_average, anomalies):
    '''
    Provides a real-time visualization plotting of the data, moving_average and anomalies detected.
    Args:
        data (numpy array): Input data stream/points to be plotted.
        moving_average (list): List of weight values based on the specified alpha.
        anomalies (list): List of detected anomalies.    
    Returns: 
        None: Displays a real-time visualization plot of the generated data and anomalies.
    '''
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    data_line, = plt.plot([], [], 'b-', label='Data Stream')
    moving_average_line, = plt.plot([], [], 'y-', label='Moving Average')
    anomaly_scatter = ax.scatter([], [], color='red', label='Anomalies')

    def init():
        ax.set_xlim(0, len(data))
        ax.set_ylim(-120, 120)
        return data_line, moving_average_line, anomaly_scatter

    def update(frame):
        xdata.append(frame)
        ydata.append(data[frame])
        
        if len(xdata) == len(ydata):
            data_line.set_data(xdata, ydata)
        
        moving_average_data = moving_average[:frame + 1]
        if len(xdata) == len(moving_average_data):
            moving_average_line.set_data(xdata, moving_average_data)
        
        anomaly_points = [(idx, val) for idx, val in anomalies if idx <= frame]
        if anomaly_points:
            x_anom, y_anom = zip(*anomaly_points)
            anomaly_scatter.set_offsets(np.c_[x_anom, y_anom])
        
        return data_line, moving_average_line, anomaly_scatter

    ani = animation.FuncAnimation(fig, update, frames=len(data), init_func=init, blit=True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data = data_generator(200)
    moving_average, anomalies = anomaly_detection(data, 0.1, 8)
    
    #Plot without realtime visualization
    #plot(data, moving_average, anomalies)
    
    realtime_visualization(data, moving_average, anomalies)
