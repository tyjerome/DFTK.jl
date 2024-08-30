import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps

# Function to read data from a single file
def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Initialize lists for x values and y values
    x_values = []
    y_values = []

    # Process each line starting from the second line
    for line in lines[1:]:
        numbers = list(map(float, line.split()))
        x_values.append(numbers[0]-6.0341) #6.2285)
        y_values.append(numbers[1]+numbers[2])
    
    return x_values, y_values

# Function to plot data from multiple files
def plot_multiple_files(filenames, output_filename, vline_x=None):
    plt.figure()  # Initialize a new figure
    
    integrals = []
    for i, filename in enumerate(filenames):
        # Read data from each file
        x_values, y_values = read_data(filename)
        # Transpose y_data to get each series in its own list
        #y_series = list(map(list, zip(*y_data)))

        # Calculate the integral using Simpson's rule
        integral = simps(y_values, x=x_values)
        integrals.append(integral)
        # Plot each series
        #for j, y_values in enumerate(y_series):
            #plt.plot(x_values, y_values, label=f'File {i + 1}, Series {j + 1}')
        plt.plot(x_values, y_values, label=f'File {filename}, Series 1 (Integral: {integral:.2f})')

    # Add vertical dashed red line if specified
    if vline_x is not None:
        plt.axvline(x=vline_x, color='red', linestyle='--', linewidth=1, label=f'Vertical Line at {vline_x}')

    # Add labels, title, and legend (optional)
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.title('Combined Plot of Multiple Files with Vertical Line')
    plt.legend()

    # Save the figure without showing it
    plt.savefig(output_filename)
    plt.close()

# List of filenames to read from
filenames = ['Si.pdos_atm#1(Si)_wfc#2(p)_nosym_noinv','Si.pdos_atm#1(Si)_wfc#2(p)_noinv']

# X-coordinate for the vertical dashed red line
vertical_line_x = 0  # Replace with your desired x-coordinate

# Plot data from the specified files and save the figure
plot_multiple_files(filenames, 'plot_Si_pdos_2p', vline_x=vertical_line_x)