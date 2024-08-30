using PyPlot
using DelimitedFiles
# Function to read data from a single file
function read_data(filename::String)
    lines = read(filename)

    # Initialize arrays for x values and y values
    x_values = []
    y_values = []

    # Process each line starting from the second line
    for i in 2:size(lines, 1)
        numbers = lines[i, :]
        push!(x_values, numbers[1] - 6.2285)
        push!(y_values, numbers[2] + numbers[3])
    end

    return x_values, y_values
end

# Function to plot data from multiple files
function plot_multiple_files(filenames::Vector{String}, output_filename::String; vline_x::Union{Float64, Nothing}=nothing)
    figure()  # Initialize a new figure
    
    integrals = []
    for (i, filename) in enumerate(filenames)
        # Read data from each file
        x_values, y_values = read_data(filename)
        
        # Plot the series
        plot(x_values, y_values, label="File $(i + 1), Series 1")
    end

    # Add vertical dashed red line if specified
    if vline_x !== nothing
        axvline(x=vline_x, color="red", linestyle="--", linewidth=1, label="Vertical Line at $vline_x")
    end

    # Add labels, title, and legend (optional)
    xlabel("X Values")
    ylabel("Y Values")
    title("Combined Plot of Multiple Files with Vertical Line")
    legend()

    # Save the figure without showing it
    savefig(output_filename, dpi = 300)
    close()
end

# List of filenames to read from
filenames = ["/home/yongjoong/hubbardu_new_funciton_0702/DFTK.jl/examples/qe_examples/Si.pdos_atm#1(Si)_wfc#1(s)",
             "/home/yongjoong/hubbardu_new_funciton_0702/DFTK.jl/examples/qe_examples/Si.pdos_atm#1(Si)_wfc#2(p)"]

# X-coordinate for the vertical dashed red line
vertical_line_x = 0.0  # Replace with your desired x-coordinate

# Plot data from the specified files and save the figure
plot_multiple_files(filenames, "plot_Si_nosym.png", vline_x=vertical_line_x)
