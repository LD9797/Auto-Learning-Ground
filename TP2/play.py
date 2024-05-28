import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a plot
fig, ax = plt.subplots()

# Plot data
ax.plot(x, y)

# Set custom ticks and labels
custom_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
custom_labels = ['0', 'π/2', 'π', '3π/2', '2π']
ax.set_xticks(custom_ticks)  # Set the positions for the ticks
ax.set_xticklabels(custom_labels)  # Set the custom labels for the ticks

# Optionally, remove the x-axis line while keeping the labels
# ax.spines['bottom'].set_visible(False)  # This hides the bottom spine (axis line)

# Show the plot
plt.show()
