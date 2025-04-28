import matplotlib.pyplot as plt
import numpy as np

# Data from the user
n_values = [1, 2, 3, 4, 5]
success_rates = [0.3888888889, 0.1081081081, 0.09523809524, 0.05405405405, 0]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(n_values, success_rates, marker='o', linestyle='-', color='blue', linewidth=2, markersize=8)

# Add labels and title
plt.xlabel('n', fontsize=14)
plt.ylabel('Success Rate', fontsize=14)
plt.title('Success Rate vs n', fontsize=16)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Add data labels
for i, (x, y) in enumerate(zip(n_values, success_rates)):
    plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=10)

# Set y-axis to start from 0
plt.ylim(bottom=0)

# Customize x-axis ticks to show all n values
plt.xticks(n_values)

# Save the plot
plt.tight_layout()
plt.savefig('success_rate_vs_n.png')

# Show the plot
plt.show()