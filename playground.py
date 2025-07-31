import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file into a pandas DataFrame
# Replace 'your_loss_data.csv' with the actual path to your CSV file
# Ensure your CSV has a column for epoch/iteration and a column for loss values
df = pd.read_csv('output/loss_per_epoch.csv')

# Assuming your CSV has columns named 'epoch' and 'loss'
# Adjust these column names if your CSV uses different names
epochs = df['epoch']
loss_values = df['loss']

# Plotting the loss curve
plt.figure(figsize=(10, 6))  # Optional: Adjust figure size
plt.plot(epochs, loss_values, label='Training Loss', color='blue')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.yscale('log')  # Set the y-axis to a logarithmic scale
tick_locations = range(1, len(df['epoch']) + 1, 1)
plt.xticks(tick_locations)
plt.grid(True)  # Optional: Add a grid for better readability
plt.legend()
plt.savefig('output/loss_curve.png')
plt.close()
