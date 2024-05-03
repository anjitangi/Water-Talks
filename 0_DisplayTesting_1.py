import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import cv2

# Load the tensors
predictions_tensor = torch.load('predictions100.pth')
labels_tensor = torch.load('labels100.pth')

# Convert tensors to numpy arrays
predictions_np = predictions_tensor.numpy()
labels_np = labels_tensor.numpy()

# Prepare the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
cmap = 'hot'  # Heatmap color scheme

# Initialize with the first frame
pred_img = ax1.imshow(predictions_np[0, 0], cmap=cmap)
label_img = ax2.imshow(labels_np[0, 0], cmap=cmap)
ax1.title.set_text('Prediction')
ax2.title.set_text('Label')

def update(frame_number):
    # Scaling and clipping
    scaled_predictions = predictions_np[frame_number, 0] * 10  # Scale the predictions
    np.clip(scaled_predictions, 0, 255, out=scaled_predictions)  # Clip the scaled predictions

    scaled_labels = labels_np[frame_number, 0] * 10  # Scale the labels
    np.clip(scaled_labels, 0, 255, out=scaled_labels)  # Clip the scaled labels

    pred_img.set_data(scaled_predictions)
    label_img.set_data(scaled_labels)
    return pred_img, label_img

# Create the animation
ani = FuncAnimation(fig, update, frames=100, blit=True, interval=50)  # Update every 50 ms
# Set up formatting for the movie files
writer = PillowWriter(fps=15)  # Specify the fps

# Save the animation
ani.save('heatmap_animation.gif', writer=writer)

# Show the plot
plt.show()
