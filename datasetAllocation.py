import torch

# Load the entire dataset
dataset = torch.load('data_labels_3.pth')
data, labels = dataset['data'], dataset['labels']

# Print the shape of the full dataset
print("Full dataset shape:", data.shape)

# Determine the number of samples you need for an eleventh of the dataset
num_samples = data.shape[0] // 20 # Integer division to get an eleventh

# Select an eleventh of the dataset
subset_data = data[:num_samples]
subset_labels = labels[:num_samples]

# Print the shape of the subset dataset
print("Subset dataset shape:", subset_data.shape)

# Save the subset to files
torch.save({'data': subset_data, 'labels': subset_labels}, 'subset_data_labels_20th.pth')
