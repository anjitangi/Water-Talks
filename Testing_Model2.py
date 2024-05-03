import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class FourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.modes1 = modes1
        self.modes2 = modes2
        
    def forward(self, x):
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat)
        # Apply complex weights to Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(2), x.size(3)))

class FNO3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, height, width):
        super(FNO3d, self).__init__()
        #print("Downsampling")
        self.downsample = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=4, padding=1)  # Downsampling
        # Calculate new dimensions after convolution
        # Formula for output size: (W - F + 2P) / S + 1, where:
        # W is input size, F is filter size, P is padding, S is stride
        reduced_height = (height - 2 + 2 * 1) // 4 + 1
        reduced_width = (width - 2 + 2 * 1) // 4 + 1
        #reduced_height = height
        #reduced_width = width
        #Linear(batch_size, in_cahnnels*width*height)
        self.fc0 = nn.Linear(in_channels*reduced_height*reduced_width,32*reduced_height*reduced_width)  # Adjusted dimensions
        self.fourier_layers = nn.ModuleList([FourierLayer(32, 32, modes1, modes2) for _ in range(4)])
        self.fc1 = nn.Linear(32*reduced_height*reduced_width, out_channels * reduced_height * reduced_width)  # Adjusted dimensions

    def forward(self, x):

        x = self.downsample(x)
        batch_size, channels, height, width = x.shape

        # Flatten the spatial dimensions before the linear layer
        x = x.view(batch_size, channels * height * width)
        print("Fully-Connected Layer")
        x = F.gelu(self.fc0(x))

        # Reshape back to original spatial dimensions with new channels
        x = x.view(batch_size, 32, height, width)

        # Process through Fourier layers
        for layer in self.fourier_layers:
            print("Fourier Layer")
            x = F.gelu(layer(x))

        # Flatten again before final projection
        x = x.view(batch_size, -1)
        print("Final Fully-Connected Layer")
        x = self.fc1(x)
        # Reshape to expected output dimensions
        x = x.view(batch_size, out_channels, height, width)
        return x


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def standardize_data(data):
    # Compute the mean and std along the batch dimension
    mean = data.mean(dim=(0, 2, 3), keepdim=True)
    std = data.std(dim=(0, 2, 3), keepdim=True)
    # Standardize data
    normalized_data = (data - mean) / std
    return normalized_data, mean, std

def normalize_batch(batch):
    # Separate data and labels
    data_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]

    # Convert lists to tensors
    data = torch.stack(data_list)
    labels = torch.stack(labels_list)

    # Normalize data and get mean and std
    normalized_data, mean, std = standardize_data(data)

    return normalized_data, labels, mean, std

# Load dataset
dataset = torch.load('data_labels_1.pth')
data, labels = dataset['data'], dataset['labels']

# Assuming the data is [n_samples, channels, height, width]
tensor_dataset = TensorDataset(data, labels)
# Create DataLoader
data_loader = DataLoader(tensor_dataset, batch_size=10, shuffle=True, collate_fn=normalize_batch)

# Parameters
in_channels = 2  # X and Y components
out_channels = 2  # Predicting next X and Y components
modes1, modes2 = 12, 12  # Fourier modes
height = 240
width = 320

# Load the model
model = FNO3d(in_channels, out_channels, modes1, modes2, height, width)
model.load_state_dict(torch.load('FourierNeuralOperatorModel.pth'))
model.eval()

predictions = []
original_labels = []

with torch.no_grad():
    for i, (velocity_field, target_field, mean, std) in enumerate(data_loader):
        if i >= 100:  # Stop after collecting 100 samples
            break
        output = model(velocity_field)
        # Reverse normalization
        output_renorm = output * std + mean
        target_field_renorm = target_field * std + mean
        
        # Upscale predictions and labels back to original size (320 x 240)
        upscaled_output = F.interpolate(output_renorm, size=(240, 320), mode='bilinear', align_corners=False)
        upscaled_target = F.interpolate(target_field_renorm, size=(240, 320), mode='bilinear', align_corners=False)
        
        predictions.append(upscaled_output)
        original_labels.append(upscaled_target)

# Convert list of tensors to a single tensor
predictions_tensor = torch.cat(predictions, dim=0)
labels_tensor = torch.cat(original_labels, dim=0)

# Save the tensors
torch.save(predictions_tensor, 'predictions100.pth')
torch.save(labels_tensor, 'labels100.pth')
