import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity
import traceback
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# Applying automatic mixed precision
from torch.cuda.amp import autocast
os.environ['OMP_NUM_THREADS'] = '1'

def standardize_data(data):
    # Compute the mean and std along the batch dimension
    mean = data.mean(dim=(0, 2, 3), keepdim=True)
    std = data.std(dim=(0, 2, 3), keepdim=True)
    # Standardize data
    normalized_data = (data - mean) / std
    return normalized_data

# Define a function to apply normalization to a batch
def normalize_batch(batch):
     # We first separate these into separate lists/tensors
    data_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]

    # Convert lists to tensors
    data = torch.stack(data_list)  # This stacks the data tensors along a new dimension (batch dimension)
    labels = torch.stack(labels_list)  # Same for labels

    # Check that all tensors are on the same device
    first_device = data.device
    assert all(d.device == first_device for d in data_list), "Not all tensors are on the same device"
    assert labels.device == first_device, "Labels are not on the same device as data"
    
    normalized_data = standardize_data(data)
    return normalized_data, labels

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
        target_height = 64
        target_width = 32

        # Calculate kernel and stride sizes
        kernel_height = height // target_height
        kernel_width = width // target_width
        stride_height = kernel_height
        stride_width = kernel_width
        print("Height % kernel" + str(height%kernel_height))
        print("Width  % kernel" + str(width%kernel_width))
        # Ensure the calculated sizes are integers and match the target dimensions
        assert height % kernel_height == 0, "Height not evenly divisible"
        assert width % kernel_width == 0, "Width not evenly divisible"

        # Initialize the MaxPool2d without padding (since no padding is desired)
        self.downsample = nn.MaxPool2d(kernel_size=(kernel_height, kernel_width), stride=(stride_height, stride_width))

        # Check output dimensions with dummy input
        dummy_input = torch.zeros(1, in_channels, height, width)
        reduced_size = self.downsample(dummy_input).shape
        reduced_height, reduced_width = reduced_size[2]-3, reduced_size[3]-1
 
        print("Weight Dimensions Input:")
        print("Height: " + str(reduced_height) + ", Width:" + str(reduced_width) + ", Channels:" + str(in_channels))

        self.fc0 = nn.Linear(in_channels*reduced_height*reduced_width,32*reduced_height*reduced_width)  # Adjusted dimensions
        self.fourier_layers = nn.ModuleList([FourierLayer(32, 32, modes1, modes2) for _ in range(4)])
        self.fc1 = nn.Linear(32*reduced_height*reduced_width, out_channels * reduced_height * reduced_width)  # Adjusted dimensions

    def forward(self, x):

        x = self.downsample(x)
        x = x[:,:,:-3,:-1]
        batch_size, channels, height, width = x.shape

        # Flatten the spatial dimensions before the linear layer
        print("Input Size:")
        print("Height: " + str(height) + ", Width:" + str(width) + ", Channels:" + str(channels))
        x = x.contiguous()
        x = x.view(batch_size, channels * height * width)
        print("Fully-Connected Layer")
        x = F.gelu(self.fc0(x))

        # Reshape back to original spatial dimensions with new channels
        x = x.view(batch_size, 32, height, width)

#        x = x.type(torch.complex64)
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

def downsample(tensor):
     # Target dimensions
    target_height = 64
    target_width = 32

        # Calculate kernel and stride sizes
    kernel_height = height // target_height
    kernel_width = width // target_width
    stride_height = kernel_height
    stride_width = kernel_width


        # Initialize the MaxPool2d without padding (since no padding is desired)
    m = nn.MaxPool2d(kernel_size=(kernel_height, kernel_width), stride=(stride_height, stride_width))
    return m(tensor)

# Parameters
in_channels = 2  # X and Y components
out_channels = 2  # Predicting next X and Y components
modes1, modes2 = 12, 12  # Fourier modes
    
# Load dataset
dataset = torch.load('subset_data_labels_20th.pth')
data, labels = dataset['data'], dataset['labels']
data = data[:, :, :-8, :-14]
height = data.shape[2]
width = data.shape[3]
labels = labels[:, :, :-8, :-14]
print(data.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FNO3d(in_channels, out_channels, modes1, modes2, height, width).to(device)
#model = nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
labels = labels.to(device)
data = data.to(device)

print("Data type:", data.dtype)
num_elements = data.nelement()
element_size = data.element_size()  # size in bytes for each element
data_memory = num_elements * element_size / (1024 ** 3)  # memory in GB

print(f"Approximate memory usage of the dataset: {data_memory} GB")

# Create TensorDataset
tensor_dataset = TensorDataset(data, labels)
# Create DataLoader
data_loader = DataLoader(tensor_dataset, batch_size=2,shuffle=True, collate_fn=normalize_batch)

# Assuming labels is a Tensor of your labels data
label_mean = labels.mean()
label_std = labels.std()

def normalize_labels(labels):
    return (labels - label_mean) / label_std

output_means = []
output_stds = []

try:
    # Modify your training loop to normalize labels
    for epoch in range(10):
        epoch_loss = 0
        for velocity_field, target_field in data_loader:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                record_shapes=True, 
                profile_memory=True, # Enables memory profiling
                with_stack=True) as prof:
                with record_function("model_inference"):
                    optimizer.zero_grad()
                    output = model(velocity_field)
                    output_means.append(output.mean(dim=(0, 2, 3)))  # Mean over batch, height, and width
                    output_stds.append(output.std(dim=(0, 2, 3)))    # Std over batch, height, and width
                    target_field_downsampled = downsample(target_field)
                    target_field_downsampled = target_field_downsampled[:,:,:-3,:-1]   
                    norm_target_field = normalize_labels(target_field_downsampled)  # Normalize the labels here
                    loss = criterion(output, norm_target_field)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            print(f"Epoch {epoch}: Loss = {epoch_loss / len(data_loader)}")

except Exception as e:
    traceback.print_exc() 
    print(f"An error occurred: {e}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")
    del loss, output, input, data, labels, target_field_downsampled  # Remove references to tensors
    torch.cuda.empty_cache() 
else:
    # Calculate the overall mean and std across all batches
    output_mean = torch.stack(output_means).mean(dim=0)
    output_std = torch.stack(output_stds).mean(dim=0)

    # Save the statistics
    torch.save({'mean': output_mean, 'std': output_std}, 'output_stats.pth')

    # Save the model
    torch.save(model.state_dict(), 'FourierNeuralOperatorModel.pth')
