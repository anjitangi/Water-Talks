import cv2
import numpy as np
import torch  # Assuming you're using PyTorch for neural network operations
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def adjustImage(frame, mode, contrastBool, threshBool,
                upperThresh, lowerThresh, contrast, brightness):
    """
    Adjust the image's contrast, brightness, and apply thresholding either in Grayscale or HSV.

    Parameters:
    - frame: Input image in BGR format.
    - mode: Color mode for processing - 'HSV' or 'Gray'.
    - contrastBool: Boolean to apply contrast adjustment.
    - threshBool: Boolean to apply thresholding.
    - upperThresh: Upper threshold limit.
    - lowerThresh: Lower threshold limit.
    - contrast: Contrast factor.
    - brightness: Brightness offset.

    Returns:
    - Processed image based on selected mode and adjustments.
    """
    # frame = cv2.resize(frame, (320,240))
    # print(frame.shape)
    if mode == 'HSV':
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if contrastBool:
            # Apply contrast and brightness adjustments on the Value channel
            hsv[:,:,2] = cv2.convertScaleAbs(hsv[:,:,2], alpha=contrast, beta=brightness)
        if threshBool:
            # Apply thresholding on the Value channel
            _, hsv[:,:,2] = cv2.threshold(hsv[:,:,2], lowerThresh, upperThresh, cv2.THRESH_BINARY)
        # Convert back to BGR for output if needed
        processed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    elif mode == 'Gray':
        # Convert BGR to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if contrastBool:
            # Apply contrast and brightness adjustments
            gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
        if threshBool:
            # Apply thresholding
            _, gray = cv2.threshold(gray, lowerThresh, upperThresh, cv2.THRESH_BINARY)
        processed_img = gray
    else:
        raise ValueError("Mode should be 'HSV' or 'Gray'.")

    return processed_img

def extract_and_compute_flows(video_path, mode, contrastBool, threshBool, upperThresh, lowerThresh, contrast, brightness):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        exit()
    ret, prev_frame = cap.read()
    prev_frame = adjustImage(prev_frame, mode, contrastBool, threshBool, upperThresh, lowerThresh, contrast, brightness)
    prev_frame = prev_frame[0:1200, 675:1250]

    vector_data = []
    vector_labels = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = adjustImage(frame, mode, contrastBool, threshBool, upperThresh, lowerThresh, contrast, brightness)
        current_frame = current_frame[0:1200, 675:1250]

        # Compute the optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vector_data.append(flow)

        # Prepare for the next iteration
        prev_frame = current_frame

    # Converting list to tensors and creating labels
    vector_data_np = np.array(vector_data)  # Convert list to a single numpy ndarray
    vector_data_tensor = torch.tensor(vector_data_np)
    vector_labels_tensor = vector_data_tensor[1:]  # Future vectors as labels (shift by 1)

    cap.release()
    return vector_data_tensor[:-1], vector_labels_tensor  # Exclude the last flow as it has no future vector

# Usage
video_path = 'KleinKill 11s - 28 Dug Rd.mp4'
mode = "Gray"
contrastBool = True
threshBool = True
lowerThresh = 65
upperThresh = 250
contrast = 0.25
brightness = 28

data, labels = extract_and_compute_flows(video_path, mode, contrastBool, threshBool, upperThresh, lowerThresh, contrast, brightness)

# Assuming data is [n_samples, height, width, 2] where 2 represents flow x and y components
data = data.permute(0, 3, 1, 2)  # Rearrange to [n_samples, 2, height, width]
labels = labels.permute(0, 3, 1, 2)  # Same rearrangement for labels

torch.save({
    'data': data,
    'labels': labels
}, 'data_labels_3.pth')
