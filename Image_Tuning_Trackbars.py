import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from google.colab.patches import cv2_imshow

def adjustImage(frame, mode, contrastBool, threshBool,
                upperThresh, lowerThresh, contrast, brightness):
    if mode == 'HSV':
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if contrastBool:
            hsv[:,:,2] = cv2.convertScaleAbs(hsv[:,:,2], alpha=contrast, beta=brightness)
        if threshBool:
            _, hsv[:,:,2] = cv2.threshold(hsv[:,:,2], lowerThresh, upperThresh, cv2.THRESH_BINARY)
        processed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    elif mode == 'Gray':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if contrastBool:
            gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
        if threshBool:
            _, gray = cv2.threshold(gray, lowerThresh, upperThresh, cv2.THRESH_BINARY)
        processed_img = gray
    else:
        raise ValueError("Mode should be 'HSV' or 'Gray'.")

    return processed_img

# Function to get current slider values and update the image
def updateImage(*args):
    contrast = cv2.getTrackbarPos('Contrast', 'Adjustments')
    brightness = cv2.getTrackbarPos('Brightness', 'Adjustments')
    lowerThresh = cv2.getTrackbarPos('Lower Threshold', 'Adjustments')
    upperThresh = cv2.getTrackbarPos('Upper Threshold', 'Adjustments')

    new_frame = adjustImage(frame, mode, contrastBool, threshBool, upperThresh, lowerThresh, contrast, brightness)
    cv2.imshow('Adjustments', new_frame)

video_path = '/content/vecteezy_snow-falling-background_1623412.mp4'
mode = "Gray"
contrastBool = True
threshBool = True

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cv2_imshow(frame)

# Create a window for sliders
cv2.namedWindow('Adjustments')

# Create trackbars for contrast, brightness, and thresholds
cv2.createTrackbar('Contrast', 'Adjustments', 1, 10, updateImage)  # Slider for contrast
cv2.createTrackbar('Brightness', 'Adjustments', 50, 255, updateImage)  # Slider for brightness
cv2.createTrackbar('Lower Threshold', 'Adjustments', 0, 255, updateImage)  # Slider for lower threshold
cv2.createTrackbar('Upper Threshold', 'Adjustments', 255, 255, updateImage)  # Slider for upper threshold

# Show initial image with default values
updateImage()

# Wait for key press and exit
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
