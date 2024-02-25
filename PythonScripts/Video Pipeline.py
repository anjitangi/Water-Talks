import cv2
import numpy as np

global_frame = None

#THRESHOLD TRACKBARS
# def threshold_on_trackbar(val):
#     global global_frame
#     if global_frame is None:
#         return  # Exit if there's no frame (initial call or error)

#     # This function gets called whenever the trackbar position changes
#     # Retrieve the current positions of the lower and upper trackbars
#     lower_thresh = cv2.getTrackbarPos('Lower Threshold', 'Binary Threshold')
#     upper_thresh = cv2.getTrackbarPos('Upper Threshold', 'Binary Threshold')

#     # Re-apply the threshold with the new values
#     _, thresh = cv2.threshold(global_frame, lower_thresh, upper_thresh, cv2.THRESH_BINARY)
    
#     # Display the updated binary image
#     cv2.imshow('Binary Threshold', thresh)

#EROSION DILATION TRACKBARS
# def erosion_dilation_on_trackbar(val):
#     global global_frame
#     if global_frame is None:
#         return  # Exit if there's no frame (initial call or error)

#     # This function gets called whenever the trackbar position changes
#     # Retrieve the current positions of the lower and upper trackbars
#     erosion_iteration = cv2.getTrackbarPos('Erosion Iteration', 'Erosion & Dilation')
#     dilation_iteration = cv2.getTrackbarPos('Dilation Iteration', 'Erosion & Dilation')
#     kernel_size = cv2.getTrackbarPos('kernel_size', 'Erosion & Dilation')

#     # Re-apply the threshold with the new values
#     # Define a kernel using getStructuringElement
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

#     # Perform erosion
#     erosion = cv2.erode(global_frame, kernel, iterations=erosion_iteration)

#     #Perform Dilation
#     dilation = cv2.dilate(erosion, kernel, iterations=dilation_iteration)
    
#     # Display the updated binary image
#     cv2.imshow('Erosion & Dilation', dilation)

def compute_difference(frame1,frame2):
    return  cv2.absdiff(frame1,frame2)

def initialize_kalman(obj_id):
    kf = cv2.KalmanFilter(4, 2)  # 4 dynamic params (x, y, dx, dy), 2 measurement params (x, y)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kf.statePre = np.zeros((4,1), np.float32)
    kf.errorCovPre = np.eye(4, dtype=np.float32)
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf, obj_id

def predict_kalman(kfs):
    predictions = []
    for kf in kfs:
        prediction = kf.predict()
        predictions.append((prediction[0], prediction[1]))  # Extract x, y from prediction
    return predictions

def update_kalman(kf, x, y):
    measurement = np.array([[np.float32(x)], [np.float32(y)]], dtype = np.float32)
    kf.correct(measurement)

def match_contours_to_predictions(contours, predicted_positions):
    matched_indices = {}
    distance_matrix = np.zeros((len(contours), len(predicted_positions)))

    # Calculate the distance from each contour centroid to each predicted position
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        centroid = np.array([x + w / 2, y + h / 2])

        for j, pred_pos in enumerate(predicted_positions):
            pred_pos_array = np.array(pred_pos)
            distance_matrix[i, j] = np.linalg.norm(centroid - pred_pos_array)

    # Perform matching based on the minimum distance
    for _ in range(min(len(contours), len(predicted_positions))):
        # Find the smallest distance in the matrix
        min_idx = np.unravel_index(np.argmin(distance_matrix, axis=None), distance_matrix.shape)
        contour_idx, prediction_idx = min_idx

        # Assign the contour to the predicted position
        matched_indices[contour_idx] = prediction_idx

        # Set the distances for the matched contour and prediction to infinity to exclude them from further matching
        distance_matrix[contour_idx, :] = np.inf
        distance_matrix[:, prediction_idx] = np.inf

    return matched_indices
   

# Initialize the video
cap = cv2.VideoCapture("Wallkill Videos\\KleinKill15s - 28 Dug Rd.mp4")

#THRESHOLD TRACKBARS
# cv2.namedWindow('Binary Threshold')
# cv2.createTrackbar('Lower Threshold', 'Binary Threshold', 0, 255, threshold_on_trackbar)
# cv2.createTrackbar('Upper Threshold', 'Binary Threshold', 255, 255, threshold_on_trackbar)

#EROSION DILATION TRACKBARS
# cv2.namedWindow('Erosion & Dilation')
# cv2.createTrackbar('Erosion Iteration', 'Erosion & Dilation', 0, 15, erosion_dilation_on_trackbar)
# cv2.createTrackbar('Dilation Iteration', 'Erosion & Dilation', 0, 15, erosion_dilation_on_trackbar)
# cv2.createTrackbar('kernel_size', 'Erosion & Dilation', 1, 15, erosion_dilation_on_trackbar)

# Parameters
tracked_contours = []  # To store the centroid of each tracked contour
trajectories = {}
next_obj_id = 0  # Start with 0 and increment for each new object
kalman_filters_with_id = [] 

num_frames_for_background = 20
count = 0

single_kf, _ = initialize_kalman(0)  # Initialize Kalman filter for the first object
trajectory = []  # To store the trajectory of the single tracked object

# Initialize a running sum of frames
background_accumulator = None

while count < num_frames_for_background:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame.")
        break

    # Convert the frame to float
    float_frame = frame.astype(np.float32)

    # Initialize or accumulate the frames
    if background_accumulator is None:
        background_accumulator = float_frame
    else:
        background_accumulator += float_frame

    count += 1

# Calculate the average to obtain the background frame
background_frame = (background_accumulator / num_frames_for_background).astype(np.uint8)

# ret, reference_frame = cap.read()
largest_contour = None
while True:

    ret, frame = cap.read()

    if not ret:
        break
    
    diff = compute_difference(frame,background_frame)
    # cv2.imshow("Motion Captured", diff)

    # Binarize the image
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Create trackbars for lower and upper threshold
    # global_frame = gray
    # threshold_on_trackbar(0)

    thresh = cv2.threshold(gray, 7, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow("Post-Thresholding", thresh)
    
    # Create trackbars for erosion and dilation
    # global_frame = thresh
    # erosion_dilation_on_trackbar(0)
    

    # Define a kernel using getStructuringElement
    kernel_size = 2  # You can adjust the kernel size as needed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Perform erosion
    erosion = cv2.erode(thresh, kernel, iterations=3)
    erosion = erosion[350:750,:]
    # cv2.imshow("Post-Erosion", erosion)

    #Perform Dilation
    # dilation = cv2.dilate(erosion, kernel, iterations=2)
    # dilation = dilation[350:750,:]

    # cv2.imshow("Pre-Canny", dilation)
    # Apply Canny edge detection
    edges = cv2.Canny(erosion, 100,125)

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #ONE CONTOUR
    predicted_position = single_kf.predict()
    if len(trajectory) == 0 and contours:
        largest_contour = max(contours, key=cv2.contourArea) if contours else None

    # If contours are detected, find the one closest to the predicted position
    if contours:
        closest_contour = min(contours, key=lambda cnt: np.linalg.norm(np.array(cv2.boundingRect(cnt)[:2]) - predicted_position[:2]))
        x, y, w, h = cv2.boundingRect(largest_contour)
        centroid = (x + w//2, y + h//2)
        update_kalman(single_kf, centroid[0], centroid[1])
        trajectory.append(centroid)  # Update the single trajectory
        largest_contour = closest_contour
    
    frame_with_one_trajectory = erosion.copy()
    # Convert the single-channel grayscale image to a 3-channel BGR image
    frame_with_one_trajectory_color = cv2.merge([frame_with_one_trajectory, frame_with_one_trajectory, frame_with_one_trajectory])

    if len(trajectory) > 1:
        for i in range(1, len(trajectory)):
            cv2.line(frame_with_one_trajectory_color, trajectory[i - 1], trajectory[i], (0, 255, 0), 1)

    cv2.imshow("Single Trajectory", frame_with_one_trajectory_color)

    # ALL CONTOURS
    #  # Predict the positions of tracked contours
    # predicted_positions = predict_kalman([kf for kf, _ in kalman_filters_with_id])

    # # Match detected contours to tracked contours based on nearest predicted position
    # matched_indices = match_contours_to_predictions(contours, predicted_positions)

    # new_kalman_filters_with_id  = []
    # for i, cnt in enumerate(contours):
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     centroid = (x + w//2, y + h//2)

    #     if i in matched_indices:
    #         kf_index = matched_indices[i]
    #         kf, obj_id = kalman_filters_with_id[kf_index]
    #         update_kalman(kf, centroid[0], centroid[1])
    #         trajectories[obj_id].append(centroid)  # Update trajectory
    #     else:
    #         kf, obj_id = initialize_kalman(next_obj_id)
    #         next_obj_id += 1  # Prepare the next object ID
    #         update_kalman(kf, centroid[0], centroid[1])
    #         trajectories[obj_id] = [centroid]

    #     new_kalman_filters_with_id.append((kf, obj_id))

    # kalman_filters_with_id = new_kalman_filters_with_id  # Update the list of Kalman filters for the next frame

    # # After updating the trajectories for the current frame

    # # Create a copy of the frame for drawing
    # frame_with_trajectories = erosion.copy()

    # # Iterate over each object's trajectory
    # for obj_id, points in trajectories.items():
    #     # Check if the trajectory has at least two points
    #     if len(points) > 1:
    #         # Draw the entire trajectory
    #         for i in range(1, len(points)):
    #             cv2.line(frame_with_trajectories, points[i - 1], points[i], (0, 0, 0), 1)

    # # Display the frame with trajectories
    # cv2.imshow("Trajectories", frame_with_trajectories)


    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
    
filename = f"trajectory_{0}.txt"
np.savetxt(filename, np.array(trajectory), fmt='%d')
# for obj_id, trajectory in trajectories.items():
#     filename = f"trajectory_{obj_id}.txt"
#     np.savetxt(filename, np.array(trajectory), fmt='%d')

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

