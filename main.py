import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Lip landmark indices (combining upper outer, lower outer, upper inner, and lower inner)
lip_landmark_indices = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,  # Upper outer
    146, 91, 181, 84, 17, 314, 405, 321, 375,         # Lower outer
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, # Upper inner
    95, 88, 178, 87, 14, 317, 402, 318, 324           # Lower inner
]

# Function to create a lip mask based on the lip landmarks
def create_lip_mask(image, landmarks):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Create an empty mask
    lip_points = []

    # Gather lip points from the landmark list
    for idx in lip_landmark_indices:
        x = int(landmarks.landmark[idx].x * image.shape[1])
        y = int(landmarks.landmark[idx].y * image.shape[0])
        lip_points.append((x, y))
    
    # Convert lip points to a NumPy array
    lip_points = np.array(lip_points, np.int32)
    
    # Fill the lip area on the mask
    cv2.fillPoly(mask, [lip_points], 255)

    return mask

# Function to apply lipstick color to the detected lip area
def apply_lipstick(image, mask, color, opacity=0.6):
    # Create an image of the selected lipstick color
    colored_lips = np.zeros_like(image)
    colored_lips[:] = color  # e.g., (B, G, R) tuple for color
    
    # Apply the lip mask to the colored lips
    colored_lips = cv2.bitwise_and(colored_lips, colored_lips, mask=mask)
    
    # Blend the colored lips with the original image using opacity
    blended = cv2.addWeighted(colored_lips, opacity, image, 1 - opacity, 0)
    
    # Combine the blended lipstick with the rest of the image (where mask is not applied)
    final_image = np.where(colored_lips > 0, blended, image)
    
    return final_image

# Capture video (from webcam)
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the image to RGB for MediaPipe processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    # If face landmarks are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Create a lip mask from the detected lip landmarks
            lip_mask = create_lip_mask(image, face_landmarks)

            # Define lipstick color in BGR format (e.g., red: (0, 0, 255))
            lipstick_color = (115, 104, 193)  # Change this to different BGR values for different colors

            # Apply lipstick to the lips using the mask
            final_image = apply_lipstick(image, lip_mask, lipstick_color, opacity=0.6)

            # Display the final image with lipstick applied
            cv2.imshow('Virtual Lipstick Application', final_image)

    else:
        # Show original image if no landmarks are detected
        cv2.imshow('Virtual Lipstick Application', image)

    # Press 'Esc' to exit
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
