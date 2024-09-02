#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import mediapipe as mp

# Load the Haar cascade for face detection
face_cap = cv2.CascadeClassifier(r"C:/Users/Shrawani Gongshe/Downloads/haarcascade_frontalface_default.xml")

# Initialize Mediapipe Hands for finger detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the video capture
video_cap = cv2.VideoCapture(0)

# Check if the video capture was initialized correctly
if not video_cap.isOpened():
    print("Cannot open camera")
    exit()

def draw_shape(frame, landmarks, shape_type):
    # Extract the coordinates of the fingertips
    finger_positions = [(int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])) for i in [4, 8, 12, 16, 20]]
    
    if shape_type == 'circle':
        center = finger_positions[0]
        radius = int(((finger_positions[1][0] - finger_positions[0][0])**2 + (finger_positions[1][1] - finger_positions[0][1])**2)**0.5)
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        
    elif shape_type == 'square':
        top_left = finger_positions[0]
        bottom_right = finger_positions[1]
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
        
    elif shape_type == 'rectangle':
        top_left = finger_positions[0]
        bottom_right = finger_positions[1]
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        
    elif shape_type == 'triangle':
        pt1 = finger_positions[0]
        pt2 = finger_positions[1]
        pt3 = finger_positions[2]
        cv2.polylines(frame, [np.array([pt1, pt2, pt3], np.int32)], isClosed=True, color=(255, 0, 255), thickness=2)

while True:
    ret, video_data = video_cap.read()
    
    # Check if the frame was captured correctly
    if not ret:
        print("Failed to capture video frame")
        break

    # Convert the captured frame to grayscale
    gray_frame = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cap.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Convert the frame to RGB for hand detection
    rgb_frame = cv2.cvtColor(video_data, cv2.COLOR_BGR2RGB)

    # Detect hands and fingers
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(video_data, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks for drawing shapes
            landmarks = hand_landmarks.landmark

            # Define the shape type (you can change this to 'circle', 'square', 'rectangle', 'triangle')
            shape_type = 'circle'  # Example shape type

            # Draw the shape based on detected hand landmarks
            draw_shape(video_data, landmarks, shape_type)

    # Display the video feed with detected faces and drawn shapes
    cv2.imshow("video_live", video_data)

    # Break the loop if the 'a' key is pressed
    if cv2.waitKey(10) & 0xFF == ord('a'):
        break

# Release the video capture and close all OpenCV windows
video_cap.release()
cv2.destroyAllWindows()


# In[ ]:




