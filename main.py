import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing.
mp_drawing = mp.solutions.drawing_utils

# Open the webcam.
cap = cv2.VideoCapture(0)

# Screen dimensions for scaling coordinates.
screen_width, screen_height = pyautogui.size()

# Variable to store previous y-coordinate of the index finger tip for tap detection.
prev_y = None

# Variable to control frame skipping.
frame_skip = 2
frame_count = 0

# Variable to store the time for performance tracking.
prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Flip the frame horizontally for a later selfie-view display.
    frame = cv2.flip(frame, 1)
    # Resize the frame to reduce the load.
    frame = cv2.resize(frame, (640, 480))
    # Convert the BGR image to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image and detect hands.
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the landmark for the index finger tip (landmark 8).
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, c = frame.shape
            finger_x, finger_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            
            # Move the mouse to the detected index finger position.
            screen_x = int(index_finger_tip.x * screen_width)
            screen_y = int(index_finger_tip.y * screen_height)
            pyautogui.moveTo(screen_x, screen_y, duration=0.1)

            # Detect tap motion.
            if prev_y is not None and (prev_y - finger_y) > 20:
                pyautogui.click()
            prev_y = finger_y

            # Draw the hand landmarks on the frame.
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Calculate and display FPS.
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame.
    cv2.imshow('Hand Tracking', frame)

    # Break the loop on pressing 'q'.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
