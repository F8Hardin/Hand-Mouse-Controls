import cv2
import mediapipe as mp
import json
import time
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,  
    max_num_hands=2,  
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

# Create output directories
json_filename = "gestures.json"
image_folder = "gesture_images"
os.makedirs(image_folder, exist_ok=True)

# Load existing gesture data if the file exists
if os.path.exists(json_filename):
    with open(json_filename, "r") as json_file:
        try:
            gesture_data = json.load(json_file) 
        except json.JSONDecodeError:
            gesture_data = []
else:
    gesture_data = []

print("Press SPACE to start a 3-second timer before capturing hand data.")
print("Enter a gesture name after capture (or type 'CANCEL' to discard).")
print("Press 'q' to quit.")

# Countdown management variables
countdown_active = False
countdown_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Exiting...")
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Handle active countdown
    if countdown_active:
        elapsed_time = time.time() - countdown_start_time
        remaining_time = max(0, 3 - int(elapsed_time))

        # Display countdown on screen
        cv2.putText(frame, f"Capturing in {remaining_time}...", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

        if elapsed_time >= 3:  # Countdown finished, capture the frame
            print("Frame captured!")
            countdown_active = False  # Reset countdown state

            # Process the captured frame
            results = hands.process(frame_rgb)

            # Extract landmark data
            frame_landmarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_tensor = []
                    for landmark in hand_landmarks.landmark:
                        hand_tensor.append({"x": landmark.x, "y": landmark.y, "z": landmark.z})
                    frame_landmarks.append(hand_tensor)

                # Ensure 2-hand structure (impute zeroed landmarks if only one hand is detected)
                while len(frame_landmarks) < 2:
                    zero_hand = [{"x": 0, "y": 0, "z": 0} for _ in range(21)]
                    frame_landmarks.append(zero_hand)
            else:
                # No hands detected → Assume "no_gesture" with 42 zeroed landmarks (two hands)
                zero_hand = [{"x": 0, "y": 0, "z": 0} for _ in range(21)]
                frame_landmarks = [zero_hand, zero_hand]  # Two hands, all zeroes

            # Ask for a gesture name (or auto-label as "no_gesture")
            if results.multi_hand_landmarks:
                gesture_name = input("Enter gesture name (or type 'CANCEL' to discard): ").strip()
                # Save the gesture data

                existing_names = [g["pose"] for g in gesture_data]
                base_name = gesture_name
                counter = 1

                
                if base_name.upper() == "CANCEL":
                    print("Gesture discarded. No data was saved.")
                else:
                    while gesture_name in existing_names:
                        gesture_name = f"{base_name}_{counter}"
                        counter += 1

                    print(f"Gesture '{gesture_name}' saved.")

                    # Save the captured image with blur effect
                    image_filename = f"{gesture_name}.jpg" 
                    image_path = os.path.join(image_folder, image_filename)

                    # Apply Gaussian blur to the image
                    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
                    cv2.imwrite(image_path, blurred_frame)

                    print(f"Blurred image saved to {image_path}")

                    # Add to the existing gesture dataset
                    gesture_data.append({
                        "pose": gesture_name,
                        "image_path": image_path,
                        "landmarks": frame_landmarks
                    })

                    # Save to a single JSON file
                    with open(json_filename, "w") as json_file:
                        json.dump(gesture_data, json_file, indent=4)

                    print(f"Updated JSON file: {json_filename}")

    # Show the camera feed
    cv2.imshow("Hand Tracking - Snap Mode", frame)

    # Check for user input
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # Space bar to start countdown
        if not countdown_active:
            print("Get into position! Capturing in 3 seconds...")
            countdown_active = True
            countdown_start_time = time.time()

    elif key == ord('q'):  # Quit
        print("Exiting program...")
        break

cap.release()
cv2.destroyAllWindows()
