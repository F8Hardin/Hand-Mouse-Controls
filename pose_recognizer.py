import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import pyautogui
import sys
import argparse
import asyncio
import threading

trackingModifier = 4 #multiply the change in hand position by this amount to increase mouse movement
smoothingFactor = .4 #between 0 and 1, 1 for no smoothing, .4 feels good
movement_threshold = 25 #how far before updating position, 25 feels good
pyautogui.FAILSAFE = False
show_video = True
pose_names = ['leftClickLeft', 'leftClickRight', 'movementLeft', 'movementRight', 'noPose', 'rightClickLeftHand', 'rightClickRightHand']
previous_x, previous_y, smoothed_delta_x, smoothed_delta_y = None, None, None, None
isLeftClicking = False
isRightClicking = False
minConfidence = .5
imageBlur = 21 #for camera visibility

# Load the trained model
model = tf.keras.models.load_model("models/modelTwo.h5", compile=True)
print("Model loaded successfully!")

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

# Create async background thread
async_loop = asyncio.new_event_loop()
def run_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()
threading.Thread(target=run_async_loop, args=(async_loop,), daemon=True).start()

def move_mouse_based_on_landmark(landmark):
    global previous_x, previous_y, smoothed_delta_x, smoothed_delta_y

    screen_width, screen_height = pyautogui.size()  # Get screen resolution

    # Convert normalized (0-1) coordinates to screen pixel positions
    x_screen = int(landmark.x * screen_width * trackingModifier)
    y_screen = int(landmark.y * screen_height * trackingModifier)

    if previous_x is None or previous_y is None:
        smoothed_delta_x, smoothed_delta_y = 0, 0
        previous_x, previous_y = x_screen, y_screen
        
        #center the mouse
        pyautogui.moveTo(screen_width // 2, screen_height // 2, duration=.1)

    # Calculate the difference from the last moved position
    diff_x = abs(x_screen - previous_x)
    diff_y = abs(y_screen - previous_y)
    delta_x = previous_x - x_screen
    delta_y = y_screen - previous_y

    #print("Diff X:", diff_x, ", Diff Y:", diff_y)

    if diff_x > movement_threshold or diff_y > movement_threshold:
        smoothed_delta_x = (smoothingFactor * delta_x) + ((1 - smoothingFactor) * smoothed_delta_x)
        smoothed_delta_y = (smoothingFactor * delta_y) + ((1 - smoothingFactor) * smoothed_delta_y)

        # Move the mouse based on delta
        #pyautogui.moveRel(smoothed_delta_x, smoothed_delta_y, duration=.1)
        # Schedule the async move in the background event loop instead of calling it directly.
        asyncio.run_coroutine_threadsafe(
            async_move_mouse(smoothed_delta_x, smoothed_delta_y),
            async_loop
        )

        previous_x, previous_y = x_screen, y_screen

async def async_move_mouse(delta_x, delta_y):
    await asyncio.to_thread(pyautogui.moveRel, delta_x, delta_y, duration=0.1)

def extract_landmarks(results):
    flat_list = []

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for point in hand.landmark:
                flat_list.append([point.x, point.y, point.z])

    # Ensure exactly 42 landmarks (pad missing hands with zeros)
    while len(flat_list) < 42:
        flat_list.append([0, 0, 0])  # Padding

    # If there are extra landmarks (more than 42), truncate the list WHY ARE THERE EXTRA?? Check mediapipe
    flat_list = flat_list[:42]

    # Convert to NumPy array and flatten
    return np.array(flat_list, dtype=np.float32).flatten().reshape(1, 42, 3) 

def checkPoseCommand(currentPose, confidence, mediaPipeResults):
    global previous_x, previous_y, smoothed_delta_x, smoothed_delta_y, isLeftClicking, isRightClicking
    # Check number of hands detected
    #NOTE: Labels are backwards in mediapipe, left hand is RIGHT and right hand is Left
    #NOTE: When only one hand present, its always labeled as the right
    currentHandLabel = "right"
    currentHand = None
    if mediaPipeResults.multi_hand_landmarks:
        if len(mediaPipeResults.multi_hand_landmarks) == 2:
            if "Left" in currentPose: #note this is 'right' for media pipe, which is correct for our case
                currentHandLabel = "Left"
                currentHand = mediaPipeResults.multi_hand_landmarks[0]
            else: #left in media pipe, right in our case
                currentHandLabel = "Right"
                currentHand = mediaPipeResults.multi_hand_landmarks[1]
        else:
            currentHandLabel = mediaPipeResults.multi_handedness[0].classification[0].label
            #swap labels to match our case
            if currentHandLabel == "Right":
                currentHandLabel = "Left"
            else:
                currentHandLabel = "Right"
            currentHand = mediaPipeResults.multi_hand_landmarks[0]

    #print("Current pose:", currentPose)

    if currentPose == pose_names[2]:
        handleTrackingHand(currentHand)
    elif currentPose == pose_names[3]:
        handleTrackingHand(currentHand)
    elif currentPose == pose_names[0] or currentPose == pose_names[1]: #add some tracking check before allowing left and right clicks
        isLeftClicking = setMouseDown(True, "left", isLeftClicking)
        handleTrackingHand(currentHand)
    elif currentPose == pose_names[5] or currentPose == pose_names[6]:
        isRightClicking = setMouseDown(True, "right", isRightClicking)
    else:
        previous_x, previous_y, smoothed_delta_x, smoothed_delta_y = None, None, None, None

    if currentPose != pose_names[0] and currentPose != pose_names[1] and currentPose != "Unknown":
        isLeftClicking = setMouseDown(False, "left", isLeftClicking)
        isRightClicking = setMouseDown(False, "right", isRightClicking)

def setMouseDown(value, buttonToClick, isClickingButton):
    if value and not isClickingButton:
        pyautogui.mouseDown(button = buttonToClick)
        return True
    elif not value and isClickingButton:
        pyautogui.mouseUp(button = buttonToClick)
        return False
    return isClickingButton

def handleTrackingHand(currentHand): #THIS LOGIC SLOWING DOWN CODE
    #uses a landmark on the provided current hand to update logic
    landmarkToTrack = extractTrackedLandmark(currentHand)
    move_mouse_based_on_landmark(landmarkToTrack)
    return 0

def extractTrackedLandmark(currentHand):
    #gets the desired landmark from the hand to use for tracking
    return currentHand.landmark[0]

# Pose tracking control loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Exiting...")
        break

    blurred_frame = cv2.GaussianBlur(frame, (imageBlur, imageBlur), 0)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        if show_video:
            for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(blurred_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Extract landmarks and predict gesture every N frames
    if results.multi_hand_landmarks:
        inputData = extract_landmarks(results)
        prediction = model(inputData, training=False)  # More stable results
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        currentPose = "loading..."
        if confidence > minConfidence:  # Avoid unreliable predictions
            currentPose = pose_names[predicted_class]
        else:
            currentPose = "Unknown"
            print("Predictions:")
            for i in range(len(pose_names)):
                print("Pose:", pose_names[i], ", Value: ", prediction[0][i])
            print("=" * 10)

        if show_video:
            cv2.putText(frame, f"Gesture: {currentPose} ({confidence:.2f})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        checkPoseCommand(currentPose, confidence, results)


    if show_video:
        cv2.imshow("Gesture Detection", blurred_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
