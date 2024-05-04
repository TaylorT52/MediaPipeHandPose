#hand gesture, pose detection using yolo 
#author @ taylor tam

##### IMPORTS #####
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

from ultralytics import YOLO
import os
from ultralytics.utils.plotting import Annotator 
import json
from google.protobuf.json_format import MessageToDict
import time
import send_it2
import cv2
import torch

##### LOADING STUFF #####
#load some stuff!
weights_loc = "weights/best.pt"
yolo_model = YOLO(weights_loc)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#Use GPU
print("CUDA available: ", torch.cuda.is_available())
print("Current device: ", torch.cuda.current_device())
print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

#starting variables
padding = 30
img_counter = 0

#mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
desired_aspect_ratio = 1 
standard_size = (350, 350)
gestures = ["speed_inc", "speed_dec", "to_right", "to_left", "bumper"]

#load base gestures
with open("base_gestures.json", "r") as infile:
    data = json.load(infile)

##### GESTURE MATCHING #####
#preprocessing for gesture matching
def preprocess_image(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([140, 100, 100])
    upper_pink = np.array([170, 255, 255])
    mask = cv2.inRange(hsv_img, lower_pink, upper_pink)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

#match gestures w/ orb + bf 
def match_gestures(handedness, img2, threshold=110):
    print(handedness)
    img2_processed = preprocess_image(img2)
    orb = cv2.ORB_create()
    
    for val in gestures:
        des1_list = data[val]
        des1 = np.array(des1_list)
        kp2, des2 = orb.detectAndCompute(img2_processed, None)

        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            if des1.dtype != np.uint8:
                des1 = des1.astype(np.uint8)
            if des2.dtype != np.uint8:
                des2 = des2.astype(np.uint8)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > threshold:
                print("The gestures are similar.")
                if val == "to_right" and handedness == "Right":
                    return "to_left"
                elif val == "to_left" and handedness == "Left":
                    return "to_right"
                else:
                    return val
        else:
            print("One or both sets of descriptors are missing or empty.")
    return ""

##### PROCESS FRAMES! #####
#process each frame
def read_frame(frame, hands): 
    print('reading frame')
    handedness = ""
    start = ""
    
    #image loading
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    results = hands.process(image)
    canvas_for_yolo = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = frame.shape

    #for drawing
    canvas = np.zeros_like(image)

    #draw results on black canvas
    if results.multi_hand_landmarks:
        for handLMs in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                canvas, handLMs, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )
        handedness = results.multi_handedness[0].classification[0].label
    width = canvas_for_yolo.shape[1]

    #perform YOLO predictions
    yolo_results = yolo_model.predict(frame)

    for r in yolo_results:
        #annotate boxes
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            mirrored_x_min = width - b[2]
            mirrored_x_max = width - b[0]
            mirrored_box = [mirrored_x_min-padding, b[1]-padding, mirrored_x_max+padding, b[3]+padding]
            # c = box.cls

            #create cropped canvas for saving
            save_me = canvas[max(int(mirrored_box[1]), 0):min(int(mirrored_box[3]), canvas.shape[0]),
                            max(int(mirrored_box[0]), 0):min(int(mirrored_box[2]), canvas.shape[1])]
            h, w = save_me.shape[:2]
            current_aspect_ratio = w / h
    
            # Calculate padding
            if current_aspect_ratio < desired_aspect_ratio:
                # Pad sides
                new_width = int(desired_aspect_ratio * h)
                pad_width = (new_width - w) // 2
                padded_image = cv2.copyMakeBorder(save_me, 0, 0, pad_width, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                # Pad top and bottom
                new_height = int(w / desired_aspect_ratio)
                pad_height = (new_height - h) // 2
                padded_image = cv2.copyMakeBorder(save_me, pad_height, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
            # Resize image to standard size
            resized_image = cv2.resize(padded_image, standard_size, interpolation=cv2.INTER_AREA)
            # base_img  = cv2.imread("base_gestures/start_base.png")
            start = match_gestures(handedness, resized_image)
            
            if len(start) != 0:
                cv2.putText(canvas, start, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Hand: " + handedness, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if cv2.waitKey(10) & 0xFF == ord('s'):
                img_name = f"cropped_hand_{img_counter}.png"
                cv2.imwrite("saved_imgs/" + img_name, resized_image)
                print(f"{img_name} saved.")
                img_counter += 1   

    return canvas, start

#process the gestures & send to controller
def process_gesture(gesture, nx, controller_idx):
    if gesture == "speed_inc":
        send_it2.speed_up(nx, controller_idx)
    elif gesture == "speed_dec":
        send_it2.slow_down(nx, controller_idx)
    elif gesture == "to_right":
        send_it2.turn_right(nx, controller_idx)
    elif gesture == "to_left":
        send_it2.turn_left(nx, controller_idx)
    elif gesture == "bumper": 
        print("bumper!")

##### VIDEO CAPTURE! #####
#video capture, display, and process gestures
def cap_video():
    cap = cv2.VideoCapture('/dev/video0') 
    cap.set(cv2.CAP_PROP_FPS, 30)
    nx, controller_index = send_it2.connect_controller()
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("no ret")
                break  

            canvas, gesture = read_frame(frame, hands)
            process_gesture(gesture, nx, controller_index)
            print("hello")

            #display
            cv2.imshow('Hand Skeleton', canvas)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

########## TESTING! ##########
#no camera no problem
def mimic_capture():
    #set up controller
    nx, controller_index = send_it2.connect_controller()

    #set up time constants
    frame_rate = 60
    delay = 1.0 / frame_rate
    start_time = time.time()

    #for testing
    counter = 0

    while True:
        elapsed_time = time.time() - start_time
        time_to_wait = delay - elapsed_time
        
        #run some test commands!
        if counter == 0: 
            send_it2.turn_right(nx, controller_index)
            time.sleep(0.2)
            send_it2.turn_left(nx, controller_index)
            time.sleep(0.2)
            send_it2.speed_up(nx, controller_index)
            counter += 1

        if time_to_wait > 0:
            time.sleep(time_to_wait)

#run it!
# cap_video()
            

########## TESTING MEDIAPIPE ##########
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.zeros_like(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image 


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)            

def cap_video_mp():
    cap = cv2.VideoCapture("/dev/video0")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame_mp(frame)
        # Display the frame
        cv2.imshow('MediaPipe Pose', result)

        # Exit if 'q' keypyt
        cv2.waitKey(1)

def process_frame_mp(frame):
     # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(mp_image)

    # Process the frame with MediaPipe Pose
    result = draw_landmarks_on_image(frame_rgb, detection_result)
    return result

cap_video_mp()