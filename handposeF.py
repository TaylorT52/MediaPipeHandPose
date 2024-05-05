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
#Load some stuff!
weights_loc = "weights/best.pt"
yolo_model = YOLO(weights_loc)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#Use GPU
print("CUDA available: ", torch.cuda.is_available())
print("Current device: ", torch.cuda.current_device())
print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)           

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

#Starting variables
padding = 30

#Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
desired_aspect_ratio = 1 
standard_size = (350, 350)
gestures = ["speed_inc", "speed_dec", "to_right", "to_left", "bumper_right", "bumper_left"]
orb = cv2.ORB_create()

#load base gestures
with open("base_gestures_new.json", "r") as infile:
    data = json.load(infile)

##### GESTURE MATCHING #####
#match gestures w/ orb + bf 
def match_gestures(handedness, img2, threshold=110):
    img2_processed = img2
    
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

##### MEDIAPIPE! #####
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_dir = "Right"
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.zeros_like(rgb_image)
    max_x, max_y, min_x, min_y = 0,0,0,0

    if len(hand_landmarks_list) > 0:
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
            min_x = int(min(x_coordinates) * width) - MARGIN
            min_y = int(min(y_coordinates) * height) - MARGIN
            max_x = int(max(x_coordinates) * width) + MARGIN
            max_y = int(max(y_coordinates) * height) + MARGIN

            hand_dir = handedness[0].category_name

    annotated_image = cv2.flip(annotated_image, 1)
    return hand_dir, max_x, max_y, min_x, min_y, annotated_image     

def cap_video_mp():
    #print FPS
    frame_count = 0
    total_time = 0
    start_time = time.time()

    ###### capture a video ######
    cap = cv2.VideoCapture("/dev/video2")
    nx, controller_index = send_it2.connect_controller()
    print(f"Capture is working: {cap.isOpened()}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("no ret!")
            break

        ###### process frame w/ mp ######
        handedness, max_x, max_y, min_x, min_y, result = process_frame_mp(frame)

        ###### crop and add padding ######
        if max_x != 0:
            new_min_x = result.shape[1] - max_x
            new_max_x = result.shape[1] - min_x
            cropped = result[min_y:max_y, new_min_x:new_max_x]
            h, w = cropped.shape[:2]
            if h != 0:
                current_aspect_ratio = w / h
            
                # Calculate padding
                if current_aspect_ratio < desired_aspect_ratio:
                    new_width = int(desired_aspect_ratio * h)
                    pad_width = (new_width - w) // 2
                    padded_image = cv2.copyMakeBorder(cropped, 0, 0, pad_width, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                else:
                    new_height = int(w / desired_aspect_ratio)
                    pad_height = (new_height - h) // 2
                    padded_image = cv2.copyMakeBorder(cropped, pad_height, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
                resized_image = cv2.resize(padded_image, standard_size, interpolation=cv2.INTER_AREA)

                ###### match to a gesture ######
                start = match_gestures(handedness, resized_image)
                print(start)
                process_gesture(start, nx, controller_index)

                #Fps shenanigans
                frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - start_time

                if elapsed_time > 1:  # Update every second
                    fps = frame_count / elapsed_time
                    print("Average FPS:", fps)
                    frame_count = 0
                    start_time = time.time()

    cap.release()
    print('Game finished!')

def process_frame_mp(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(mp_image)
    result = draw_landmarks_on_image(frame_rgb, detection_result)
    return result

cap_video_mp()