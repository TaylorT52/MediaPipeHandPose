##### IMPORTS #####
import mediapipe as mp
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import os
from ultralytics.utils.plotting import Annotator 
import json
from google.protobuf.json_format import MessageToDict
import os


##### LOADING STUFF #####
#load some stuff!
weights_loc = "weights/best.pt"
yolo_model = YOLO(weights_loc)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#starting variables
padding = 30
img_counter = 0
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
desired_aspect_ratio = 1 
standard_size = (350, 350)
gestures = ["speed_inc", "speed_dec", "to_right", "to_left"]

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
            
            cv2.putText(canvas, "Hand: " + handedness, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if cv2.waitKey(10) & 0xFF == ord('s'):
                img_name = f"cropped_hand_{1}.png"
                cv2.imwrite("saved_imgs/" + img_name, resized_image)
                print(f"{img_name} saved.")

    return canvas, start

#video capture, display, and process gestures
def cap_video_comp():
    cap = cv2.VideoCapture(1) 
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("no ret")
                break  

            canvas, gesture = read_frame(frame, hands)

            #display
            cv2.imshow('Hand Skeleton', canvas)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

cap_video_comp()