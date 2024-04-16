#hand gesture, pose detection using yolo 
#author @ taylor tam

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
import time
import send_it2


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
#TODO: take ns, controller_index
def process_gesture(gesture):
    if gesture == "speed_inc":
        print("speed_inc")
    elif gesture == "speed_dec":
        print("speed_dec")
    elif gesture == "to_right":
        print("to_right")
    elif gesture == "to_left":
        print("to left")

##### VIDEO CAPTURE! #####
#video capture, display, and process gestures
def cap_video():
    camera_indices = find_available_cameras()
    cap = cv2.VideoCapture(camera_indices[0]) 
    print("this works!!")
    #TODO uncomment this :) && import sendit2
    nx, controller_index = send_it2.connect_controller()
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  

            canvas, gesture = read_frame(frame, hands)
            #TODO: SEND ns, controller_index
            process_gesture(gesture)

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
            
#find cameras
def find_available_cameras(max_to_test=10):
    available_indices = []
    for i in range(max_to_test):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)  # Using V4L2 backend
        if cap.isOpened():
            available_indices.append(i)
            cap.release()
        else:
            break  # Stop the loop if no camera is found at the current index
    return available_indices
        
#run it!
cap_video()