🎮 Gesture-Controlled Mario Kart on Nintendo Switch

This project implements a real-time hand gesture recognition system that allows users to play Mario Kart on the Nintendo Switch using only hand movements—no physical controllers required.

Using a custom-trained YOLOv8 model for hand detection, Google’s MediaPipe for hand keypoint estimation, and ORB-based computer vision matching, the system recognizes player gestures and relays them to the Switch by emulating JoyCon Bluetooth protocols. It runs on an NVIDIA Jetson Orin and achieves a latency of 165ms and 30fps.

# hand poses! 
- handpose.ipynb --> runs YOLO + mediapipe + hand gesutres
- gesture_sim.ipynb --> to save binary feature vectors of base gestures 
- handposeF.py --> python script for YOLO + mediapipe + hand gestures
- send_it.py --> sudo python3 send_it.py connects to switch
# some notes
- jetson orin, ubuntu 18
