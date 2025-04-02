# 🎮 Gesture-Controlled Mario Kart on Nintendo Switch

Real-time hand gesture recognition system that allows users to play *Mario Kart* on the Nintendo Switch using only hand movements—no physical controllers required.

Custom-trained YOLOv8 model for hand detection, Google’s MediaPipe for hand keypoint estimation, and ORB-based computer vision matching, system recognizes player gestures and relays them to the Switch by emulating JoyCon Bluetooth protocols. It runs on an NVIDIA Jetson Orin and achieves a latency of 165ms and 30fps.

---

## 🚀 Features

- 🎯 **Hand Gesture Recognition** via YOLOv8 + MediaPipe  
- ⚙️ **Real-Time Matching** using ORB, FAST, and BRIEF descriptors  
- 🔄 **Bluetooth Emulation** to spoof JoyCon HID protocols  
- 🎮 **Control Mario Kart** with natural hand motions  
- ⚡ **Low Latency (165ms)** and **Smooth FPS (30fps)** gameplay  

---

## 🧠 System Overview

1. **YOLOv8** detects bounding boxes for hands in each video frame.  
2. **MediaPipe** overlays a 21-point hand skeleton.  
3. **ORB + BFMatcher** compares hand structure to preset gestures.  
4. **nxbt (Python Bluetooth API)** emulates JoyCon inputs.  
5. **Jetson Orin** relays inputs to the Switch.

---

## ✋ Recognized Gestures

| Gesture        | Action            |
|----------------|--------------------|
| ✋ Open Palm    | Brake              |
| 🤚 Tilt Right   | Steer Right        |
| 🤚 Tilt Left    | Steer Left         |
| ✊ Fist         | Accelerate         |
| 👉 Point Right  | Use Item (Right)   |
| 👉 Point Left   | Use Item (Left)    |

---

## 🧰 Tech Stack

- **Hardware**: NVIDIA Jetson Orin  
- **Machine Learning**: YOLOv8, MediaPipe, ORB  
- **Computer Vision**: OpenCV, FAST, BRIEF  
- **Bluetooth Emulation**: [nxbt](https://github.com/Brikwerk/nxbt)  
- **Languages**: Python

---

## 📦 Installation & Setup

### Requirements

- NVIDIA Jetson Orin (Ubuntu 18)  
- Python 3.8+  
- Camera (USB or CSI)  
- Nintendo Switch

### directory 
- handpose.ipynb --> runs YOLO + mediapipe + hand gesutres
- gesture_sim.ipynb --> to save binary feature vectors of base gestures 
- handposeF.py --> python script for YOLO + mediapipe + hand gestures
- send_it.py --> sudo python3 send_it.py connects to switch

### Clone & Install

```bash
git clone https://github.com/yourusername/gesture-kart-switch.git
cd gesture-kart-switch
pip install -r requirements.txt
