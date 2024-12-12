# MexE402_Finals_Bool-Datinguinoo
# Tracking a Ball in a Video
Use the HSV-based object detection code to track a colored ball in a recorded video.
## Ball tracking in a Volleyball Game: Choco Mucho vs. Petro Gazz

### I. Introduction
Tracking objects in videos is a fundamental task in computer vision, with a wide range of applications, including surveillance, sports analysis, and robotics. In sports analysis, particularly volleyball, accurately tracking a ball's movement is crucial for understanding gameplay and improving overall performance.

This project focuses on ball tracking in a volleyball match between Choco Mucho and Petro Gazz. The task involves identifying and following the ball throughout the video, overcoming challenges such as rapid movements, occlusions, and similarly colored elements in the scene. Accurate ball tracking provides insights into play strategies, ball trajectories, and match statistics, offering valuable tools for coaches, analysts, and fans.

### II. Abstract
This project aims to develop a computer vision-based system to track the volleyball during a match between Choco Mucho and Petro Gazz. The primary objective is to accurately identify and follow the ball throughout the video, addressing challenges such as rapid motion, occlusions, and visually similar objects.

The approach involves preprocessing video frames to enhance visual features, applying detection models for ball localization, and employing tracking algorithms to maintain continuity. YOLO (You Only Look Once) is used for detection to achieve optimal results.

The expected outcome is a robust and efficient ball tracking system capable of providing accurate and continuous ball trajectories throughout the match. This has potential applications in sports analytics, enabling coaches and analysts to derive meaningful insights into game strategies, player performance, and match statistics.

### III. Project Methods
Here are the step-by-step process to achieve our goal:

Video Preprocessing: Enhancing visual features in video frames.
Detection: Using YOLO for ball localization in each frame.
Tracking: Applying algorithms to ensure continuous ball tracking despite occlusions and rapid motion.

### 1. Import all necessary libraries

```python
!nvidia-smi
```
<img width="838" alt="Screen Shot 2024-12-12 at 17 11 38" src="https://github.com/user-attachments/assets/c46577c0-0d37-47af-920b-f40b089e2540" />

```python
!pip install ultralytics
```
<img width="1058" alt="Screen Shot 2024-12-12 at 17 11 58" src="https://github.com/user-attachments/assets/25c3d74a-5ec4-4766-8829-ff6eea08be24" />
<img width="1058" alt="Screen Shot 2024-12-12 at 17 12 24" src="https://github.com/user-attachments/assets/a9dd40ae-e8b7-4ee6-a81b-08f591c34948" />

```python
from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
display.clear_output()

!yolo mode=checks
```
<img width="1058" alt="Screen Shot 2024-12-12 at 17 12 37" src="https://github.com/user-attachments/assets/bc01744d-474d-42f5-9b7c-586b9f603b1e" />
<img width="1058" alt="Screen Shot 2024-12-12 at 17 12 59" src="https://github.com/user-attachments/assets/c27e11d0-f2cf-4911-bc4c-4ba614a3117f" />

```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="r2tVeGW8kcAFzcMUF1St")
project = rf.workspace("bogart").project("volleyballgame")
version = project.version(2)
dataset = version.download("yolov8")
```
<img width="1058" alt="Screen Shot 2024-12-12 at 17 13 08" src="https://github.com/user-attachments/assets/c02d706d-7385-4b27-82b1-fe083ececbae" />
<img width="1058" alt="Screen Shot 2024-12-12 at 17 13 26" src="https://github.com/user-attachments/assets/87c7f894-c444-4934-9a43-1becf1f160fe" />

```python
!yolo task=detect mode=train model=yolov8m.pt data={dataset.location}/data.yaml epochs=20 imgsz=640 plots=True
```

<img width="1058" alt="Screen Shot 2024-12-12 at 19 09 31" src="https://github.com/user-attachments/assets/234adc20-a84e-4ca8-b252-6442450197b6" />
<img width="1051" alt="Screen Shot 2024-12-12 at 19 10 04" src="https://github.com/user-attachments/assets/610c70d7-97cb-4461-b725-1d4d8286646e" />
<img width="1051" alt="Screen Shot 2024-12-12 at 19 27 23" src="https://github.com/user-attachments/assets/5fe392af-9176-488b-85f2-32dae2dacc8e" />
<img width="1051" alt="Screen Shot 2024-12-12 at 19 28 08" src="https://github.com/user-attachments/assets/95629ec0-76dc-405a-ba17-6aa748898430" />
<img width="1051" alt="Screen Shot 2024-12-12 at 19 28 18" src="https://github.com/user-attachments/assets/75dc5112-5a52-491e-8881-f30b74bd4bde" />
<img width="1051" alt="Screen Shot 2024-12-12 at 19 28 35" src="https://github.com/user-attachments/assets/a573252f-5074-459d-8726-1bc456749895" />

### 2. Create Visualization

```python
Image(filename=f'/content/runs/detect/train/confusion_matrix.png')
```
![CON](https://github.com/user-attachments/assets/8d69972e-cd6a-4391-b468-f942c8348e45)

```python
Image(filename=f'/content/runs/detect/train/results.png')
```
![GRAPH](https://github.com/user-attachments/assets/b5cc3f5b-58ac-4f78-8341-fff3a3256701)

### 3. Validate a Trained Object Detection Model

```python
!yolo task=detect mode=val model=/content/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
```
<img width="1051" alt="Screen Shot 2024-12-12 at 20 00 50" src="https://github.com/user-attachments/assets/5e3ffed6-18af-4068-be80-a0feec63b638" />

```python
!yolo task=detect mode=predict model=/content/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True
```
<img width="1051" alt="Screen Shot 2024-12-12 at 20 01 07" src="https://github.com/user-attachments/assets/6dbb4ea0-630d-4bba-a14e-29af75936009" />

### Create Path and Display Output

```python
import glob
from IPython.display import Image, display

for image_path in glob.glob('/content/runs/detect/predict/*.jpg'):
  display(Image(filename=image_path, width=600))
  print("\n")
```
![V1](https://github.com/user-attachments/assets/45aef8ca-3b88-4572-adbc-c14573252ad2)
![V2](https://github.com/user-attachments/assets/222a204b-7ffb-46a6-bb2e-d3ec7502efc1)
![V3](https://github.com/user-attachments/assets/4e6bb46f-99c3-4ed1-9e1c-7cab645a4427)
![V4](https://github.com/user-attachments/assets/6ca416c5-d43d-4d17-9b6d-56015419634a)
![V5](https://github.com/user-attachments/assets/f9ee902e-f05b-4d7d-9e3a-d9d254f28499)



