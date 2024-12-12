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
The project employs the following step-by-step process to achieve its goals:

Video Preprocessing: Enhancing visual features in video frames.
Detection: Using YOLO for ball localization in each frame.
Tracking: Applying algorithms to ensure continuous ball tracking despite occlusions and rapid motion.

### Import all necesarry libraries

```python
!nvidia-smi
```
<img width="838" alt="Screen Shot 2024-12-12 at 17 11 38" src="https://github.com/user-attachments/assets/c46577c0-0d37-47af-920b-f40b089e2540" />

