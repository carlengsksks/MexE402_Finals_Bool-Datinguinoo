# MexE402_Finals_Bool-Datinguinoo
# Tracking a Ball in a Video
Use the HSV-based object detection code to track a colored ball in a recorded video.
## Ball tracking in a Volleyball Game: Choco Mucho vs. Petro Gazz

## Introduction
Tracking objects on videos is a fundamental task in computer vision. It has a wide range of application from surveillance and sports analysis to robotics. In sports anaylsis, specifically volleyball, it is crucial to track a ball's movement accurately, understanding gameplays and improve overall performace. 

This project focuses on ball tracking of a volleyball match between Choco Mucho and Petro Gazz. The task involves identifying and following the ball throughout the video, despite challenges such as rapid movements, occlusions, and the presence of similarly colored elements. Successful ball tracking provides insights into play strategies, ball trajectories, and match statistics, offering valuable tools for coaches, analysts, and fans.

## Abstract
This project aims to develop a computer vision-based system for tracking the volleyball in a match between Choco Mucho and Petro Gazz. The primary objective is to accurately identify and follow the ball throughout the video, despite challenges such as rapid motion, occlusions, and visually similar objects in the scene.

The approach involves preprocessing video frames to enhance visual features, applying detection models for ball localization, and employing tracking algorithms to ensure continuity. YOLO (You Only Look Once) is used for detection to achieve optimal results.

The expected outcome of the project is a robust and efficient ball tracking system capable of providing accurate and continuous ball trajectories throughout the match. This has potential applications in sports analytics, allowing coaches and analysts to extract meaningful insights into game strategies, player performance, and match statistics.

## Project Methods
Here is the step-by-step processs on how we achieved the goal of this project.

### Import all necesarry libraries



- Video Preprocessing

  - Import the volleyball match video.
  - Convert video frames to a suitable format and resolution for processing.
  - Apply noise reduction and image enhancement techniques to improve frame quality.
