# Autonomous Drone Navigation in GPS-Denied Environments

## Overview

This project implements an autonomous drone navigation framework for exploring unknown GPS-denied environments using MuJoCo simulation. The system combines Frontier-Based Exploration, Occupancy Grid Mapping, IMU trajectory tracking, and graph-based path planning to autonomously explore, map, and safely return to the starting position.

The project was developed during my Research Internship at the Centre of Excellence in Product Design & Smart Manufacturing (CEPDSM), Maulana Azad National Institute of Technology (MANIT), Bhopal.

---

## Problem Statement

Autonomous drones operating in indoor, underground, or disaster environments cannot rely on GPS for localization. The objective of this project is to enable safe and efficient autonomous exploration while constructing an environmental map and computing an optimal return path.

---

## Objectives

- Explore unknown environments autonomously.
- Generate real-time occupancy grid maps.
- Detect and rank exploration frontiers.
- Avoid obstacles during navigation.
- Compute the optimal return path using graph search algorithms.
- Evaluate exploration coverage and navigation performance.

---

## Technologies Used

- Python
- MuJoCo Simulator
- Occupancy Grid Mapping
- Frontier-Based Exploration
- A* Algorithm
- Dijkstra's Algorithm
- IMU Trajectory Tracking
- VS Code
- Git & GitHub

---

## Project Workflow

### 1. Environment Initialization
- Load the MuJoCo simulation environment.
- Initialize the drone and occupancy grid map.

### 2. Autonomous Exploration
- Detect unexplored frontier cells.
- Rank candidate frontiers.
- Navigate towards the highest-priority frontier.

### 3. Environment Mapping
- Continuously update the occupancy grid.
- Detect free space and obstacles.
- Build a complete map of the environment.

### 4. Path Planning
- Use A* and Dijkstra's Algorithm to generate efficient navigation and return paths.

### 5. Mission Analysis
- Evaluate exploration coverage.
- Record flight trajectory.
- Analyze mapping performance and navigation efficiency.

---

## Features

- Autonomous Frontier-Based Exploration
- Real-Time Occupancy Grid Mapping
- Dynamic Obstacle Detection
- IMU Trajectory Tracking
- A* Path Planning
- Dijkstra-Based Return Navigation
- Coverage Metrics Evaluation
- Mission Performance Analysis
- MuJoCo Physics-Based Simulation

---

## Algorithms Used

### Frontier-Based Exploration
Detects unexplored frontier regions and guides the drone toward areas that maximize map coverage.

### Occupancy Grid Mapping
Represents the environment as a grid to identify free, occupied, and unexplored regions.

### A* Algorithm
Computes efficient exploration paths while avoiding obstacles.

### Dijkstra's Algorithm
Calculates the shortest return path from the final exploration point to the starting location.

---

## Project Structure

```
Autonomous-Drone-Navigation
│
├── frontier_detection.py
├── frontier_ranking.py
├── occupancy_grid_mapping.py
├── imu_trajectory.py
├── mission_manager.py
├── coverage_metrics.py
├── post_mission_analysis.py
├── sensor_fusion.py
├── dynamic_obstacles.py
├── scene.xml
├── assets/
├── results/
└── README.md
```

---

## Results

- Successfully explored unknown environments autonomously.
- Generated occupancy grid maps in real time.
- Detected and ranked frontier regions for efficient exploration.
- Computed optimal navigation and return paths using A* and Dijkstra's Algorithm.
- Evaluated mission coverage, flight trajectory, and mapping performance within the MuJoCo simulator.

---

## Skills Demonstrated

- Python Programming
- Robotics Simulation
- Autonomous Navigation
- Frontier-Based Exploration
- Occupancy Grid Mapping
- Path Planning
- A* Algorithm
- Dijkstra's Algorithm
- IMU Trajectory Tracking
- Performance Evaluation
- Research & Technical Documentation
