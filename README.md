# Autonomous Drone Navigation in GPS-Denied Environments

## Overview

This project presents an autonomous drone navigation framework capable of exploring unknown environments without relying on GPS. The system combines Frontier-Based Exploration, Occupancy Grid Mapping, and graph-based path planning algorithms to enable efficient exploration, obstacle avoidance, and autonomous return to the starting location.

The project was developed during my Research Internship at the Centre of Excellence in Product Design & Smart Manufacturing (CEPDSM), Maulana Azad National Institute of Technology (MANIT), Bhopal.

---

## Problem Statement

Traditional drone navigation depends heavily on GPS signals. However, GPS is unavailable or unreliable in environments such as:

- Indoor buildings
- Underground tunnels
- Disaster zones
- Dense forests
- Industrial facilities

This project addresses the challenge by enabling autonomous exploration and navigation using onboard sensing and intelligent path planning.

---

## Objectives

- Explore unknown environments autonomously.
- Generate a real-time occupancy grid map.
- Detect obstacles during exploration.
- Identify unexplored frontier regions.
- Compute the optimal return path after exploration.
- Navigate safely without GPS.

---

## Technologies Used

- Python
- MuJoCo Simulator
- Occupancy Grid Mapping
- Frontier-Based Exploration
- A* Algorithm
- Dijkstra's Algorithm
- Graph Search
- VS Code
- Git & GitHub

---

## Project Workflow

### Step 1: Environment Initialization
- Load the simulation environment in MuJoCo.
- Initialize the drone and occupancy grid.

### Step 2: Autonomous Exploration
- Detect frontier cells between explored and unexplored regions.
- Select the nearest frontier.
- Navigate towards the selected frontier.

### Step 3: Environment Mapping
- Update the occupancy grid continuously.
- Mark free space and obstacles.
- Build a complete map of the environment.

### Step 4: Path Planning
- Apply A* Algorithm for efficient navigation.
- Use Dijkstra's Algorithm to compute the optimal return path.

### Step 5: Autonomous Return
- After completing exploration, the drone returns safely to the starting position using the shortest available path.

---

## Features

- GPS-independent navigation
- Frontier-Based Exploration
- Real-time Occupancy Grid Mapping
- Autonomous obstacle avoidance
- Optimal path planning
- Autonomous return-to-base
- MuJoCo-based simulation

---

## Algorithms Used

### Frontier-Based Exploration
Identifies the boundary between explored and unexplored regions to maximize exploration efficiency.

### Occupancy Grid Mapping
Represents the environment as a grid where each cell stores occupancy information.

### A* Algorithm
Computes an efficient path to exploration targets while avoiding obstacles.

### Dijkstra's Algorithm
Calculates the shortest path from the exploration endpoint back to the starting position.

---

## Project Structure

```
Autonomous-Drone-Navigation/
│
├── src/
│   ├── exploration.py
│   ├── mapping.py
│   ├── path_planning.py
│   ├── controller.py
│
├── assets/
│
├── models/
│
├── results/
│
├── README.md
│
└── requirements.txt
```

---

## Results

- Successfully explored unknown environments.
- Generated occupancy grid maps in real time.
- Detected obstacles autonomously.
- Computed optimal exploration and return paths.
- Demonstrated reliable GPS-denied navigation within the MuJoCo simulator.

---


---

## Skills Demonstrated

- Python Programming
- Robotics Simulation
- Autonomous Navigation
- Occupancy Grid Mapping
- Frontier-Based Exploration
- Path Planning
- A* Algorithm
- Dijkstra's Algorithm
- Problem Solving
- Research & Technical Documentation

---
