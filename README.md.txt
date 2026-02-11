# 🤖 Sonso-AI: Advanced Bipedal Locomotion

**Sonso-AI** is a reinforcement learning project focused on bipedal walking and obstacle avoidance within a 2D physics-based environment. This project marks my evolution from traditional Deep Q-Networks (DDQN) to a more sophisticated **Hierarchical Reinforcement Learning (HRL)** approach using **Soft Actor-Critic (SAC)**.

---

## 🧠 Architecture Overview

This agent utilizes a **Master-Worker hierarchy** to decouple strategic decision-making from motor execution:

1.  **High-Level Manager (Heuristic-based):** Analyzes long-range Lidar data to determine the optimal target velocity ($v_{target}$). It switches between "Sprinting" and "Obstacle-Prepping" modes based on environmental threats.
2.  **Low-Level Worker (SAC Policy):** A neural network with a **30-dimensional observation space**. It receives the target velocity as a high-level "command" and translates it into precise motor torques across 4 joints (Hips and Knees).



---

## 🛠 Technical Implementation

### 1. Soft Actor-Critic (SAC)
The agent uses SAC for its ability to handle **continuous action spaces** and its maximum entropy framework. This encourages the agent to explore various gaits, preventing it from getting stuck in "sub-optimal" or "robotic" walking patterns.

### 2. Domain Randomization (DR)
To ensure **robustness**, the training environment randomizes physical properties every episode:
* **Gravity:** Fluctuates between $700.0$ and $1100.0$.
* **Friction:** Ranges from $0.4$ to $2.0$.
* **Mass:** Body mass is randomized to force the agent to learn true balance rather than memorizing a specific weight distribution.

### 3. Sensory Input (Lidar & Proprioception)
* **Long-range Lidar:** Detects upcoming terrain obstacles at a distance of 700px.
* **Joint Sensors:** Real-time tracking of angles, angular velocity, and ground contact for both legs.
* **Command Awareness:** The 30th input neuron is the normalized velocity goal provided by the Manager.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install stable-baselines3 gymnasium pymunk pygame
2. Clone and Train
Bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/Sonso-AI.git](https://github.com/YOUR_USERNAME/Sonso-AI.git)

# Run the training script
python main.py
📈 Training Progress & Resilience
The model successfully recovered from a mid-training power outage thanks to a robust checkpointing system (saving every 50,000 steps). Currently, the agent shows a strong upward trend in ep_rew_mean and ep_len_mean, demonstrating successful convergence and adaptive behavior across different gravity settings.

Developed by a 13-year-old AI enthusiast exploring the frontiers of Robotics and Reinforcement Learning.