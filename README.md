# Multimodal Integration of Motor Imagery BCI, VR and Robots

> A cognitive workload–aware BCI–VR–robotics framework that integrates motor imagery EEG decoding, adaptive virtual reality training, and assistive exoskeleton control with optimized EEG channel selection for portable rehabilitation.

![Project Overview](project%20display.png)
<!-- If the image doesn't show, rename or replace with your own figure. -->

---

## 🧠 Project Overview

This project builds a **multimodal motor imagery (MI) brain–computer interface (BCI) system** that fuses:

- **High-density EEG** for decoding lower-limb motor imagery,
- **Adaptive virtual reality (VR) tasks** that adjust difficulty based on cognitive workload,
- **Assistive lower-limb robotics** (exoskeleton) for engaging rehabilitation training.

By combining **cognitive workload detection** with **hierarchical EEG channel selection**, the system aims to deliver **accurate, portable, and engaging rehabilitation** for users who need gait assistance or lower-limb functional recovery.

---

## 🎯 Core Objectives

### 1. Cognitive Load Detection

- Design EEG-based models to estimate **three levels of cognitive workload** (Low / Medium / High).
- Use workload estimation to:
  - Filter out trials with unstable brain states,
  - Adapt VR task difficulty in real time,
  - Maintain a balance between challenge and comfort in rehabilitation.

### 2. Hierarchical Density–Entropy Channel Selection

- Start from **high-density EEG** and automatically select a **compact subset of electrodes**.
- Use **density–entropy–based criteria** to:
  - Emphasize “hard” trials near decision boundaries,
  - Preserve discriminative information while reducing channel count,
  - Improve portability and reduce computational cost.

### 3. BCI–VR–Robot Integration

- Connect **MI-BCI**, **VR environments** and **assistive exoskeleton** into a unified loop:
  - Motor imagery → decoded into control commands,
  - VR scene responds and provides visual feedback,
  - Robot executes lower-limb assistance in sync with user intention and workload state.

---

## 🏗️ Repository Structure

The repository is organized into the following main components:

```bash
.
├─ channel selection/
│  └─ Scripts / notebooks for density–entropy–based EEG channel selection
│
├─ data collection and processing/
│  └─ Pipelines for EEG data acquisition, preprocessing, and feature extraction
│
├─ example of brainwave file/
│  └─ Example EEG/brainwave file format for reference
│
├─ example result - fig/
│  └─ Example figures and visualization results
│
├─ demo.gif
│
├─ project display.pdf / project display.png
│  └─ Project poster / overview figure
│
└─ slide.pdf
   └─ Project presentation slides
