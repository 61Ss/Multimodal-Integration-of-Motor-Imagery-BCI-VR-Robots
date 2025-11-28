# Multimodal Integration of Motor Imagery BCI, VR and Robots

> A cognitive workload–aware BCI–VR–robotics framework that integrates motor imagery EEG decoding, adaptive virtual reality training, and assistive exoskeleton control with optimized EEG channel selection for portable rehabilitation.

---

## 🎥 Demo

<p align="center">
  <img src="demo.gif" alt="BCI–VR–Robot integration demo" width="70%">
</p>

A short teaser of our **closed-loop BCI–VR–robot** pipeline: motor imagery EEG is decoded in real time, VR feedback adapts accordingly, and the assistive robot responds to user intention and cognitive workload.

---

## 🖼 Project Poster

<p align="center">
  <img src="project%20display.png" alt="Project overview poster" width="70%">
</p>

> SURF-2025-0497 · School of Advanced Technology  
> Supervisors: **Dr. Rui Yang**, **Dr. Mengjie Huang**  
> Team: **Bihao You**, **Yize Liu**, **Yun Zhang**, **Yutong Zhu**, **Zihan Yu**

---

## 🧠 Overview

This repository contains the code, data examples, and figures for our Summer Undergraduate Research Fellowship (SURF) project:

> **Multimodal Integration of Motor Imagery Brain–Computer Interface (BCI) with Adaptive Virtual Reality (VR) Environments for Assistive Robotics**

The project builds a **multimodal motor imagery BCI** system that fuses:

- **High-density EEG** for decoding lower-limb motor imagery (MI),
- **Adaptive VR tasks** that adjust difficulty based on **cognitive workload**,
- **Assistive lower-limb robotics (exoskeleton)** for engaging rehabilitation training.

Our framework aims to:

1. **Sense** the user’s motor intention and mental workload from EEG in real time,  
2. **Adapt** VR task difficulty and robot assistance dynamically,  
3. **Optimize** EEG channel subsets for **portability, robustness, and efficiency**.

---

## 🎯 Core Objectives

### Objective 1 — Cognitive Load Detection

Early motor imagery experiments showed that **high cognitive workload** can destabilize MI performance. This motivates a **real-time workload detection** module that:

- Classifies **three workload levels**: Low (0), Medium (1), High (2),
- Filters out high-load trials to stabilize EEG signals,
- Provides workload labels to **adapt VR difficulty** and **robot assistance**.

**Key points:**

- EEG features include time-, frequency-, and complexity-based descriptors (e.g., variance, skewness, kurtosis, entropy, cross-channel relations).
- A tree-based model (e.g., **XGBoost**) performs 3-class workload classification.
- A **two-stage strategy**:
  1. Stage 1: Low vs (Medium + High),
  2. Stage 2: Medium vs High.
- Results:
  - Initial 3-class accuracy ≈ **82.5%**,
  - Two-stage strategy improves overall accuracy by ≈ **6%**,
  - F1-scores for **Medium** and **High** workload improve by ≈ **12%**,
  - t-SNE visualization shows **clearer separation** between workload clusters after hierarchical modeling.

---

### Objective 2 — Collaborative Hierarchical Density–Entropy Channel Selection (CHDECS)

High-density EEG (many electrodes) is powerful but **impractical** for daily rehabilitation: heavy setup, long preparation, high computational cost, and poor portability. We propose **CHDECS**, a hierarchical channel selection framework that:

- Identifies **compact yet informative** subsets of EEG channels (e.g., top-10/20/30),
- Focuses on **“hard” trials** near decision boundaries via **density–entropy optimization**,
- Balances **task-level**, **subject-level**, and **group-level** information.

**Core ideas:**

1. **Multi-View Collaborative Information Gain**  
   - Integrates time-, frequency-, and energy-domain features,  
   - Models local electrode interactions from a **game-theoretic perspective**.

2. **Density–Entropy Optimization**  
   - Uses dual thresholds on **sample density** near the decision boundary and **prediction entropy**,  
   - Emphasizes informative, ambiguous trials that shape the classifier’s boundary.

3. **Hierarchical Channel Selection**  
   - **Task level**: captures global discriminative patterns,  
   - **Subject level**: adapts to individual variability,  
   - **Group level**: promotes cross-subject generalization.

**Experimental highlight:**

- CHDECS-selected **top-10/20/30 channels** achieve accuracy close to (or comparable with) using **all channels**,  
- While significantly **reducing computation, setup time, and hardware complexity**,  
- Laying the foundation for **lightweight, portable** BCI systems deployable in real rehabilitation scenarios.

---

### Objective 3 — Tri-Modal BCI–VR–Robot Integration

Objective 3 integrates **exoskeletons** with VR/BCI (Objectives 1 & 2) to form a **closed-loop, brain-driven rehabilitation system**:

- **BCI EEG Cap — Emotiv Epoc Flex**  
  - Provides MI and workload-related EEG with a **reduced, optimized channel set** from CHDECS.

- **VR Headset — Meta Quest 3**  
  - Presents lower-limb tasks (e.g., leg raise, stepping, walking) with adjustable difficulty,  
  - Adapts target size, timing, obstacle density, or scene complexity based on **real-time workload and performance**.

- **Assistive Robot — MileBot Max Exoskeleton**  
  - Delivers precise lower-limb assistance (e.g., hip/knee joint flexion–extension),  
  - Assistance level adapts to:
    - Current **cognitive load** (reduce assistance to increase challenge, or increase when overloaded),
    - **Task success rate** and fatigue.

**Integrated pipeline (conceptual):**

> EEG Acquisition → Preprocessing & CHDECS → MI Decoding + Workload Estimation → VR Difficulty Adapter → Robot Controller → Multimodal Feedback (visual, proprioceptive, performance logs)

This integration transforms **repetitive rehabilitation** into **interactive, game-like challenges**, improving **engagement**, **effectiveness**, and potentially **recovery speed**.

---

## 🧪 Experiments & Results (High-Level)

### Experiment 1 — Cognitive Workload Classification

- **Goal**: Build a generalized, real-time classifier for 3-level cognitive workload based on EEG.
- **Design**: Task paradigm with three difficulty levels (Low / Medium / High) inducing different workload states.
- **Method**:
  - Feature extraction from multi-channel EEG,
  - XGBoost-based 3-class classifier,
  - Two-stage refinement (Low vs [Mid+High], then Mid vs High).
- **Results**:
  - Baseline 3-class accuracy ≈ **82.5%**,  
  - Two-stage strategy improves accuracy by ≈ **6%**,  
  - **F1-score** for Medium and High workload increases by ≈ **12%**,  
  - Workload clusters become clearly separable in **t-SNE** space.

### Experiment 2 — Motor Imagery & CHDECS Channel Selection

- **Goal**: Evaluate CHDECS on MI classification and quantify performance under different channel budgets.
- **Single-trial structure (example):**
  - t = 0 s: Fixation cross appears with an auditory cue,
  - t = 2 s: Left or right arrow appears for 1 s,
  - t = 2–7 s: Subject performs **motor imagery of the corresponding leg**,
  - t = 7–10 s: Rest period (cross disappears; subject relaxes).
- **Comparison**:
  - Full set of channels vs CHDECS-selected **top-10 / top-20 / top-30** subsets.
- **Findings**:
  - CHDECS maintains **high decoding accuracy** with far fewer channels,
  - Greatly reduces **computational load** and **deployment cost**,
  - Provides a practical route toward **wearable MI-BCI** in clinical rehab.

---

## 🧩 Repository Structure

```bash
.
├─ channel selection/
│  └─ Scripts / notebooks for CHDECS:
│     - Multi-view collaborative information gain
│     - Density–entropy optimization
│     - Hierarchical channel ranking and top-k selection
│
├─ data collection and processing/
│  └─ EEG data pipelines:
│     - Data import and formatting
│     - Preprocessing and artifact handling
│     - Feature extraction for MI and workload
│
├─ example of brainwave file/
│  └─ Example EEG / brainwave file for reference (format / structure)
│
├─ example result - fig/
│  └─ Example figures and plots from experiments
│
├─ demo.gif
│  └─ Short demonstration of BCI–VR–robot integration
│
├─ project display.pdf
├─ project display.png
│  └─ SURF project poster and overview figure
│
└─ slide.pdf
   └─ Project presentation slides
