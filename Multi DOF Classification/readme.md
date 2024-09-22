# Improved Classification Control for Multi-DOF EMG-Controlled Exoskeleton

## Introduction
This project aimed to improve classification control over the MyoPro exoskeleton movements in multiple degrees of freedom, facilitated by a wrist add-on extension. The exoskeleton movements (or states) currently implemented are defined as follows:

- **0** – Rest
- **1** – Grasp
- **2** – Open
- **3** – Pronate
- **4** – Supinate

## General Procedure
Prerecorded training datasets for hand gestures were used. The data was first preprocessed, and features were categorized based on the corresponding kinematic position. The preprocessed features (X inputs) and states (Y predictions) were then used to train two models: a Convolutional Neural Network (CNN) and a K-Nearest Neighbor (KNN) classification model. The state predictions are used to control the MyoPro exoskeleton and wrist add-on kinematic positions in real-time.

## Datasets
All offline analyses were conducted using the datasets collected by the NeurRobotics Lab at the University of Utah.

## File Descriptions

### Model Training:
- **`preprocessData.m`**: Function – Aligns features and kinematics, defines states, filters unclassified data points, selects relevant feature channels, and returns features and states ready for model training.
- **`trainAgnosticContinuousCNN.m`**: Function – Adjusts the CNN model with padding for continuous state prediction.
- **`knnClassifier.m`**: Function – Trains a KNN model using preprocessed features and states.

### Offline Analysis:
- **`modelComparison.m`**: Performs a statistical comparison of KNN vs. CNN across datasets, evaluating accuracy, prediction times, ROC curves, and aggregate confusion matrices.
- **`visualizeKinematics.m`**: Visualizes the features, states, and corresponding exoskeleton hand and wrist kinematic positions.
- **`confidenceTesting.m`**: Visualizes and evaluates the impact of confidence level-based state transitions.
- **`cnnVisualization.m`**: Visualizes CNN convolutional layers using feature maps.
- **`knnVisualization.m`**: Visualizes KNN classification decision boundaries.

### Feedback Decode:
The following functions are located in the `FeedbackDecode` dependencies folder and are used to run the classification models in real-time:

- **`trainKNN.m`**: Trains a KNN model after preprocessing input kinematics and features.
- **`trainCNN.m`**: Trains a CNN model after preprocessing input kinematics and features.
- **`testKNN.m`**: Returns predicted kinematics using input features and the trained KNN model.

## Findings
The KNN model had statistically higher accuracy and lower prediction times across the 6 datasets compared to the CNN model.

## Next Steps
- Implement control for additional degrees of freedom (DOFs), such as wrist flexion.
- Add a new class in Feedback Decode for the wrist add-on.
- Evaluate the wrist add-on and control algorithm using an online clothespin relocation test, assessing:
  - **Cognitive load**
  - **Compensation angles**
