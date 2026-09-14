# Manual Neural Network with Backpropagation

## Project Overview

This project implements a basic neural network from scratch using Python and NumPy.

The main objective is to understand and implement forward propagation, loss calculation, manual backpropagation, and gradient descent without relying on machine learning libraries that automatically calculate gradients.

The network is used for a simple student pass/fail classification task.

---

## Problem Statement

The objective is to build a neural network that predicts whether a student is likely to pass or fail based on:

- Study hours
- Attendance
- Previous test score

This is a binary classification problem.

---

## Dataset

The project uses a small synthetic dataset created for educational purposes.

The dataset contains three input features:

1. Study hours
2. Attendance percentage
3. Previous test score

The target variable is:

- `0` → Fail
- `1` → Pass

The dataset is stored in:

`data/dataset.csv`

---

## Network Architecture

The neural network has the following architecture:

**3 → 4 → 1**

- 3 input neurons
- 4 hidden neurons
- 1 output neuron

The hidden layer uses the sigmoid activation function.

The output neuron also uses sigmoid activation because this is a binary classification problem.

---

## Input Scaling

The input features have different numerical ranges, so the inputs are scaled before being passed into the network as i didnt want any one input to be overpowering.

The scaling used is:

- Study hours ÷ 8
- Attendance ÷ 100
- Previous test score ÷ 100

This keeps the input values approximately between 0 and 1 and helps make training more stable. 

---

## Forward Propagation

For the first layer:

Z1 = XW1 + B1

A1 = sigmoid(Z1)

For the output layer:

Z2 = A1W2 + B2

A2 = sigmoid(Z2)

The final output `A2` represents the network's prediction.

---

## Loss Function

The project uses squared error loss:

L = 1/2 (A2 - Y)²

The loss measures the difference between the predicted output and the expected output.this is the value we are basically trying to minimise in our backpropogation maths

---

## Manual Backpropagation

Backpropagation was implemented manually using the chain rule.this is because i was more comfortable with maths being visible to me.

The gradients are calculated layer by layer starting from the output layer and moving backwards through the network.

For the output layer:

dA2 = A2 - Y

dZ2 = dA2 × sigmoid'(Z2)

dW2 = A1ᵀ × dZ2

dB2 = dZ2

For the hidden layer:

dA1 = dZ2 × W2ᵀ

dZ1 = dA1 × sigmoid'(Z1)

dW1 = Xᵀ × dZ1

dB1 = dZ1

---

## Gradient Descent

The parameters are updated using gradient descent:

W = W - learning_rate × dW

B = B - learning_rate × dB

The learning rate used in this project is:

`0.01`

I used a randomly generic rate of learning.

---

## Training

The network is trained for:

`1000 epochs`

Each training example is processed individually and the weights and biases are updated after each example.Im basically training it over 1000 times in one run 

The loss is printed every 100 epochs to monitor training.The loss was seen to be lower every 100 turns.

---

## Training Results

The loss decreased during training, indicating that the network was learning from the training data.

Example:

| Epoch | Loss |
|------:|-----:|
Epoch: 0 Loss: 0.1199569154999475
Epoch: 100 Loss: 0.11856322857111996
Epoch: 200 Loss: 0.11768669106749849
Epoch: 300 Loss: 0.11687190708750213
Epoch: 400 Loss: 0.11604013339566854
Epoch: 500 Loss: 0.11517461775856279
Epoch: 600 Loss: 0.11426885770335524
Epoch: 700 Loss: 0.11331798418858738
Epoch: 800 Loss: 0.11231738491343775
Epoch: 900 Loss: 0.11126252361187135

*The exact values may vary between runs because the network weights are randomly initialized.*

---

## Prediction

After training, the network can be used to make predictions on new students.

Example input:

- Study hours: 4
- Attendance: 72
- Previous test score: 58

The input is scaled and passed through the trained network using forward propagation.

The final sigmoid output is interpreted using a threshold of 0.5:

- Output ≥ 0.5 → Pass
- Output < 0.5 → Fail

---

## Model Saving

The trained weights and biases are saved using NumPy's `.npz` format.It should be noted that this model saving part wasnt done by me as i am not as good as python to be able to deal with stuff like this and therefore some external influence was used.

The saved model contains:

- W1
- W2
- B1
- B2

The file is stored as:

`model.npz`

This allows the trained parameters to be preserved after training.

---

## Technologies Used

- Python
- NumPy

---

## Project Structure

```text
manual-neural-network/
│
├── neural_network.py
├── model.npz
├── README.md
├── requirements.txt
│
└── data/
    └── dataset.csv