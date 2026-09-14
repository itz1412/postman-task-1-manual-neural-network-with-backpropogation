import numpy as np
import os


# defining the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# defining the derivative of sigmoid
def diffsig(x):
    return np.exp(-x) / ((1 + np.exp(-x)) ** 2)


# 
# INPUT DATA
# 

X = np.array([
    [1, 55, 35],
    [2, 60, 40],
    [2, 65, 45],
    [3, 60, 42],
    [3, 70, 50],
    [4, 65, 55],
    [4, 75, 60],
    [5, 70, 65],
    [5, 80, 70],
    [6, 75, 72]
])


# expected output
Y = np.array([
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1]
])


# 
# SCALE THE INPUT
# 

X = X / np.array([8, 100, 100])


# 
# INITIALIZE WEIGHTS AND BIASES
# 

weights = []
bais = []

W1 = np.random.randn(3, 4)
W2 = np.random.randn(4, 1)

weights.append(W1)
weights.append(W2)

B1 = np.zeros((1, 4))
B2 = np.zeros((1, 1))

bais.append(B1)
bais.append(B2)


print("New model created.")


# 
# LEARNING RATE
# 

learning_rate = 0.01


# --------------------------------
# TRAINING LOOP
# --------------------------------

for epoch in range(1000):

    total_loss = 0

    for i in range(len(X)):

        # take one student at a time
        x = X[i:i+1]
        y = Y[i:i+1]


        # --------------------------------
        # FORWARD PROPAGATION
        # --------------------------------

        Z1 = (x @ weights[0]) + bais[0]
        A1 = sigmoid(Z1)

        Z2 = (A1 @ weights[1]) + bais[1]
        A2 = sigmoid(Z2)


        # calculate loss
        loss = 0.5 * (A2 - y) ** 2

        total_loss += loss.item()


        # --------------------------------
        # BACK PROPAGATION
        # --------------------------------

        dA2 = A2 - y

        dZ2 = dA2 * diffsig(Z2)

        DW2 = A1.T @ dZ2
        DB2 = dZ2

        dA1 = dZ2 @ weights[1].T

        dZ1 = dA1 * diffsig(Z1)

        DW1 = x.T @ dZ1
        DB1 = dZ1


        # --------------------------------
        # UPDATE WEIGHTS AND BIASES
        # --------------------------------

        weights[1] -= learning_rate * DW2
        bais[1] -= learning_rate * DB2

        weights[0] -= learning_rate * DW1
        bais[0] -= learning_rate * DB1


    # print loss every 100 epochs
    if epoch % 100 == 0:

        average_loss = total_loss / len(X)

        print(
            "Epoch:",
            epoch,
            "Loss:",
            average_loss
        )


# --------------------------------
# SAVE TRAINED MODEL
# --------------------------------

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model.npz"
)

np.savez(
    model_path,
    W1=weights[0],
    W2=weights[1],
    B1=bais[0],
    B2=bais[1]
)

print("Model saved.")


# --------------------------------
# PREDICTION
# --------------------------------

# New student
# Study hours = 4
# Attendance = 72
# Previous test score = 58

new_student = np.array([
    [4, 72, 58]
])


# Scale the new student's input
new_student = new_student / np.array([
    8, 100, 100
])


# --------------------------------
# FORWARD PROPAGATION FOR NEW STUDENT
# --------------------------------

Z1 = new_student @ weights[0] + bais[0]

A1 = sigmoid(Z1)

Z2 = A1 @ weights[1] + bais[1]

A2 = sigmoid(Z2)


# --------------------------------
# FINAL PREDICTION
# --------------------------------

prediction = A2.item()

print("\nNew Student Prediction:", prediction)


# convert prediction into Pass / Fail

if prediction >= 0.5:

    print("Result: Pass")

else:

    print("Result: Fail")



