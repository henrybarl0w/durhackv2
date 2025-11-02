import random
import math

def generateInputLayer(board, head, apple):
    # board + head + apple
    input_layer = []

    rows = len(board)
    cols = len(board[0])
    hr, hc = head
    ar, ac = apple

    # --- 7x7 grid around head ---
    for dr in range(-3, 3):
        for dc in range(-3, 3):
            r, c = hr + dr, hc + dc
            if 0 <= r < rows and 0 <= c < cols:
                input_layer.append(board[r][c])
            else:
                # outside bounds = treat as wall
                input_layer.append(1)

    # --- distances to walls (normalized 0–1) ---
    dist_top = hr / rows
    dist_bottom = (rows - hr - 1) / rows
    dist_left = hc / cols
    dist_right = (cols - hc - 1) / cols
    input_layer.extend([dist_top, dist_bottom, dist_left, dist_right])

    # --- relative position of apple (normalized) ---
    apple_vert = (ar - hr) / rows    # positive = apple below
    apple_horiz = (ac - hc) / cols   # positive = apple to right
    input_layer.extend([apple_vert, apple_horiz])

    #print(len(input_layer))

    return input_layer

# ARCHITECTURE: 42 x 24 x 16 x 4

# weights: 3D array. weights[layer][neuron][connectsto]

random.seed(57)

weights = [
    [
        [random.random() for j in range(24)] for i in range(42)
    ], 
    [
        [random.random() for j in range(16)] for i in range(24)
    ],
    [
        [random.random() for j in range(4)] for i in range(16)
    ]
]

# biases: 2D array: biases[layer][neuron]
biases = [
    [], 
    [random.random() for j in range(24)],
    [random.random() for j in range(16)],
    [random.random() for j in range(4)]
]

def feed_forward(inputLayer):
    # inputs: [49 square for things around head] + [distance from top, bottom, left, right] + [distance to apple up] + [distance to apple left]
    hiddenlayer1 = [0 for i in range(24)] 
    hiddenlayer2 = [0 for i in range(16)]
    outputlayer = [0 for i in range(4)]
    for nextNeuronIndex in range(len(hiddenlayer1)):
        for currentNeuronIndex in range(len(inputLayer)):
            hiddenlayer1[nextNeuronIndex] += weights[0][currentNeuronIndex][nextNeuronIndex] * inputLayer[currentNeuronIndex]
        hiddenlayer1[nextNeuronIndex] += biases[1][nextNeuronIndex]
        hiddenlayer1[nextNeuronIndex] = max(0, hiddenlayer1[nextNeuronIndex]) # activation function
    
    for nextNeuronIndex in range(len(hiddenlayer2)):
        for currentNeuronIndex in range(len(hiddenlayer1)):
            hiddenlayer2[nextNeuronIndex] += weights[1][currentNeuronIndex][nextNeuronIndex] * inputLayer[currentNeuronIndex]
        hiddenlayer2[nextNeuronIndex] += biases[2][nextNeuronIndex]
        hiddenlayer2[nextNeuronIndex] = max(0, hiddenlayer2[nextNeuronIndex]) # activation function

    for nextNeuronIndex in range(len(outputlayer)):
        for currentNeuronIndex in range(len(hiddenlayer2)):
            outputlayer[nextNeuronIndex] += weights[2][currentNeuronIndex][nextNeuronIndex] * inputLayer[currentNeuronIndex]
        outputlayer[nextNeuronIndex] += biases[3][nextNeuronIndex]
        outputlayer[nextNeuronIndex] = math.tanh(outputlayer[nextNeuronIndex]) # activation function

    # normalise output layer
    sumOutputs = 0
    for output in outputlayer:
        sumOutputs += output

    for i in range(len(outputlayer)):
        outputlayer[i] = outputlayer[i] / sumOutputs
    
    return hiddenlayer1, hiddenlayer2, outputlayer

import random
import math

GAMMA = 0.9  # discount factor

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

def choose_action(output, epsilon=0.1):
    """ε-greedy policy for exploration."""
    if random.random() < epsilon:
        return random.randint(0, len(output) - 1)
    return output.index(max(output))

def backpropagate(inputLayer, target_output, learning_rate=0.01):
    """
    Backpropagation using ReLU activation and mean-squared error loss.
    target_output: list of desired output values (same length as outputlayer)
    """
    global weights, biases

    # Forward pass (reuse your feed_forward structure)
    hidden1 = [0 for _ in range(24)]
    hidden2 = [0 for _ in range(16)]
    output = [0 for _ in range(4)]

    for j in range(len(hidden1)):
        for i in range(len(inputLayer)):
            hidden1[j] += weights[0][i][j] * inputLayer[i]
        hidden1[j] += biases[1][j]
        hidden1[j] = relu(hidden1[j])

    for j in range(len(hidden2)):
        for i in range(len(hidden1)):
            hidden2[j] += weights[1][i][j] * hidden1[i]
        hidden2[j] += biases[2][j]
        hidden2[j] = relu(hidden2[j])

    for j in range(len(output)):
        for i in range(len(hidden2)):
            output[j] += weights[2][i][j] * hidden2[i]
        output[j] += biases[3][j]

    # --- Compute output layer deltas ---
    output_deltas = [0 for _ in range(len(output))]
    for j in range(len(output)):
        error = target_output[j] - output[j]
        output_deltas[j] = error  # linear output layer

    # --- Hidden layer 2 deltas ---
    hidden2_deltas = [0 for _ in range(len(hidden2))]
    for i in range(len(hidden2)):
        downstream = sum(output_deltas[j] * weights[2][i][j] for j in range(len(output)))
        hidden2_deltas[i] = downstream * relu_derivative(hidden2[i])

    # --- Hidden layer 1 deltas ---
    hidden1_deltas = [0 for _ in range(len(hidden1))]
    for i in range(len(hidden1)):
        downstream = sum(hidden2_deltas[j] * weights[1][i][j] for j in range(len(hidden2)))
        hidden1_deltas[i] = downstream * relu_derivative(hidden1[i])

    # --- Update weights and biases ---
    # Hidden2 → Output
    for i in range(len(hidden2)):
        for j in range(len(output)):
            weights[2][i][j] += learning_rate * output_deltas[j] * hidden2[i]
    for j in range(len(output)):
        biases[3][j] += learning_rate * output_deltas[j]

    # Hidden1 → Hidden2
    for i in range(len(hidden1)):
        for j in range(len(hidden2)):
            weights[1][i][j] += learning_rate * hidden2_deltas[j] * hidden1[i]
    for j in range(len(hidden2)):
        biases[2][j] += learning_rate * hidden2_deltas[j]

    # Input → Hidden1
    for i in range(len(inputLayer)):
        for j in range(len(hidden1)):
            weights[0][i][j] += learning_rate * hidden1_deltas[j] * inputLayer[i]
    for j in range(len(hidden1)):
        biases[1][j] += learning_rate * hidden1_deltas[j]


def train_Q_network(state, next_state, action, reward, done, lr=0.01):
    """One Q-learning update step using target value."""
    _, _, q_values = feed_forward(state)
    _, _, next_q_values = feed_forward(next_state)

    target_q = q_values[:]
    if done:
        target = reward
    else:
        target = reward + GAMMA * max(next_q_values)
    target_q[action] = target

    backpropagate(state, target_q, learning_rate=lr)

    

def relu(x):
    return x if x > 0 else 0

def relu_derivative(x):
    return 1 if x > 0 else 0

def backpropagate(inputLayer, target, learning_rate=0.01):
    """
    Performs one backward pass using ReLU activations and MSE loss.
    Updates global weights and biases in place.
    """

    # Forward pass — store z and a for each layer
    z1 = [0 for _ in range(24)]
    a1 = [0 for _ in range(24)]
    z2 = [0 for _ in range(16)]
    a2 = [0 for _ in range(16)]
    z3 = [0 for _ in range(4)]
    a3 = [0 for _ in range(4)]

    # --- Layer 1 (input -> 24) ---
    for j in range(24):
        total = biases[1][j]
        for i in range(len(inputLayer)):
            total += inputLayer[i] * weights[0][i][j]
        z1[j] = total
        a1[j] = relu(total)
        print(a1[j], end= " ")
    print()
    # --- Layer 2 (24 -> 16) ---
    for j in range(16):
        total = biases[2][j]
        for i in range(24):
            total += a1[i] * weights[1][i][j]
        z2[j] = total
        a2[j] = relu(total)
        print(a2[j], end= " ")
    print()

    # --- Output layer (16 -> 4) ---
    for j in range(4):
        total = biases[3][j]
        for i in range(16):
            total += a2[i] * weights[2][i][j]
        z3[j] = total
        a3[j] = relu(total)
        print(a3[j], end="")
    print()

    # === Backward pass ===

    # Output layer delta (MSE)
    delta3 = [0 for _ in range(4)]
    for j in range(4):
        error = a3[j] - target[j]
        delta3[j] = error * relu_derivative(z3[j])

    # Hidden layer 2 delta
    delta2 = [0 for _ in range(16)]
    for i in range(16):
        err_sum = 0
        for j in range(4):
            err_sum += delta3[j] * weights[2][i][j]
        delta2[i] = err_sum * relu_derivative(z2[i])

    # Hidden layer 1 delta
    delta1 = [0 for _ in range(24)]
    for i in range(24):
        err_sum = 0
        for j in range(16):
            err_sum += delta2[j] * weights[1][i][j]
        delta1[i] = err_sum * relu_derivative(z1[i])

    # === Update weights & biases ===

    # Layer 3 (16 -> 4)
    for i in range(16):
        for j in range(4):
            weights[2][i][j] -= learning_rate * delta3[j] * a2[i]
    for j in range(4):
        biases[3][j] -= learning_rate * delta3[j]

    # Layer 2 (24 -> 16)
    for i in range(24):
        for j in range(16):
            weights[1][i][j] -= learning_rate * delta2[j] * a1[i]
    for j in range(16):
        biases[2][j] -= learning_rate * delta2[j]

    # Layer 1 (input -> 24)
    for i in range(len(inputLayer)):
        for j in range(24):
            weights[0][i][j] -= learning_rate * delta1[j] * inputLayer[i]
    for j in range(24):
        biases[1][j] -= learning_rate * delta1[j]

    return a3  # output activations
