import random

def generateInputLayer(board, head, apple):
    """
    board: 2D list (grid)
    head: [row, col]
    apple: [row, col]
    returns: list of normalized inputs
    """
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

    print(len(input_layer))

    return input_layer



# ARCHITECTURE: 42 x 24 x 16 x 4

# weights: 3D array. weights[layer][neuron][connectsto]

random.seed(60)

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
    
    for nextNeuronIndex in range(len(hiddenlayer2)):
        for currentNeuronIndex in range(len(hiddenlayer1)):
            hiddenlayer2[nextNeuronIndex] += weights[1][currentNeuronIndex][nextNeuronIndex] * inputLayer[currentNeuronIndex]
        hiddenlayer2[nextNeuronIndex] += biases[2][nextNeuronIndex]

    for nextNeuronIndex in range(len(outputlayer)):
        for currentNeuronIndex in range(len(hiddenlayer2)):
            outputlayer[nextNeuronIndex] += weights[2][currentNeuronIndex][nextNeuronIndex] * inputLayer[currentNeuronIndex]
        outputlayer[nextNeuronIndex] += biases[3][nextNeuronIndex]
    
    return hiddenlayer1, hiddenlayer2, outputlayer
    

