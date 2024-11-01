import math

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
# Dot product function
def dot_product(inputs, weights):
    return sum(i * w for i, w in zip(inputs, weights))


# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, weights_input_hidden=None, weights_hidden_output=None):
        self.learning_rate = learning_rate


        # Initialize weights for input to hidden layer from user input
        if weights_input_hidden:
            self.weights_input_hidden = weights_input_hidden
        else:
            self.weights_input_hidden = [[0 for _ in range(hidden_size)] for _ in range(input_size + 1)]  # Placeholder if not provided


        # Initialize weights for hidden to output layer from user input
        if weights_hidden_output:
            self.weights_hidden_output = weights_hidden_output
        else:
            self.weights_hidden_output = [0 for _ in range(hidden_size + 1)]  # Placeholder if not provided


    def forward(self, X):
        # Add bias to input
        inputs_with_bias = [1] + X
        
        # Hidden layer activations
        self.hidden_layer = []
        for weights in zip(*self.weights_input_hidden):
            hidden_input = dot_product(inputs_with_bias, weights)
            self.hidden_layer.append(sigmoid(hidden_input))
        
        # Add bias to hidden layer
        hidden_with_bias = [1] + self.hidden_layer
        
        # Output layer activation
        self.output = sigmoid(dot_product(hidden_with_bias, self.weights_hidden_output))
        return self.output


    def backward(self, X, y):
        # Calculate output error
        output_error = y - self.output
        
        # Derivative of output
        output_delta = output_error * sigmoid_derivative(self.output)


        # Backpropagate error to hidden layer
        hidden_with_bias = [1] + self.hidden_layer
        hidden_errors = []
        for h, w in zip(hidden_with_bias, self.weights_hidden_output):
            hidden_errors.append(output_delta * w * sigmoid_derivative(h))
        
        # Update weights from hidden to output layer
        for i in range(len(self.weights_hidden_output)):
            self.weights_hidden_output[i] += self.learning_rate * output_delta * hidden_with_bias[i]


        # Update weights from input to hidden layer
        inputs_with_bias = [1] + X
        for i in range(len(self.weights_input_hidden)):
            for j in range(len(self.weights_input_hidden[0])):
                self.weights_input_hidden[i][j] += self.learning_rate * hidden_errors[j+1] * inputs_with_bias[i]


    def train(self, X, y, epochs):
        for epoch in range(epochs):
            for inputs, target in zip(X, y):
                self.forward(inputs)
                self.backward(inputs, target)


    def predict(self, X):
        output = self.forward(X)
        # Convert output to binary (0 or 1) using a threshold of 0.5
        return 1 if output >= 0.5 else 0


# Function to take weight input from the user
def get_weights(input_size, hidden_size, output_size):
    print('\n*** Weight Initialization ***')
    print("\nPlease enter the weights from the input layer to the hidden layer:")
    weights_input_hidden = []
    for i in range(input_size + 1):  # +1 for bias
        row = []
        for j in range(hidden_size):
            weight = float(input(f"Weight from input node {i} to hidden node {j}: "))
            row.append(weight)
        weights_input_hidden.append(row)


    print("\nPlease enter the weights from the hidden layer to the output layer:")
    weights_hidden_output = []
    for i in range(hidden_size + 1):  # +1 for bias
        weight = float(input(f"Weight from hidden node {i} to output node: "))
        weights_hidden_output.append(weight)


    return weights_input_hidden, weights_hidden_output


# User input function for logical gates and learning parameters
def get_user_input():
    print("\n=== Logical Gate Selection ===")
    gate = input('''Select the logical gate to train the network:
    1. AND Gate 
    2. OR Gate 
    Your choice (1/2): ''')


    if gate == "1":
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [0, 0, 0, 1]
        print("\n*** AND Gate Selected ***")
    elif gate == "2":
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [0, 1, 1, 1]
        print("\n*** OR Gate Selected ***")
    else:
        print("Invalid gate chosen. Exiting...")
        exit()


    learning_rate = float(input("\nEnter the learning rate (e.g., 0.1): "))
    epochs = int(input("Enter the number of training epochs: "))


    return X, y, learning_rate, epochs


# Main program
print("\n** Neural Network Backpropagation Model **")
if __name__:


    # Get user input for training
    X, y, learning_rate, epochs = get_user_input()


    # Get user-provided weights
    input_size = 2  # For logical gates with two inputs
    hidden_size = 2  # We can choose 2 hidden nodes for simplicity
    output_size = 1  # Binary output


    weights_input_hidden, weights_hidden_output = get_weights(input_size, hidden_size, output_size)


    # Initialize neural network with user-provided weights
    nn = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, learning_rate=learning_rate,
                       weights_input_hidden=weights_input_hidden, weights_hidden_output=weights_hidden_output)


    # Train the neural network (no per-epoch output)
    nn.train(X, y, epochs)


    print('\n** Training Complete! **')


    # Testing the trained network
    print("\n** Testing the Neural Network **")
    while True:
        test_input = input("\nEnter two test inputs (e.g., 1 0) or type 'exit' to quit: ")
        if test_input.lower() == 'exit':
            print("\n** Exiting the Program. Goodbye! **")
            break
        # Parse input and predict output
        test_input = [int(i) for i in test_input.split()]
        output = nn.predict(test_input)
        print(f"==> The neural network predicts the output as: {output}")
