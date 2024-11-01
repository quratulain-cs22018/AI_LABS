# Step function (activation function)
def step_function(x):
    return 1 if x >= 0 else 0


# Dot product function
def dot_product(inputs, weights):
    return sum(i * w for i, w in zip(inputs, weights))


# Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        # Initialize weights (including bias)
        self.weights = [0.0] * (input_size + 1)  # +1 for the bias term
        self.learning_rate = learning_rate
        self.epochs = epochs


    def train(self, X, y):
        for epoch in range(self.epochs):
            print(f"\n** Training Epoch {epoch + 1} **")
            for inputs, label in zip(X, y):
                # Add bias input (1) to each input sample
                inputs_with_bias = [1] + inputs


                # Calculate linear combination
                linear_output = dot_product(inputs_with_bias, self.weights)
                prediction = step_function(linear_output)
                
                # Calculate error
                error = label - prediction


                # Update weights and bias
                self.weights = [w + self.learning_rate * error * inp for w, inp in zip(self.weights, inputs_with_bias)]


                print(f"-> Weights after update: {self.weights}")


    def predict(self, X):
        # Add bias input (1) to the test input
        inputs_with_bias = [1] + X
        linear_output = dot_product(inputs_with_bias, self.weights)
        return step_function(linear_output)


# Function to get user input for gate selection and learning parameters
def get_user_input():
    print("\n=== Perceptron Training: Logical Gate Selection ===")
    gate = input('''Select the logical gate to train the perceptron:
    1. AND Gate 
    2. OR Gate 
    Your choice (1/2): ''')


    if gate == "1":
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [0, 0, 0, 1]
        print("\n*** You have selected AND Gate ***")
    elif gate == "2":
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [0, 1, 1, 1]
        print("\n*** You have selected OR Gate ***")
    else:
        print("Invalid choice! Please restart the program.")
        exit()


    learning_rate = float(input("\nPlease enter the learning rate (e.g., 0.1): "))
    epochs = int(input("Enter the number of training epochs: "))


    return X, y, learning_rate, epochs


# Main program
print("\n** Welcome to the Perceptron Training Program **")
if __name__ == "__main__":


    # Get user input for training
    X, y, learning_rate, epochs = get_user_input()


    # Initialize perceptron with 2 inputs (for logical gates)
    perceptron = Perceptron(input_size=2, learning_rate=learning_rate, epochs=epochs)


    # Train the perceptron
    perceptron.train(X, y)


    print('\n** Training Complete! **')


    # Testing the trained perceptron
    print("\n** Perceptron Testing Phase **")
    while True:
        test_input = input("\nPlease enter two inputs (e.g., 1 0) to test, or type 'exit' to quit: ")
        if test_input.lower() == 'exit':
            print("\n** Exiting the Program. Goodbye! **")
            break
        # Parse input and predict output
        test_input = [int(i) for i in test_input.split()]
        output = perceptron.predict(test_input)
        print(f"==> The perceptron predicts the output as: {output}")
