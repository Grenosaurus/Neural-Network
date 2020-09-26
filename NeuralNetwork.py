"""
 Neural network program for single neuron. Program is done for learning about neural network 
 in general (Made: Jauaries Loyala -- 08.04.2020).
"""

# Python packets
import time

import numpy as np


# Nueral Network class
class NeuralNetwork():

    def __init__(self):

        np.random.seed(1) # Generates the same numbers every time the program runs

        """
         Modelling a single neuron, with 3 input and 1 output connection, and assignning a random weight 
         to a matrix (3 x 1 matrix). The range of value are -1 to 1 and the mean value is 0.
        """
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1


    # Program calculates the sigmoid value
    def sigmoid(self, x):

        sigmoid = 1/(1 + np.exp(-x)) # Describesa an S shaped curve

        return sigmoid


    # Program calculates the deriative of the sigmoid curve
    def sigmoidDeriative(self, x):

        sigmoid_deriative = x * (1 - x) # Gradient of the sigmoid curve

        return sigmoid_deriative


    # Goes through the errors and trials
    def train(self, inputs, outputs, iterations):

        for i in range(iterations):
            output = self.think(inputs) # Pass the training set through the nueral network
            error = outputs - output # Calculates the Error
            adjustment = np.dot(inputs.T, error * self.sigmoidDeriative(output)) # Avoiding changes in the weights when the input value is 0

            self.synaptic_weights += adjustment # Adjust the weight


    def think(self, inputs):

        # Passes the input through the Neural Network
        think = self.sigmoid(np.dot(inputs.astype(float), self.synaptic_weights))

        return think




# Main function for calling Neural Network
def main():

    NN = NeuralNetwork() # Calling Neural network for single neuron

    print('Random starting synaptic weights: \n%s' % NN.synaptic_weights)

    # Values used
    matrix = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]] # Matrix [4 * 3]
    vector = [[0, 1, 1, 0]] # Matrix [4 * 1]
    iteration_number = 10000 # Iteration
    testing_input = [1, 0, 0] # Testing input

    # Training values
    input_set = np.array(matrix)
    output_set = np.array(vector).T # Vector transposed

    # Using the train function in Neural Network class
    NN.train(input_set, output_set, iteration_number)

    print('Synaptic weights after training: \n%s' % NN.synaptic_weights)
    print('\nTest INPUT: [1, 0, 0] --> OUTPUT: %s\n' % NN.think(np.array(testing_input)))
    print('Give own inputs!')

    # Reading values from terminal
    input_1, input_2, input_3 = str(input('INPUT_1: ')), str(input('INPUT_2: ')), str(input('INPUT_3: '))

    # Generating the values into a [3 * 1] matrix
    matrix_new = [int(input_1), int(input_2), int(input_3)]

    print('\nINPUT: [%s, %s, %s] --> OUTPUT: %s' % (input_1, input_2, input_3, NN.think(np.array(matrix_new))))


# Do not touch!
if __name__ == "__main__":

    # Time of main function
    start_time = time.time() # Start time of the main function
    main()
    end_time = time.time() # End time of the main function

    # Delta time
    Delta_time_seconds = end_time - start_time # Difference between start and end time (defined in seconds)
    Delta_time_minutes = Delta_time_seconds/60 # Difference between start and end time (defined in minutes) | 1 min = 60 s

    # Round value of delta time
    Delta_time_seconds_round = round(Delta_time_seconds, 2)
    Delta_time_minutes_round = round(Delta_time_minutes, 2)

    # Prints the time took for the program to run
    print("\nProgram took %s seconds (~ %s minutes)." % (Delta_time_seconds_round, Delta_time_minutes_round))

