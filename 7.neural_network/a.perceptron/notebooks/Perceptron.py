import numpy as np
from sklearn.metrics import accuracy_score
class Perceptron:
    def __init__(self):
        self.weights = None

    def weighting(self, input):
        weighted_sum = np.dot(input, self.weights)
        return weighted_sum # To-Do

    def activation(self, weighted_input):
        if weighted_input >= 0:
            return 1
        else:
            return -1 # To-Do

    def predict(self, inputs):
        # adding a 1 to the first position of each input (adding the bias term)
        new_inputs = np.insert(inputs, 0, 1, axis=1)# To-Do
        # a list of final prediction for each test sample
        predictions = []
        for input in new_inputs:
            weighted_input = self.weighting(input)# To-Do
            prediction = self.activation(weighted_input)# To-Do
            predictions.append(prediction)
        # converting the list to a numpy array
        predictions = np.array(predictions)
        return predictions

    def fit(self, inputs, outputs, learning_rate=0.1, epochs=64):
        # adding a 1 to the first position of each input (adding the bias term)
        new_inputs = np.insert(inputs, 0, 1, axis=1) # To-Do
        # initializing the weights
        self.weights = np.random.rand(new_inputs.shape[1])
        # training loop
        for epoch in range(epochs):
            for sample, target in zip(new_inputs, outputs):
                weighted_input = self.weighting(sample)# To-Do (using self.weighting on sample)
                pred = self.activation(weighted_input)
                diff =  target - pred# To-Do (based on target and self.activation)
                self.weights += diff * learning_rate * sample # To-Do (based on self.weights, learning_rate, diff and sample)

