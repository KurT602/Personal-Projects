import numpy as np
import json, time
from datetime import datetime

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Glorot / Xavier init (zero-mean, scaled variance)
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/(input_size + hidden_size))
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/(hidden_size + output_size))
        self.bias_output = np.zeros((1, output_size))
        
        self.epochs_completed = 0
        self.loss_history = []
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, input_data):
        self.input = input_data  # (N, input_size)
        self.hidden_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden  # (N, hidden)
        self.hidden_output = sigmoid(self.hidden_input)  # (N, hidden)
        self.output_logits = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output  # (N, out)
        self.output = softmax(self.output_logits)  # (N, out)
        return self.output

    def backward(self, input_data, output_data, learning_rate):
        N = input_data.shape[0]
        # For softmax + categorical cross-entropy, gradient of logits is (p - y)/N
        dZ2 = (self.output - output_data) / N  # (N, out)
        dW2 = self.hidden_output.T.dot(dZ2)    # (hidden, out)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Backprop into hidden layer
        hidden_error = dZ2.dot(self.weights_hidden_output.T)  # (N, hidden)
        dZ1 = hidden_error * sigmoid_derivative(self.hidden_output)  # (N, hidden)
        dW1 = input_data.T.dot(dZ1)  # (in, hidden)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update params (gradient descent)
        self.weights_hidden_output -= learning_rate * dW2
        self.bias_output            -= learning_rate * db2
        self.weights_input_hidden  -= learning_rate * dW1
        self.bias_hidden           -= learning_rate * db1

    def train(self, input_data, output_data, epochs, learning_rate, epolog_freq=1000, save_path="model.npz",config_path="config.json"):
        for epoch in range(self.epochs_completed,epochs):
            probs = self.forward(input_data)
            # categorical cross-entropy loss
            loss = -np.mean(np.sum(output_data * np.log(probs + 1e-12), axis=1))
            self.backward(input_data, output_data, learning_rate)
            self.loss_history.append(loss)
            if epoch % epolog_freq == 0:
                print(f"[{datetime.fromtimestamp(time.time())}]: Epoch {epoch}, Loss: {loss:.6f}")
                self.save_model(save_path,config_path)
            self.epochs_completed += 1

    def predict(self, input_data):
        probs = self.forward(input_data)
        return probs   # returns class index

    def save_model(self, save_path, config_path):
        np.savez(save_path,
                 weights_input_hidden=self.weights_input_hidden,
                 bias_hidden=self.bias_hidden,
                 weights_hidden_output=self.weights_hidden_output,
                 bias_output=self.bias_output,
                 epochs_completed=self.epochs_completed,
                 loss_history=np.array(self.loss_history))  # Convert to numpy array for saving

        config = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
        }
        with open(config_path, "w") as f:
            json.dump(config, f)
        print("saved model")
    
    def load_model(self, save_path, config_path):
        """Loads the model's weights, biases, and training state."""
        try:
            with np.load(save_path) as data:
                self.weights_input_hidden = data['weights_input_hidden']
                self.bias_hidden = data['bias_hidden']
                self.weights_hidden_output = data['weights_hidden_output']
                self.bias_output = data['bias_output']
                self.epochs_completed = data['epochs_completed'].item()  # Load epochs
                self.loss_history = data['loss_history'].tolist()  # Load loss history
            with open(config_path, "r") as f:
                config = json.load(f)
                self.input_size = config["input_size"]
                self.hidden_size = config["hidden_size"]
                self.output_size = config["output_size"]
                # No need to set learning rate here, as it's used in train()
            print("Model loaded successfully.")

        except FileNotFoundError:
            print("No saved model found. Starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}")