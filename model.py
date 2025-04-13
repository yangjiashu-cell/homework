import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exps / np.sum(exps, axis=0, keepdims=True)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', reg_lambda=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.reg_lambda = reg_lambda
        
        # 初始化参数
        if activation == 'relu':
            self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size)
        else:
            self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))
    
    def forward(self, X):
        self.X = X.T  # (input_size, n_samples)
        self.Z1 = np.dot(self.W1, self.X) + self.b1
        if self.activation == 'relu':
            self.A1 = relu(self.Z1)
        elif self.activation == 'sigmoid':
            self.A1 = sigmoid(self.Z1)
        else:
            raise ValueError("Unsupported activation")
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2
    
    def backward(self, y):
        n_samples = y.shape[1]
        dZ2 = self.A2 - y
        dW2 = np.dot(dZ2, self.A1.T) / n_samples + self.reg_lambda * self.W2
        db2 = np.sum(dZ2, axis=1, keepdims=True) / n_samples
        dA1 = np.dot(self.W2.T, dZ2)
        if self.activation == 'relu':
            dZ1 = dA1 * relu_derivative(self.Z1)
        elif self.activation == 'sigmoid':
            dZ1 = dA1 * sigmoid_derivative(self.Z1)
        dW1 = np.dot(dZ1, self.X.T) / n_samples + self.reg_lambda * self.W1
        db1 = np.sum(dZ1, axis=1, keepdims=True) / n_samples
        return dW1, db1, dW2, db2
    
    def compute_loss(self, y):
        n_samples = y.shape[1]
        logp = -np.log(self.A2 * y + 1e-15)  # 添加小值避免 log(0)
        loss = np.sum(logp) / n_samples
        reg = (self.reg_lambda / 2) * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return loss + reg