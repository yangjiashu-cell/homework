import numpy as np

class Trainer:
    def __init__(self, model, learning_rate=0.001, batch_size=64, epochs=50, lr_decay=0.1, decay_epochs=20):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_decay = lr_decay
        self.decay_epochs = decay_epochs
        self.train_losses = []  # To store training losses
        self.val_accs = []      # To store validation accuracies

    def train(self, X_train, y_train, X_val, y_val):
        n_samples = X_train.shape[0]
        best_val_acc = 0
        lr = self.learning_rate

        for epoch in range(self.epochs):
            # Learning rate decay
            if epoch > 0 and epoch % self.decay_epochs == 0:
                lr *= self.lr_decay

            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices].T  # Adjust shape if necessary

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[:, i:i + self.batch_size]

                # Forward pass
                output = self.model.forward(X_batch)
                # Compute loss
                loss = self.model.compute_loss(y_batch)
                # Backward pass
                dW1, db1, dW2, db2 = self.model.backward(y_batch)
                # Update parameters
                self.model.W1 -= lr * dW1
                self.model.b1 -= lr * db1
                self.model.W2 -= lr * dW2
                self.model.b2 -= lr * db2

            # Evaluate on validation set
            val_acc = self.evaluate(X_val, y_val)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")

            # Store metrics
            self.train_losses.append(loss)
            self.val_accs.append(val_acc)

            # Save best model (optional)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.npy')

    def evaluate(self, X, y):
        # Placeholder: Replace with your model's evaluation logic
        output = self.model.forward(X)
        predictions = np.argmax(output, axis=0)
        labels = np.argmax(y, axis=1)  # Adjust based on y's shape
        return np.mean(predictions == labels)

    def save_model(self, filename):
        np.save(filename, {'W1': self.model.W1, 'b1': self.model.b1, 
                          'W2': self.model.W2, 'b2': self.model.b2})
    def load_model(self, filename):
        params = np.load(filename, allow_pickle=True).item()
        self.model.W1 = params['W1']
        self.model.b1 = params['b1']
        self.model.W2 = params['W2']
        self.model.b2 = params['b2']