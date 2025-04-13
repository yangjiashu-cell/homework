import numpy as np

class Trainer:
    def __init__(self, model, learning_rate=0.001, batch_size=64, epochs=50, lr_decay=0.1, decay_epochs=20):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_decay = lr_decay
        self.decay_epochs = decay_epochs
    
    def train(self, X_train, y_train, X_val, y_val):
        n_samples = X_train.shape[0]
        best_val_acc = 0
        lr = self.learning_rate
        
        for epoch in range(self.epochs):
            # 学习率下降
            if epoch > 0 and epoch % self.decay_epochs == 0:
                lr *= self.lr_decay
            
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices].T  # (output_size, n_samples)
            
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[:, i:i + self.batch_size]
                
                # 前向传播
                output = self.model.forward(X_batch)
                # 计算损失
                loss = self.model.compute_loss(y_batch)
                # 反向传播
                dW1, db1, dW2, db2 = self.model.backward(y_batch)
                # 更新参数
                self.model.W1 -= lr * dW1
                self.model.b1 -= lr * db1
                self.model.W2 -= lr * dW2
                self.model.b2 -= lr * db2
            
            # 验证
            val_acc = self.evaluate(X_val, y_val)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 保存最优模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.npy')
    
    def evaluate(self, X, y):
        output = self.model.forward(X)
        predictions = np.argmax(output, axis=0)
        labels = np.argmax(y, axis=1)
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