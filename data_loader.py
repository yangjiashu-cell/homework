import numpy as np
import pickle

def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'], dict[b'labels']

def load_cifar10_data(data_dir):
    X_train, y_train = [], []
    for i in range(1, 6):
        X, y = load_cifar10_batch(f'{data_dir}/data_batch_{i}')
        X_train.append(X)
        y_train.append(y)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test, y_test = load_cifar10_batch(f'{data_dir}/test_batch')
    
    # 归一化
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # One-hot 编码
    num_classes = 10
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]
    
    return X_train, y_train_onehot, X_test, y_test_onehot

# 使用示例
X_train, y_train, X_test, y_test = load_cifar10_data('cifar-10-batches-py')