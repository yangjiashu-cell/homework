# homework
以下是根据用户提供的文件和要求撰写的中文 README 文件，内容清晰指明如何进行训练和测试，并符合作业要求。文件中包含项目简介、文件结构、环境要求、数据准备、模型训练、模型测试、结果分析、模型权重获取方式以及注意事项。

---

# 项目 README

## 项目简介

本项目旨在从零开始构建一个三层神经网络分类器，并在 CIFAR-10 数据集上实现图像分类任务。项目严格遵循作业要求，自主实现了反向传播算法，未使用任何支持自动微分的深度学习框架（如 PyTorch 或 TensorFlow），主要依赖 NumPy 进行数值计算。代码支持自定义隐藏层大小、激活函数类型，并实现了 SGD 优化器、学习率衰减、交叉熵损失和 L2 正则化等功能。通过网格搜索超参数（隐藏层大小、学习率、正则化强度），找到最佳模型配置，并在测试集上评估分类准确率。

## 文件结构

- **`data_loader.py`**: 负责加载和预处理 CIFAR-10 数据集，包括像素归一化和标签的 one-hot 编码。
- **`model.py`**: 定义三层神经网络模型，支持 ReLU 和 Sigmoid 激活函数，包含前向传播和反向传播实现。
- **`trainer.py`**: 实现训练逻辑，包括 SGD 优化器、学习率衰减、交叉熵损失计算和 L2 正则化，支持保存验证集上表现最佳的模型权重。
- **`neural network.ipynb`**: 使用 `epoch=50` 进行网格搜索，探索不同超参数组合下的模型性能。
- **`neural network final.ipynb`**: 使用 `epoch=100` 进行网格搜索，进一步优化超参数并可视化第一层权重。
- **`results.csv`**: 记录 `neural network.ipynb` 中 `epoch=50` 的网格搜索结果，包括训练过程中的 loss 和验证集准确率。
- **`results final.csv`**: 记录 `neural network final.ipynb` 中 `epoch=10` 的网格搜索结果，包含最佳超参数配置。
- **`best_model.npy`**: 训练完成后的最优模型权重文件。

## 环境要求

- **Python 版本**: 3.6 或以上
- **依赖库**:
  - NumPy（用于数值计算）
  - Matplotlib（用于可视化）


## 数据准备

1. 下载 CIFAR-10 数据集（Python 版本）：
   - 从官方网站 [CIFAR-10 数据集](https://www.cs.toronto.edu/~kriz/cifar.html) 下载 `cifar-10-python.tar.gz`。
   - 解压后，将数据文件夹 `cifar-10-batches-py` 放置于项目根目录下。
2. 数据加载和预处理：
   - `data_loader.py` 会自动加载数据集并进行预处理（像素值归一化至 [0, 1]，标签转换为 one-hot 编码）。

## 模型训练

### 使用 Jupyter Notebook 进行训练

本项目提供两个 notebook 文件用于训练和超参数搜索：

1. **`neural network final.ipynb`**:
   - 使用 `epoch=10` 进行网格搜索，测试隐藏层大小（128、256、512）、学习率（0.01、0.1、0.0001）和正则化强度（0.0001、0.001、0.01）。
   - **运行方法**：
     - 打开 `neural network final.ipynb`。
     - 依次运行所有单元格，完成数据加载、模型训练和超参数搜索。
     - 训练结果（loss 和验证集准确率）会输出到 `results.csv`。
   - **特点**：训练时间较短，适合初步探索超参数影响。

2. **`neural network.ipynb`**:
   - 使用 `epoch=50` 进行网格搜索，进一步优化超参数组合。
   - **运行方法**：
     - 打开 `neural network.ipynb`。
     - 依次运行所有单元格，完成数据加载、模型训练和超参数搜索。
     - 训练结果会输出到控制台，并保存最佳模型权重到 `best_model.npy`。
   - **特点**：训练时间较长，可能获得更高性能，同时包含第一层权重可视化。

**注意**：
- 网格搜索会自动保存验证集上表现最佳的模型权重到 `best_model.npy`。
- 训练过程中会打印每个 epoch 的 loss 和验证集准确率。



## 模型测试

1. **加载模型权重**：
   - 确保 `best_model.npy` 已放置在项目根目录下。
2. **运行测试**：
   - 在 `trainer.py` 中，`Trainer` 类提供了 `evaluate` 方法用于测试模型性能。
   

## 结果分析

- **`results.csv`**：
  - 记录 `neural network.ipynb` 中 `epoch=50` 的网格搜索结果。
  - 包含不同超参数组合下的训练 loss 和验证集准确率。
  - 可用于分析超参数对模型性能的影响。
- **`results final.csv`**：
  - 记录 `neural network final.ipynb` 中 `epoch=10` 的网格搜索结果。
  - 最佳超参数为 `hidden_size=512, lr=0.01, reg=0.0001`
- **可视化**：
  - `neural network final.ipynb` 中包含训练过程的 loss 和准确率曲线（需运行 notebook 查看）。
  - 第一层权重（W1）的可视化展示了模型学到的特征模式。

## 模型权重

- 训练好的最优模型权重文件 `best_model.npy` 已上传至百度云：
  - **下载地址**: [百度云链接](https://pan.baidu.com/s/your-link)（请替换为实际链接）
  - 下载后，将文件放置在项目根目录下即可用于测试。

## 注意事项

- **数据路径**：运行代码前，确保 `cifar-10-batches-py` 文件夹正确放置在项目根目录下。
- **训练时间**：网格搜索（尤其是 `epoch=100`）可能需要较长时间，建议在高性能机器上运行。
- **硬件加速**：本项目未实现 GPU 加速，若有 GPU，可自行修改代码以提升训练效率。
- **依赖检查**：运行前确认已安装 NumPy 和 Matplotlib，避免环境问题。

---
