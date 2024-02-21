import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
import h5py
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
import torch.nn.init as init

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#使用全体数据集“即带N的数据”时，使用ImprovedCNN，读取数据时每一个先归一化，再读取数据完成拼接后整体归一化，缺一不可。每一批次不可再归一化，即可得到收敛，训练集准确率接近1，可用

# Define your CNN architecture
class CNN(nn.Module):
    def __init__(self, input_channels, sequence_length, output_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(16 * (sequence_length // 2), output_size)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        return out

class ModifiedCNN(nn.Module):
    def __init__(self, input_channels, sequence_length, output_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * (sequence_length // 4), 256)  # Adjust this based on your sequence_length
        self.fc2 = nn.Linear(256, output_size)
        self.dropout = nn.Dropout(0.2)

        # Initialize the convolutional layers using He initialization
        init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out

class ImprovedCNN(nn.Module):
    def __init__(self, input_channels, sequence_length, output_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)  # 添加批次标准化层
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)  # 添加批次标准化层
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)  # 添加批次标准化层
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128 * (sequence_length // 8), 256)
        self.fc2 = nn.Linear(256, output_size)
        self.dropout = nn.Dropout(0.5)
        
        # 在模型的初始化中使用He初始化
        init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # 在卷积层后应用批次标准化
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)  # 在卷积层后应用批次标准化
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)  # 在卷积层后应用批次标准化
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out



# 使用您的数据维度初始化模型
sequence_length = 255
input_channels = 1
output_size = 3
learning_rate = 0.001
batch_size = 16
num_epochs = 500

# 创建一个字典来跟踪每个类别的真正例、假正例和假负例数量
class_counts = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}  # Assuming you have 3 classes (0, 1, 2)

# 定义5折交叉验证
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Prepare your data (x, y)
def extract_data_from_excel(folder_path, num_files=500, num_data_points=25):
    data_array = []
    file_names = os.listdir(folder_path)

    # Randomly select num_files files from the list
    selected_file_names = random.sample(file_names, num_files)

    for file_name in selected_file_names:
        file_path = os.path.join(folder_path, file_name)

        # Read the Excel file using pandas
        # df = pd.read_excel(file_path)
        df = pd.read_excel(file_path, engine='openpyxl')  # 或者 engine='openpyxl'


        # Extract the second column (assuming it is the column with index 1)
        data = df.iloc[:, 1].values
        
        mean = np.mean(data, axis=0,)
        std = np.std(data, axis=0,)
        if std < 1e-4:
            std = 1e-4
        data = (data - mean) / std
        data = data + 1 

        # Append the data to the data_array
        data_array.append(data)
    data_array = pd.DataFrame(data_array).to_numpy()

    return data_array

def extract_data_from_excel_50(folder_path, num_files=50, num_data_points=25):
    data_array = []
    file_names = os.listdir(folder_path)

    # Randomly select num_files files from the list
    selected_file_names = random.sample(file_names, num_files)

    for file_name in selected_file_names:
        file_path = os.path.join(folder_path, file_name)

        # Read the Excel file using pandas
        df = pd.read_excel(file_path)

        # Extract the second column (assuming it is the column with index 1)
        data = df.iloc[:, 1].values

        # Append the data to the data_array
        data_array.append(data)
    data_array = pd.DataFrame(data_array).to_numpy()

    return data_array

folder_path = r'sheep\D'
down_data = extract_data_from_excel(folder_path)
# 这个D、S、W的效果咋比加了N的效果（DN、SN、WN）还要好呢

# folder_path = r'sheep\run'
# run_data = extract_data_from_excel_50(folder_path)

folder_path = r'sheep\S'
stand_data = extract_data_from_excel(folder_path)

folder_path = r'sheep\W'
walk_data = extract_data_from_excel(folder_path)

all_data = np.concatenate((down_data, stand_data, walk_data), axis=0)
# x = all_data[:, np.newaxis, :, np.newaxis]
x = all_data[:, np.newaxis, :]
down_labels = np.zeros(down_data.shape[0])
stand_labels = np.ones(stand_data.shape[0])
walk_labels = 2 * np.ones(walk_data.shape[0])
labels = np.concatenate((down_labels, stand_labels, walk_labels))
print("数据x shape:", x.shape)
print("标签y shape:", labels)

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long) 

# 计算均值和标准差
mean = x.mean(dim=(0, 1))
std = x.std(dim=(0, 1))

x = (x - mean) / std
print(x.shape)
print(y.shape)

x = x.to('cuda')
print(x.shape)

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize your CNN model and optimizer
model = ModifiedCNN(input_channels, sequence_length, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

val_dataset = TensorDataset(torch.Tensor(x_val), torch.LongTensor(y_val))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")
model.to(device)
# model.to('cpu')

# 创建空的列表来存储每个折叠的性能指标
fold_train_losses = []
fold_train_accuracies = []
fold_val_accuracies = []

#存储数据
data_valAcc = np.random.rand(5,num_epochs)
data_loss = np.random.rand(5,num_epochs)
data_traAcc = np.random.rand(5,num_epochs)

best_val_accuracy = 0.0
best_model_state_dict = None

# Perform 5-fold cross-validation
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y)):
    print(f"Fold {fold_idx + 1}/{num_folds}:")
    
    torch.manual_seed(fold_idx)  # 使用fold索引作为随机种子

    # Split data into training and validation sets for this fold
    x_train_fold = x[train_idx]
    y_train_fold = y[train_idx]
    x_val_fold = x[val_idx]
    y_val_fold = y[val_idx]

    # Create new model and optimizer for each fold
    model = ModifiedCNN(input_channels, sequence_length, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    # Initialize lists to store metrics for this fold
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop for this fold
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        # DataLoader for this fold
        train_dataset_fold = TensorDataset(torch.Tensor(x_train_fold), torch.LongTensor(y_train_fold))
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)

        # Initialize val_accuracies with an empty list if it's the first epoch, else use the existing list
        val_accuracies = [] if epoch == 0 else val_accuracies

        for inputs, labels in train_loader_fold:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # mean = inputs.mean(dim=(0, 1))
            # std = inputs.std(dim=(0, 1))
            # inputs = (inputs - mean) / std
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data.to('cuda'), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        train_losses.append(loss.item())
        train_accuracies.append(accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.2f}')
        data_loss[fold_idx, epoch] = loss
        data_traAcc[fold_idx, epoch] = accuracy

        # Validation for this fold
        model.eval()
        all_predictions = []
        with torch.no_grad():
            correct = 0
            total = 0
            # DataLoader for this fold
            val_dataset_fold = TensorDataset(torch.Tensor(x_val_fold), torch.LongTensor(y_val_fold))
            val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)

            for inputs, labels in val_loader_fold:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # mean = inputs.mean(dim=(0, 1))
                # td = inputs.std(dim=(0, 1))
                # inputs = (inputs - mean) / std
                optimizer.zero_grad()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data.to('cuda'), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            val_accuracies.append(accuracy)
            print(f'Validation Accuracy: {accuracy:.2f}')
        data_valAcc[fold_idx, epoch] = accuracy
        
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            best_acc = accuracy
            best_model_state_dict = model.state_dict()

        # Store metrics for this fold
        fold_train_losses.append(train_losses)
        fold_train_accuracies.append(train_accuracies)
        fold_val_accuracies.append(val_accuracies)
    
# print(fold_val_accuracies.shape)
# print(fold_val_accuracies)
 
# Save the model with the best validation accuracy
torch.save(best_model_state_dict, 'CNN-3Class-batchsize=16-dropout=0.2.pt')

# Calculate and report the average performance metrics across all folds
average_train_loss = np.mean(fold_train_losses, axis=0)
average_train_accuracy = np.mean(fold_train_accuracies, axis=0)
average_val_accuracy = np.mean(fold_val_accuracies, axis=0)

print(f'Average Training Loss: {average_train_loss[-1]:.4f}')
print(f'Average Training Accuracy: {average_train_accuracy[-1]:.2f}')
print(f'Average Validation Accuracy: {average_val_accuracy[-1]:.2f}')
print(f'best Accuracy:{accuracy:.2f}')

all_train_losses = []
all_val_accuracies = []

# for fold_idx in range(num_folds):
#     plt.figure(figsize=(12, 4))
    
#     # Plot training loss curve and save training loss data
#     plt.subplot(1, 2, 1)
#     train_loss_curve = fold_train_losses[fold_idx]
#     all_train_losses.append(train_loss_curve)
#     plt.plot(train_loss_curve, label=f'Fold {fold_idx+1}')
#     plt.xlabel('Epoch')
#     plt.ylabel('Training Loss')
#     plt.title(f'Fold {fold_idx+1} - Training Loss')
#     plt.legend()
    
#     # Plot validation accuracy curve and save validation accuracy data
#     plt.subplot(1, 2, 2)
#     val_accuracy_curve = fold_val_accuracies[fold_idx]
#     all_val_accuracies.append(val_accuracy_curve)
    
#     data = np.array(val_accuracy_curve)
#     std_err = np.std(data) / np.sqrt(data.shape[0])  # 计算标准误差
#     # 创建 x 轴的值（假设有100个数据点）
#     x = np.arange(num_epochs)

#     # # 绘制曲线图
#     # plt.figure(figsize=(10, 6))
#     # plt.plot(x, data, label='Curve')
#     # plt.fill_between(x, data - std_err, data + std_err, alpha=0.2, label='Standard Error')

#     # # 设置图例和标签
#     # plt.legend()
#     # plt.xlabel('X-axis')
#     # plt.ylabel('Y-axis')
#     # plt.title('Curve with Standard Error')

#     # # 显示图形
#     # plt.grid(True)
#     # plt.show()

# print(data_valAcc.shape)
# df = pd.DataFrame(data_valAcc)
# # 保存DataFrame到Excel文件
# excel_writer = pd.ExcelWriter(r'sheep\CNNvalAcc.xlsx', engine='xlsxwriter')
# df.to_excel(excel_writer, sheet_name='Sheet1', index=False)
# # 关闭Excel写入器
# excel_writer.save()

# print(data_loss.shape)
# df = pd.DataFrame(data_loss)
# # 保存DataFrame到Excel文件
# excel_writer = pd.ExcelWriter(r'sheep\CNNloss.xlsx', engine='xlsxwriter')
# df.to_excel(excel_writer, sheet_name='Sheet1', index=False)
# # 关闭Excel写入器
# excel_writer.save()

# print(data_traAcc.shape)
# df = pd.DataFrame(data_traAcc)
# # 保存DataFrame到Excel文件
# excel_writer = pd.ExcelWriter(r'sheep\CNNtraAcc.xlsx', engine='xlsxwriter')
# df.to_excel(excel_writer, sheet_name='Sheet1', index=False)
# # 关闭Excel写入器
# excel_writer.save()

# 创建一个（5，100）的随机数组
# data_array = np.random.rand(5, num_epochs)

data = data_valAcc
# 创建一个图形对象
mean_values = np.mean(data, axis=0)  # 计算每列的均值，即每个数据点的均值
std_err = np.std(data, axis=0) / np.sqrt(data.shape[0])  # 计算每列的标准误差

# 创建 x 轴的值（假设有100个数据点）
x = np.arange(num_epochs)

# 绘制曲线图
plt.figure(figsize=(10, 6))
for i in range(data.shape[0]):
    plt.plot(x, data[i], label=f'Curve {i + 1}')

# 绘制标准误差区域
plt.fill_between(x, mean_values - std_err, mean_values + std_err, alpha=0.2, label='Standard Error')

# 设置图例和标签
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Curves with Standard Error')

# 显示图形
plt.grid(True)
plt.show()

# 在训练完成后加载最优模型
best_model = CNN(input_channels, sequence_length, output_size)
best_model.load_state_dict(torch.load('CNN-3Class-batchsize=16-dropout=0.2.pt'))
best_model.to(device)
best_model.eval()  # 将模型切换到评估模式

# 使用最优模型对验证集进行预测
val_predictions = []
val_true_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        val_predictions.extend(predicted.cpu().numpy())
        val_true_labels.extend(labels.cpu().numpy())

# 计算混淆矩阵
conf_matrix = confusion_matrix(val_true_labels, val_predictions)

# 可视化混淆矩阵
classes = ['Class 0', 'Class 1', 'Class 2']  # 你的类别名称
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, values_format='.0f')
plt.title('Confusion Matrix - Validation Set')
plt.show()

# # 遍历新数据并计算每个类别的真正例、假正例和假负例数量
# with torch.no_grad():
#     outputs = model(inputs)
#     _, predicted = torch.max(outputs, 1)

# for i in range(len(predicted)):
#     true_label = y[i].item()
#     predicted_label = predicted[i].item()

#     if true_label == predicted_label:
#         class_counts[true_label][0] += 1  # True Positive
#     else:
#         class_counts[true_label][2] += 1  # False Negative
#         class_counts[predicted_label][1] += 1  # False Positive

# # 计算每个类别的准确率
# for label, counts in class_counts.items():
#     true_positive, false_positive, false_negative = counts
#     precision = true_positive / (true_positive + false_positive + 1e-6)  # Avoid division by zero
#     print(f'Class {label} Precision: {precision:.2f}')
