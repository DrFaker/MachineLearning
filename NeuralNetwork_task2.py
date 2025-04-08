import os
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# 读取数据
def read_data(file_path):
    with open(file_path, 'rb') as file:
        data = file.read().decode('utf-8', errors='ignore')
    return data

def read_labels(file_path):
    with open(file_path, 'r') as file:
        labels = file.read().splitlines()
    return list(map(int, labels))

# 解析文档
def parse_documents(data):
    documents = re.split(r'p\d+\naV', data)
    documents = [doc.strip() for doc in documents if doc.strip()]
    return documents

# 加载数据
train_texts_data = read_data('train_texts.dat')
train_labels_data = read_labels('train_labels.txt')
test_texts_data = read_data('test_texts.dat')

train_documents = parse_documents(train_texts_data)
test_documents = parse_documents(test_texts_data)

# 特征提取
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_documents)
X_test = vectorizer.transform(test_documents)

y_train = np.array(train_labels_data)

# 划分验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 构建神经网络模型
model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=20, alpha=1e-4,
                      solver='adam', verbose=10, random_state=42,
                      learning_rate_init=0.001)

# 训练模型
model.fit(X_train, y_train)

# 验证模型
y_val_pred = model.predict(X_val)
print(classification_report(y_val, y_val_pred))

# 预测测试集
y_test_pred = model.predict(X_test)

# 保存预测结果
np.savetxt('test_labels.txt', y_test_pred, fmt='%d')

# 保存模型
with open('news_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)