import numpy as np


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def load_data(train_file, test_file):
    # 加载训练数据
    train_data = np.loadtxt(train_file)
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    # 处理缺失值
    X_train[np.isnan(X_train)] = 0

    # 加载测试数据
    test_data = np.loadtxt(test_file)
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    # 处理缺失值
    X_test[np.isnan(X_test)] = 0

    return X_train, y_train, X_test, y_test


def logistic_regression(X_train, y_train, num_iter=1000, learning_rate=0.01):
    m, n = X_train.shape
    # 初始化参数时考虑偏置项，所以是n+1维
    theta = np.zeros(n + 1)

    # 添加偏置项
    X_train = np.hstack([np.ones((m, 1)), X_train])

    for _ in range(num_iter):
        z = np.dot(X_train, theta)
        h = sigmoid(z)
        gradient = np.dot(X_train.T, (h - y_train)) / m
        theta -= learning_rate * gradient

    return theta


def predict(X, theta):
    # 预测时也要添加偏置项
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    prob = sigmoid(np.dot(X, theta))
    return (prob >= 0.5).astype(int)


def evaluate(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    return accuracy


def main():
    # 加载数据
    X_train, y_train, X_test, y_test = load_data('horseColicTraining.txt', 'horseColicTest.txt')

    # 训练模型
    theta = logistic_regression(X_train, y_train)

    # 预测
    y_pred = predict(X_test, theta)

    # 评估
    accuracy = evaluate(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.2%}")


if __name__ == "__main__":
    main()