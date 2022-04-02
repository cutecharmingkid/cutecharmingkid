import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # 读取训练集和测试集文件
    data = pd.read_csv("boston.csv", header=0)
    row_cnt = data.shape[0]  # 506
    column_cnt = data.shape[1]  # 14

    #X = np.empty([row_cnt, column_cnt - 1])  # 测试np.shape是否正确，生成两个随机未初始化的矩阵，要-1是因为结果不要放进去
    #Y = np.empty([row_cnt, 1])

    X = np.array(data.iloc[:row_cnt, :column_cnt - 1])
    Y = np.array(data.iloc[:row_cnt, column_cnt - 1])

    X = (X - X.mean()) / X.std() #对X进行标准化处理

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    #使用理论求解法求解析解（精确解）
    y_train = y_train.reshape(455, 1)
    X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
    w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
    print(w)
    loss = np.sum((y_train - X_b.dot(w)) ** 2 / len(X_b))
    print(loss)
    X_c = np.hstack([np.ones((len(X_test), 1)), X_test])  #测试用专用数据
    y_test_0 = X_c.dot(w)  # 使用理论求解法得到的系数做预测
    print(y_test.reshape(1, 51))  #输出真实值
    print(y_test_0.reshape(1, 51))




    #使用梯度下降法求解
    W= np.ones((column_cnt, 1))   #第一位表示常数项，后面表示13个权值 ,权值向量
    X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  #基于数据的系数矩阵首列填充1的矩阵
    y_train=y_train.reshape(455,1)
    eta=[0.001,0.005,0.01,0.05,0.1]  #学习率

    for eta in eta:
        loss_record=[]
        W = np.ones((column_cnt, 1))  # 第一位表示常数项，后面表示13个权值 ,权值向量
        for time in range(100000):
            res = np.empty((len(W), 1))  # 用于存储每个量的误差值,是一列的
            res[0] = np.sum(X_b.dot(W) - y_train)
            cur = X_b.dot(W) - y_train  # 为了提高算法效率将公共部分提出优化
            for i in range(1, len(W)):
                res[i] = np.sum(np.multiply(cur, X_b[:, i].reshape(455,1)) )
                # print( X_b[:, i].shape) 用于测试矩阵的形状
            res = res * 2 / len(X_b)
            W = W - eta * res
            loss = np.sum((y_train - X_b.dot(W)) ** 2 / len(X_b))  # 这是计算总损失的式子

            if time % 1000 == 0:
                loss_record.append(loss)

        print(W)
        print("loss={}".format(loss))

        #画图
        import matplotlib.pyplot as plt

        y1 = loss_record
        x=np.linspace(1000,100000,100)

        plt.plot(x, y1, linewidth=3, color='r', marker='o',label=str(eta),
                 markerfacecolor='blue', markersize=3)

        plt.xlabel('iterations ')
        plt.ylabel('loss')
        plt.title('Convergence of the loss function')
        plt.legend()
        plt.show()

        print("leaning rate={}".format(eta))
        y_test_1 = X_c.dot(W)  # 使用梯度下降法得到的系数做预测
        print(y_test_1.reshape(1, 51))







