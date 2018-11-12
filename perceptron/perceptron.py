# charter2 perceptron demo data
import numpy as np
import random


def get_train_dataset():
    x = [3, 3, 4, 3, 1, 1]
    x = np.reshape(x, [3, 2])
    y = [1, 1, -1]
    return x, y

def cost_compute(x,w,b):
    tr_size = len(x)
    cost = 0
    for i in range(tr_size):
        xi = x[i]
        sum_x_w_b = np.dot(xi, np.transpose(w)) + b
        cost += np.abs(sum_x_w_b)  # 计算代价函数
    return cost

def train_with_batch_GD(x, y, learning_step=0.1, tr_loops=10):
    print("=======start batch gradient descent training==========")
    tr_size = len(x)
    w = np.zeros([2])
    b = 0
    wrong_indexs = [0]
    tr_count = 0
    while (len(wrong_indexs) > 0 and tr_count < tr_loops):
        wrong_indexs.clear()
        tr_count += 1
        # 判断误分类点
        for i in range(tr_size):
            xi = x[i]
            yi = y[i]
            sum_x_w_b = np.dot(xi, np.transpose(w)) + b
            if yi * sum_x_w_b <= 0:
                wrong_indexs.append(i)  # 误分类

        # 进行批量梯度下降
        for j in wrong_indexs:
            j = int(j)
            xj = x[j]
            yj = int(y[j])
            w = w + learning_step * yj * xj
            b = b + learning_step * yj
        print("loops:", tr_count, "  w:", w, "  b:", b, "  wrong_index:", wrong_indexs)
        # 输出每一轮的cost值
        print("cost:", cost_compute(x,w,b))
    print("=======end batch gradient descent training==========")
    # 继续下一轮训练直至没有误分类点或达到训练次数上限
    return w, b


def train_with_stochastic_GD(x, y, learning_step=0.1, tr_loops=10, cost_limit=10):
    print("=======start stochastic gradient descent training==========")
    tr_size = len(x)
    w = np.zeros([2])
    b = 0

    for tr_count in range(tr_loops):
        # 取样本数-1的样本进行训练
        i = random.randint(0,tr_size-1)
        xi = x[i]
        yi = y[i]
        sum_x_w_b = np.dot(xi, np.transpose(w)) + b
        if yi * sum_x_w_b <= 0:#如果分类错误则进行梯度下降
            w = w + learning_step*yi*xi
            b = b + learning_step*yi

        print("loops:", tr_count,"  w:", w, "  b:", b)
        cost = cost_compute(x,w,b)
        # 输出每一轮的cost值
        print("cost:", cost)
        if cost <= cost_limit:
            break
    print("=======end stochastic gradient descent training==========")
    # 继续下一轮训练直至没有误分类点或达到训练次数上限
    return w, b


def predict(x, w, b):
    p_y = []
    for xi in x:
        sum_x_w_b = np.dot(xi, np.transpose(w)) + b
        if sum_x_w_b > 0:
            p_y.append(1)
        else:
            p_y.append(-1)
    return p_y


# ================================
# 训练及预测
x, y = get_train_dataset()
bgd_w, bgd_b = train_with_batch_GD(x, y, 1, 100)
bgd_y = predict(x, bgd_w, bgd_b)

sgd_w, sgd_b = train_with_stochastic_GD(x,y,1,100,10)
sgd_y = predict(x, sgd_w, sgd_b)

print("y:", y)
print("batch GD:", bgd_y, bgd_w, bgd_b)
print("stochastic GD:", sgd_y, sgd_w, sgd_b)
