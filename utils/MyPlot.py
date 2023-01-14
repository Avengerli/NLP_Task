import json
import matplotlib.pyplot as plt


def MyPlot():
    # 读取训练记录
    json_path = 'D:\\data\\ResPlt.json'
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    # 拆分
    train_loss = data['train_loss']
    test_loss = data['test_loss']
    accuracy = data['accuracy']
    # 转化
    loss_train_x = list(train_loss.keys())
    loss_train_x = [float(x) for x in loss_train_x]
    loss_train_y = list(train_loss.values())
    loss_train_y = [float(y) for y in loss_train_y]
    temp_x, temp_y = [], []
    for i in range(0, 700):
        if ((i + 1) % 7) == 0:
            temp_y.append(loss_train_y[i])
    for i in range(1, 101):
        temp_x.append(float(i))
    print(len(temp_x))
    print(len(temp_y))
    loss_test_x = list(test_loss.keys())
    loss_test_x = [float(x) for x in loss_test_x]
    loss_test_y = list(test_loss.values())
    loss_test_y = [float(y) for y in loss_test_y]
    acc_x = list(accuracy.keys())
    acc_x = [float(x) for x in acc_x]
    acc_y = list(accuracy.values())
    acc_y = [float(y) * 100 for y in acc_y]
    # 输出
    print(len(loss_train_x))
    print(len(loss_test_x))
    print(len(acc_x))
    plt.plot(loss_test_x, loss_test_y, color='red', linestyle='-', label='Test Loss')
    plt.plot(temp_x, temp_y, color='green', linestyle='--', label='Train Loss')
    plt.legend(loc="upper right")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()

    plt.plot(acc_x, acc_y, color='blue', linestyle='-', label='Test Accuracy')
    plt.legend(loc="lower right")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
