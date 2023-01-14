import json
import torch
import datetime
import torchvision
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
from torch.optim import lr_scheduler
from utils.MyDataSet import MyDataSet
from model import CNN_1LeNet
from model import CNN_2AlexNet
from model import CNN_3ResNet


# 是否开启GPU加速
device = "cuda" if torch.cuda.is_available() else "cpu"
# 记录
loss_train_x, loss_train_y, loss_test_x, loss_test_y, acc_x, acc_y = [], [], [], [], [], []
# 记录 (需要根据模型修改文件名)
doc = open('./data/AlexNet.txt', 'w')
plt = open('./data/AlexPlt.txt', 'w')


# 数据集导入模块
def Data_Loader(sign):
    if sign == 'CIFAR10':
        return CIFAR10_Data_Loader()
    elif sign == 'MiniImageNet':
        return MiniImageNet_Data_Loader()
    else:
        return None


# 加载CIFAR10数据集
def CIFAR10_Data_Loader():
    # 定义Transform
    data_transform = {
        'train': transforms.Compose([transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # 获取CIFAR-10训练集
    train_data = torchvision.datasets.CIFAR10(
        root='/data/',
        train=True,
        transform=data_transform['train'],
        download=True)
    # 获取CIFAR-10测试集
    test_data = torchvision.datasets.CIFAR10(
        root='/data/',
        train=False,
        transform=data_transform['val'],
        download=True)
    # 按照每个batch 64个实例进行分割
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True,
        drop_last=False)
    # 按照每个batch 1000个实例进行分割
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=100,
        shuffle=False,
        drop_last=False)

    '''
    # 绘制训练数据第一个batch
    images, labels = next(iter(train_data_loader))
    # 将64张图片拼接为1张
    image = torchvision.utils.make_grid(images)
    # 数据格式转换
    # 4DTensor格式：
    #       N：Batch，批处理大小，表示一个batch中的图像数量
    #       C：Channel，通道数，表示一张图像中的通道数
    #       H：Height，高度，表示图像垂直维度的像素数
    #       W：Width，宽度，表示图像水平维度的像素数
    # 4DNumpy格式：
    #       N：Batch，批处理大小，表示一个batch中的图像数量
    #       H：Height，度，表示图像垂直维度的像素数
    #       W：Width，宽度，表示图像水平维度的像素数
    #       C：Channel，通道数，表示一张图像中的通道数
    # 本例中：Tensor————>Numpy为(1,2,0)
    image = image.numpy().transpose(1, 2, 0)
    # 绘制图片
    pyplot.imshow(image)
    pyplot.show()
    '''

    # 返回测试集和训练集
    return train_data_loader, test_data_loader


# 加载MiniImageNet数据集
def MiniImageNet_Data_Loader():
    # 定义路径
    root_dir = '.\\data\\Mini-ImageNet\\'
    json_path = '.\\data\\Mini-ImageNet\\classes_name.json'
    # 定义Transform
    data_transform = {
        'train': transforms.Compose([transforms.RandomResizedCrop(227),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(227),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # 获取Mini-ImageNet训练集
    train_data = MyDataSet(root_dir=root_dir,
                           csv_name="new_train.csv",
                           json_path=json_path,
                           transform=data_transform['train'])
    # 获取Mini-ImageNet测试集
    test_data = MyDataSet(root_dir=root_dir,
                          csv_name="new_val.csv",
                          json_path=json_path,
                          transform=data_transform['val'])
    # 按照每个batch 64个实例进行分割
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True,
        drop_last=False)
    # 按照每个batch 100个实例进行分割
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=100,
        shuffle=False,
        drop_last=False)

    '''
    # 绘制训练数据第一个batch
    images, labels = next(iter(train_data_loader))
    # 将64张图片拼接为1张
    image = torchvision.utils.make_grid(images)
    # 数据格式转换
    # 4DTensor格式：
    #       N：Batch，批处理大小，表示一个batch中的图像数量
    #       C：Channel，通道数，表示一张图像中的通道数
    #       H：Height，高度，表示图像垂直维度的像素数
    #       W：Width，宽度，表示图像水平维度的像素数
    # 4DNumpy格式：
    #       N：Batch，批处理大小，表示一个batch中的图像数量
    #       H：Height，度，表示图像垂直维度的像素数
    #       W：Width，宽度，表示图像水平维度的像素数
    #       C：Channel，通道数，表示一张图像中的通道数
    # 本例中：Tensor————>Numpy为(1,2,0)
    image = image.numpy().transpose(1, 2, 0)
    # 绘制图片
    pyplot.imshow(image)
    pyplot.show()
    '''

    # 返回测试集和训练集
    return train_data_loader, test_data_loader


# 模型训练模块
def train(data_loader, model, loss_fun, optimizer, epoch):
    # 设置训练标志
    model.train()
    # 平均损失
    loss_avg = 0
    # 读取每个batch的训练数据
    for batch_id, (image, label) in enumerate(data_loader):
        # GPU训练
        image, label = image.to(device), label.to(device)
        # 清空过往梯度
        optimizer.zero_grad()
        # 训练
        output = model(image)
        # 计算损失
        loss = loss_fun(output, label)
        # 反向传播
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 累计损失
        loss_avg += loss.cpu().detach().numpy()
        # 每隔100个Iteration记录一次训练效果
        if batch_id % 100 == 99:
            # 计算平均损失
            loss_train_x.append((((epoch - 1) * 7) + ((batch_id + 1) / 100)))
            loss_train_y.append((loss_avg / 100))
            # 输出
            print(f'Train Epoch: {epoch} | [' + f'{(batch_id + 1) * 64}'.rjust(5)
                  + f' / 48000] : loss = {loss_avg / 100}')
            print(f'Train Epoch: {epoch} | [' + f'{(batch_id + 1) * 64}'.rjust(5)
                  + f' / 48000] : loss = {loss_avg / 100}', file=doc)
            # 平均损失置零
            loss_avg = 0


# 模型测试模块
def test(data_loader, model, loss_fun, epoch):
    # 设置测试标志
    model.eval()
    # 测试正确率
    correct = 0
    # 平均损失
    loss_avg = 0
    # 不需要梯度的记录
    with torch.no_grad():
        for image, label in data_loader:
            # GPU测试
            image, label = image.to(device), label.to(device)
            # 测试
            output = model(image)
            # 计算损失
            loss = loss_fun(output, label)
            # 累计损失
            loss_avg += loss.cpu().detach().numpy()
            # 预测结果
            pred = output.argmax(dim=1, keepdim=True)
            # 正确个数
            correct += pred.eq(label.view_as(pred)).sum().item()
    # 记录损失函数
    loss_test_x.append(epoch)
    loss_test_y.append((loss_avg / 120))
    # 记录测试结果
    acc_x.append(epoch)
    acc_y.append((correct / len(data_loader.dataset)))
    # 输出
    print(f'Test Epoch: {epoch} : loss = ' + f'{loss_avg / 120}')
    print(f'Test Epoch: {epoch} : loss = ' + f'{loss_avg / 120}', file=doc)
    print(f'Test Epoch: {epoch} : accuracy = ' + f'{100. * (correct / len(data_loader.dataset))}')
    print(f'Test Epoch: {epoch} : accuracy = ' + f'{100. * (correct / len(data_loader.dataset))}', file=doc)
    # 设置训练标志
    model.train()


# 保存模型
def save_data():
    # 获得最佳模型
    max_acc = max(acc_y)
    max_index = acc_y.index(max_acc) + 1
    # 输出间隔符
    print('# ---------------------------------------------------- #')
    print('# ---------------------------------------------------- #', file=doc)
    # 输出最佳模型
    print(f'Best Model : AlexNet{max_index}.pt')
    print(f'Best Model : AlexNet{max_index}.pt', file=doc)
    print(f'Best Model Accuracy : {max_acc}')
    print(f'Best Model Accuracy : {max_acc}', file=doc)
    # 转化为dict
    data = {'train_loss': dict(zip(loss_train_x, loss_train_y)), 'test_loss': dict(zip(loss_test_x, loss_test_y)),
            'accuracy': dict(zip(acc_x, acc_y))}
    # 储存
    plt.write(json.dumps(data))


# 主函数
def main(sign, epochs):
    start = datetime.datetime.now()
    # 加载模型
    if sign == 1:
        model = CNN_1LeNet.CNN_1LeNet().to(device)
        model_name = 'LeNet'
        # 加载数据
        train_data_loader, test_data_loader = Data_Loader('CIFAR10')
    elif sign == 2:
        model = CNN_2AlexNet.CNN_2AlexNet().to(device)
        model_name = 'AlexNet'
        # 加载数据
        train_data_loader, test_data_loader = Data_Loader('MiniImageNet')
    else:
        model = CNN_3ResNet.CNN_3ResNet(CNN_3ResNet.ResBlock).to(device)
        model_name = 'ResNet'
        # 加载数据
        train_data_loader, test_data_loader = Data_Loader('MiniImageNet')
    # 损失函数
    loss_fun = nn.CrossEntropyLoss().to(device)
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # 学习率衰减
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    # 迭代训练
    for epoch in range(1, epochs + 1):
        # 输出间隔符
        print('# ---------------------------------------------------- #')
        print('# ---------------------------------------------------- #', file=doc)
        # 训练集训练
        train(train_data_loader, model, loss_fun, optimizer, epoch)
        # 学习率衰减
        scheduler.step()
        # 测试集测试
        test(test_data_loader, model, loss_fun, epoch)
        # 保存模型
        torch.save(model, './data/model/' + model_name + f'{epoch}.pt')
    # 记录数据
    save_data()
    # 计算用时
    end = datetime.datetime.now()
    times = (end - start).seconds
    hour = times / 3600
    times = times % 3600
    minutes = times / 60
    second = times % 60
    print('Time expended: %dh-%dm-%ds' % (hour, minutes, second))
    print('Time expended: %dh-%dm-%ds' % (hour, minutes, second), file=doc)
