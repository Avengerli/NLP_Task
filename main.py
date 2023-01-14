import CNN_Train
from utils import MyLog

if __name__ == '__main__':
    # LeNet模型，训练100epoch
    # CNN_Train.main(1, 100)

    # AlexNet模型，训练100epoch
    # CNN_Train中103、108行参数设置为227，即
    # transforms.RandomResizedCrop(227),
    # transforms.CenterCrop(227),
    CNN_Train.main(2, 100)

    # ResNet模型，训练100epoch
    # CNN_Train中103、108行参数设置为224，即
    # transforms.RandomResizedCrop(224),
    # transforms.CenterCrop(224),
    # CNN_Train.main(3,100)
