import os


def Log_Print():
    log_path = './data'
    log_name = 'ResNet.txt'
    log_file = open(os.path.join(log_path, log_name), encoding='utf-8')
    for line in log_file:
        # 输出日志记录
        print(line, end='')
