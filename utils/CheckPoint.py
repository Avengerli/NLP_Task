import torch
import os


def save_checkpoint(model, model_name, loss_fun, optimizer, epoch, doc):
    # 模型保存路径
    model_path = './data/model/' + model_name + f'{epoch}.pth'
    # 字典
    state = {
        'epoch': epoch,
        'model': model,
        'loss': loss_fun,
        'optim': optimizer.state_dict()}
    # 保存模型
    torch.save(state, model_path)
    #
    print('Checkpoint saved to {}'.format(model_path))
    print('Checkpoint saved to {}'.format(model_path), file=doc)


def load_checkpoint(model_path, model, optimizer, doc):
    # 判断断点是否存在
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        loss_fun = checkpoint['loss']
        print('Success load checkpoint {}'.format(model_path))
        print('Success load checkpoint {}'.format(model_path), file=doc)
        return model, loss_fun, optimizer, epoch
    else:
        print('There is no checkpoint {}'.format(model_path))
        print('There is no checkpoint {}'.format(model_path), file=doc)
        return None
