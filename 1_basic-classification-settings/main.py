import torch
import torch.nn as nn
from dataset.load_data import load_cifar10
from model.resnet import resnet18
from trainer import train, test
from torch.utils.data import DataLoader


def main():
    ## Option
    device = 'cuda'
    data_root = ''
    lr = 0.01
    total_epoch = 30
    batch_size = 128


    ## Load dataset
    train_dataset = load_cifar10(data_root, train=True)
    test_dataset = load_cifar10(data_root, train=False)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


    ## Load model
    model = resnet18()
    model = model.to(device)


    ## Load optimizer
    optim = torch.optim.SGD(model.parameters(), lr=lr)


    ## Load criterion
    criterion = nn.CrossEntropyLoss()


    ## Run train and test
    for epoch in range(total_epoch):
        ## Train for each single epoch
        train_result = train(train_loader, model, optim, criterion)

        ## Test for each single epoch
        test_result = test(test_loader, model, criterion)    

        print('Epoch %d/%d : train_loss %.2f, train_acc %.2f, test_loss %.2f, test_acc %.2f' \
                %(epoch, total_epoch, train_result['loss'], train_result['acc'], test_result['loss'], test_result['acc']))
        
        
if __name__=='__main__':
    main()