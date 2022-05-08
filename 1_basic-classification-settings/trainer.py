

def train(train_loader, model, optim, criterion):
    
    result = {'loss': train_loss, 'acc': train_acc}
    return result


def test(test_loader, model, criterion):
    result = {'loss': test_loss, 'acc': test_acc}
    return result
