import time
import torch
import torch.nn.functional as F


def print_info(info):
    message = ('Epoch: {}/{}, Duration: {:.3f}s, Train Loss: {:.4f}, '
               'Test Loss: {:.4f}, Test Acc: {:.4f}').format(
                   info['current_epoch'], info['epochs'], info['t_duration'],
                   info['train_loss'], info['test_loss'], info['acc'])
    print(message)


def run(model, epochs, train_loader, test_loader, optimizer, scheduler,
        device):
    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train(model, train_loader, optimizer, device)
        t_duration = time.time() - t
        scheduler.step()
        acc, test_loss = test(model, test_loader, device)

        info = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'acc': acc,
            'current_epoch': epoch,
            'epochs': epochs,
            't_duration': t_duration
        }

        print_info(info)


def train(model, train_loader, optimizer, device):
    model.train()

    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(model, test_loader, device):
    model.eval()

    correct = 0
    total_loss = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data = data.to(device)
            out = model(data)
            total_loss += F.nll_loss(out, data.y).item()
            pred = out.max(1)[1]
            correct += pred.eq(data.y).sum().item()
    return correct / len(test_loader.dataset), total_loss / len(test_loader)
