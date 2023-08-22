import torch
import torchvision
from data.load_data import load_datasets
from model.model import Net
import torch.optim as optim
import torch.nn as nn

PATH = './cifar_net.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)



def train():
    train_loader, test_loader = load_datasets()
    net = Net()
    net.to(device)

    # create loss
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    # training loop
    for epoch in range(2):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            images, labels = batch[0].to(device), batch[1].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    torch.save(net.state_dict(), PATH)

if __name__ == "__main__":
    train()