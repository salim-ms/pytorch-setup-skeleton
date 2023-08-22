import torch
import torchvision
from data.load_data import load_datasets
from model.model import Net
import torch.optim as optim
import torch.nn as nn

PATH = './cifar_net.pth'
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
print(device)


def eval():
    _, test_loader = load_datasets()
    net = Net()
    net.to(device)
    net.load_state_dict(torch.load(PATH))
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(images)
            # print(outputs)
            # print(labels.shape)
            # print(outputs.shape)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__ == "__main__":
    eval()