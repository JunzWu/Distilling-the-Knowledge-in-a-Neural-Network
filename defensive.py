import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from adversary import adversarial_eval

MNIST_transform=transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])

MNIST_trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=MNIST_transform)

MNIST_testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=MNIST_transform)

MNIST_train_loader = torch.utils.data.DataLoader(MNIST_trainset, batch_size=128,
                                                 shuffle=True, num_workers=4, pin_memory=True)
MNIST_test_loader = torch.utils.data.DataLoader(MNIST_testset, batch_size=128,
                                                shuffle=False, num_workers=4, pin_memory=True)
MNIST_adversary_loader = torch.utils.data.DataLoader(MNIST_testset, batch_size=1,
                                                     shuffle=True, num_workers=1)

CIFAR_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

CIFAR_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=CIFAR_transform)
CIFAR_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=CIFAR_transform)

CIFAR_train_loader = torch.utils.data.DataLoader(CIFAR_trainset, batch_size=128,
                                                 shuffle=True, num_workers=4, pin_memory=True)
CIFAR_test_loader = torch.utils.data.DataLoader(CIFAR_testset, batch_size=128,
                                                shuffle=False, num_workers=4, pin_memory=True)
CIFAR_adversary_loader = torch.utils.data.DataLoader(CIFAR_testset, batch_size=1,
                                                     shuffle=True, num_workers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fetch_logits(self, input, output):
    global logits
    logits = output

class Hooker_for_logits:
    def __init__(self, model):
        self.model = model
        self.model.fc3.register_forward_hook(fetch_logits)

    def forward(self, train_input):
        output = self.model(train_input)
        return output, logits

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 200)
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(200,200)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = x.view(-1, 64 * 4 * 4)

        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        output = F.log_softmax(x, dim=1)

        return output

class CIFAR_Net(nn.Module):
    def __init__(self):
        super(CIFAR_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(256,256)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = x.view(-1, 128 * 5 * 5)

        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        output = F.log_softmax(x, dim=1)

        return output

MNIST_teacher_net = MNIST_Net()
MNIST_student_net = MNIST_Net()
MNIST_teacher_net.to(device)
MNIST_student_net.to(device)

CIFAR_teacher_net = CIFAR_Net()
CIFAR_student_net = CIFAR_Net()
CIFAR_teacher_net.to(device)
CIFAR_student_net.to(device)

criterion = nn.CrossEntropyLoss()
MNIST_optimizer = optim.SGD(MNIST_student_net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True)
CIFAR_optimizer = optim.SGD(CIFAR_student_net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True)

temperature = 1

PATH = './mnist_net.pth'
MNIST_teacher_net.load_state_dict(torch.load(PATH))

MNIST_teacher_hooker = Hooker_for_logits(MNIST_teacher_net)
MNIST_student_hooker = Hooker_for_logits(MNIST_student_net)

for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(MNIST_train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        MNIST_student_outputs, MNIST_student_logits = MNIST_student_hooker.forward(inputs)
        soft_log_probs = F.log_softmax(MNIST_student_logits / temperature, dim=1)
        dis_loss = criterion(MNIST_student_outputs, labels)
        # print('dis_loss: ', dis_loss)

        _, MNIST_teacher_logits = MNIST_teacher_hooker.forward(inputs)
        soft_targets = (temperature ** 2) * F.softmax(MNIST_teacher_logits / temperature, dim=1)
        teacher_loss = F.kl_div(soft_log_probs,
                                soft_targets.detach(),
                                size_average=False) / soft_targets.shape[0]

        # zero the parameter gradients
        MNIST_optimizer.zero_grad()

        # forward + backward + optimize
        # loss = criterion(MNIST_student_outputs, labels)
        loss = 0.5 * dis_loss + 0.5 * teacher_loss
        loss.backward()
        MNIST_optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('[Epoch %d] loss: %.3f' % (epoch + 1, running_loss))
    running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in MNIST_test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = MNIST_student_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))

# torch.save(MNIST_student_net.state_dict(), PATH)

fgsm_acc, pgd_acc, fgsm_examples, pgd_examples = adversarial_eval(MNIST_teacher_net,
                                                                  device,
                                                                  MNIST_adversary_loader,
                                                                  0.10)
print('fgsm_acc: ', fgsm_acc)
print('pgd_acc: ', pgd_acc)
print('len(fgsm_examples): ', len(fgsm_examples))
print('len(pgd_examples): ', len(pgd_examples))

correct = 0
total = 0
for data in fgsm_examples:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = MNIST_student_net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on fgsm adversarial images: %d %%' % (
    100 * correct / total))

correct = 0
total = 0
for data in pgd_examples:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = MNIST_student_net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on pgd adversarial images: %d %%' % (
    100 * correct / total))

print('\nCIFAR:\n\n')

PATH = './cifar_net.pth'
CIFAR_teacher_net.load_state_dict(torch.load(PATH))

CIFAR_teacher_hooker = Hooker_for_logits(CIFAR_teacher_net)
CIFAR_student_hooker = Hooker_for_logits(CIFAR_student_net)

for epoch in range(50):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(CIFAR_train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        CIFAR_student_outputs, CIFAR_student_logits = CIFAR_student_hooker.forward(inputs)
        soft_log_probs = F.log_softmax(CIFAR_student_logits / temperature, dim=1)
        dis_loss = criterion(CIFAR_student_outputs, labels)
        # print('dis_loss: ', dis_loss)

        _, CIFAR_teacher_logits = CIFAR_teacher_hooker.forward(inputs)
        soft_targets = (temperature ** 2) * F.softmax(CIFAR_teacher_logits / temperature, dim=1)
        teacher_loss = F.kl_div(soft_log_probs,
                                soft_targets.detach(),
                                size_average=False) / soft_targets.shape[0]

        # zero the parameter gradients
        CIFAR_optimizer.zero_grad()

        # forward + backward + optimize
        # loss = criterion(CIFAR_student_outputs, labels)
        loss = 0.5 * dis_loss + 0.5 * teacher_loss
        loss.backward()
        CIFAR_optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('[Epoch %d] loss: %.3f' % (epoch + 1, running_loss))
    running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in CIFAR_test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = CIFAR_student_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))

# torch.save(CIFAR_student_net.state_dict(), PATH)

fgsm_acc, pgd_acc, fgsm_examples, pgd_examples = adversarial_eval(CIFAR_teacher_net,
                                                                  device,
                                                                  CIFAR_adversary_loader,
                                                                  0.10)
print('fgsm_acc: ', fgsm_acc)
print('pgd_acc: ', pgd_acc)
print('len(fgsm_examples): ', len(fgsm_examples))
print('len(pgd_examples): ', len(pgd_examples))

correct = 0
total = 0
for data in fgsm_examples:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = CIFAR_student_net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on fgsm adversarial images: %d %%' % (
    100 * correct / total))

correct = 0
total = 0
for data in pgd_examples:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = CIFAR_student_net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on pgd adversarial images: %d %%' % (
    100 * correct / total))

print('Finished')
