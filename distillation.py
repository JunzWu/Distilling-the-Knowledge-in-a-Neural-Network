import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Hooker_for_logits:
    def __init__(self, model):
        self.logits = []
        self.model = model
        self.model.fc.register_forward_hook(self.fetch_logits)

    def fetch_logits(self, input, output):
        self.logits = output

    def clear_logits(self):
        self.logits = []

    def forward(self, train_input):
        output = self.model(train_input)
        logits = self.logits
        self.clear_logits()
        return output, logits


class Distillation(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(Distillation, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 64, 4, 1, 2)
        self.conv2 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.batch_norm = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()

        self.fc_net = nn.Sequential(
            nn.Linear(64 * 4 * 4, 500),
            nn.Linear(500, 500),
            nn.Linear(500, TOTAL_CLASSES),
        )

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()
        # self.lstm2.reset_state()
        # self.dropout2.reset_state()

    def forward(self, x):
        x = self.ReLU(self.batch_norm(self.conv1(x)))
        x = self.pool(self.conv2(x))
        x = self.ReLU(self.batch_norm(self.conv2(x)))
        x = self.pool(self.conv2(x))
        x = self.ReLU(self.batch_norm(self.conv2(x)))
        x = self.conv3(x)
        x = self.ReLU(self.batch_norm(self.conv3(x)))
        x = self.drop(self.ReLU(self.batch_norm(self.conv3(x))))

        x = x.view(-1, 64 * 4 * 4)
        x = self.fc_net(x)

        return x


def train():
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)

    train_loss_over_epochs = []

    for epoch in range(EPOCHS):
        print(f'epoch: {epoch}')

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data[0].cuda(), data[1].cuda()

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            dis_outputs, dis_logits = distillation_model.forward(inputs)
            soft_log_probs = F.log_softmax(dis_logits / TEMPERATURE, dim=1)
            distillation_loss = criterion(distillation_outputs, labels)

            _, teacher_logits = teacher_model.forward(inputs)
            soft_targets = F.softmax(teacher_logits / TEMPERATURE, dim=1)
            dis_teacher_loss = criterion()
            distillation_loss = F.kl_div(
                soft_log_probs, soft_targets.detach(),
                size_average=False) / soft_targets.shape[0]

            loss = LOSS_WT_STUDENT * distillation_loss + LOSS_WT_DISS * distillation_loss
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)
        print('[%d] loss: %.3f' % (epoch + 1, running_loss))

        train_loss_over_epochs.append(running_loss)
        test_accuracy, test_classwise_accuracy = cal_test_accuracy()
        print(f'Accuracy on test set: {test_accuracy}%')


def main():
    global teacher_model
    teacher_model = torch.hub.load('pytorch/vision',
                                   'resnet152',
                                   pretrained=True)
    teacher_model = teacher_model.cuda()
    teacher_model = Hooker_for_logits(teacher_model)

    global distillation_model
    distillation_model = Distillation_model()
    distillation_model = distillation_model.cuda()
    distillation_model = Hooker_for_logits(distillation_model)


if __name__ == '__main__':
    main()
