import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from mlp import MLP1, MLP2
from utils import train, test, trainStudent


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=65, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.95, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--temperature',default=20.0,type=float,metavar='N',
                        help='temperature (default: 20.0)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           #transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(28, padding = 2),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    teacher = MLP1(hidden_size=1200, dropout_p=0.25).to(device)
    teacher.load_state_dict(torch.load("2T65_9937.pb"))
    teacher_opt = optim.Adam(teacher.parameters(), lr=args.lr, weight_decay=1e-4)
    student = MLP2(hidden_size=30, dropout_p=0).to(device) # no dropout
    student_opt = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-4) # no weight_decay
    
    
    # Train teacher
    print("train teacher")
    scheduler = StepLR(teacher_opt, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        for group in teacher_opt.param_groups:
            for p in group['params']:
                state = teacher_opt.state[p]
                if 'step' in state.keys():
                    if(state['step']>=1024):
                        state['step'] = 1000
        train(args, train_loader, teacher, teacher_opt, epoch, device)
        test(args, test_loader, teacher, epoch, device)
        torch.save(teacher.state_dict(), "2T"+str(epoch)+".pb")
        scheduler.step()
    
    # Train student without distillation
    print("train student without distillation")
    scheduler = StepLR(student_opt, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        for group in student_opt.param_groups:
            for p in group['params']:
                state = student_opt.state[p]
                if 'step' in state.keys():
                    if(state['step']>=1024):
                        state['step'] = 1000
        train(args, train_loader, student, student_opt, epoch, device)
        test(args, test_loader, student, epoch, device)
        scheduler.step()
    
    # Train student with distillation
    print("train student with distillation")
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    T = args.temperature
    alpha= 0.5
    for epoch in range(1, args.epochs + 1):
        for group in student_opt.param_groups:
            for p in group['params']:
                state = student_opt.state[p]
                if 'step' in state.keys():
                    if(state['step']>=1024):
                        state['step'] = 1000
        trainStudent(args, train_loader, teacher, student, T, alpha, student_opt, epoch, device)
        test(args, test_loader, student, epoch, device)
        # scheduler.step()

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")
    

if __name__ == '__main__':
    main()
