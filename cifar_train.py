import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchnet.meter as tnt
import torchvision
import torchvision.transforms as transforms
import logging
import os
import pdb

#stdout_handler = logging.StreamHandler(sys.stdout)
msglogger = logging.getLogger()
#msglogger.addHandler(stdout_handler)


class CifarTrain:
    def __init__(self, log_time, batch_sz=2):
        self.log_time = log_time
        self.best_acc = 0.0
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.load_dataset(batch_sz)

    def load_dataset(self, batch_sz):
        self.trainset = torchvision.datasets.CIFAR10(
            root='../data.cifar10', train=True, download=False, transform=self.transform_train)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_sz, shuffle=True, num_workers=2)
        
        self.testset = torchvision.datasets.CIFAR10(
            root='../data.cifar10', train=False, download=False, transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=batch_sz, shuffle=False, num_workers=2)
        
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
    
    def initialize(self, model, lr=0.1, momentum=0.9, weight_decay=1e-4):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        if self.device == 'cuda':
             self.model = torch.nn.DataParallel(model, [0])
    
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    
    # Training
    def train(self, epoch, print_step=100):
        msglogger.info("Epoch: {}".format(epoch))
        self.model.train()
        classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
    
            train_loss += loss.item()
            # _, predicted = outputs.max(1)
            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()
            classerr.add(outputs.detach(), targets)
            if ((batch_idx + 1) % print_step) == 0:
                msglogger.info('[%d / %d] ==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                    batch_idx + 1, len(self.trainloader), classerr.value()[0], classerr.value()[1], train_loss / (batch_idx + 1))
        

        


    def test(self, epoch, print_step=100):
        self.model.eval()
        classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
    
                test_loss += loss.item()
                classerr.add(outputs.detach(), targets)
                if ((batch_idx + 1) % print_step) == 0:
                    msglogger.info('[%d / %d] ==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                        batch_idx + 1, len(self.testloader), classerr.value()[0], classerr.value()[1], test_loss/(batch_idx + 1)) 
    
        # Save checkpoint.
        acc = classerr.value()[0]
        save_path = './logs/' + self.log_time + '/checkpoint/ckpt.pth'
        if acc > self.best_acc:
            save_path = './logs/' + self.log_time +'/checkpoint/best.pth'
            self.best_acc = acc
        
        print('Saving..')
        state = {
    		'net': self.model.state_dict(),
    		'acc': acc,
    		'epoch': epoch,
    	}
        if not os.path.isdir('./logs/' + self.log_time + '/checkpoint'):
    	    os.mkdir('./logs/' + self.log_time +'/checkpoint')
        torch.save(state, save_path)

        return acc
    

