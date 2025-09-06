from waggon import functions as f

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchview import draw_graph  # For visualizing model architecture

import numpy as np
from tqdm import tqdm
from scipy.stats import qmc
from datetime import datetime


class LeNet5(nn.Module):
    def __init__(
        self,
        # input layer
        input_size=32,   
        input_channels=3,

        # C1
        conv1_out=6,    
        conv1_kernel=5,

        # S2
        pool1_kernel=2,  
        pool1_stride=1,

        # C3
        conv2_out=16,    
        conv2_kernel=5,
        
        # S4
        pool2_kernel=2,  
        pool2_stride=2,

        # C5
        conv3_out=120,    
        conv3_kernel=5,

        # F6
        fc_out_size=84,

        num_classes=10   
    ):
        super().__init__()
        
        # Convolution layers
        self.convolutions = nn.Sequential(
            nn.Conv2d(input_channels, conv1_out, kernel_size=conv1_kernel),
            nn.ReLU(),
            nn.AvgPool2d(pool1_kernel, stride=pool1_stride),
            
            nn.Conv2d(conv1_out, conv2_out, kernel_size=conv2_kernel),
            nn.ReLU(),
            nn.AvgPool2d(pool2_kernel, stride=pool2_stride),
            
            nn.Conv2d(conv2_out, conv3_out, kernel_size=conv3_kernel),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()

        # Calc output shape of convolutions
        meta_tensor = torch.zeros(1, input_channels, input_size, input_size)
        meta_tensor = self.convolutions(meta_tensor)
        meta_tensor = self.flatten(meta_tensor)
        out_features = meta_tensor.shape[1]

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(out_features, fc_out_size),
            nn.ReLU(),
            nn.Linear(fc_out_size, num_classes)
        )

    def forward(self, x=None):
        x = self.convolutions(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def validate(self, val_loader, criterion, device):
        self.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        self.train()
        return val_loss / len(val_loader), 100. * correct / total

class ConvNN(f.Function):
    def __init__(self, n_obs=1, minimise=True, verbose=0, plot=False):
        super(f.Function, self).__init__()

        self.default_search_params = {
            'conv1_out'     : 6,    
            'conv1_kernel'  : 5,
            'pool1_kernel'  : 2,  
            'pool1_stride'  : 1,
            'conv2_out'     : 16,    
            'conv2_kernel'  : 5,
            'pool2_kernel'  : 2,  
            'pool2_stride'  : 2,
            'conv3_out'     : 120,    
            'conv3_kernel'  : 5,
            'fc_out_size'   : 84,
            'learning_rate' : 3e-4,
            'batch_size'    : 64,
            'optimizer_Adam': 1,  # ['Adam' -> 1, 'SGD' -> 0]
        }
        self.domain_unscaled = [
            [   1,   32],  # conv1_out
            [   2,   12],  # conv1_kernel
            [   2,    5],  # pool1_kernel
            [   1,    3],  # pool1_stride
            [  32,   64],  # conv2_out
            [   2,    8],  # conv2_kernel
            [   2,    5],  # pool2_kernel
            [   1,    3],  # pool2_stride
            [  64,  128],  # conv3_out
            [   2,    6],  # conv3_kernel
            [ 256, 1024],  # fc_out_size
            [1e-6, 1e-3],  # learning_rate
            [  32,  512],  # batch_size
            # [ 0.0,  1.0],  # optimizer_Adam
        ]
        self.dim           = len(self.domain_unscaled)
        self.domain        = np.tile([0., 1.], reps=(self.dim,1))
        self.plot          = plot
        self.name          = 'classifier'
        self.f             = lambda x: self.__call__(x)
        self.log_transform = False
        self.log_eps       = 1e-8
        self.sigma         = 1e-1
        self.n_obs         = n_obs
        self.minimise      = minimise
        self.seed          = 73
        self.verbose       = verbose

        # load dataset
        self.train_dataset, self.test_dataset = self.__load_cifar10(
            'benchmarks/conv_nn/cifar10_train_data',
            'benchmarks/conv_nn/cifar10_test_data'
        )

    def __call__(self, params: np.array): 
        NUM_EPOCHS = 10
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        results = []
        search_space = np.copy(params)

        # scaling params to actual values (in floats)
        for i in range(search_space.shape[1]):
            search_space[:, i] = qmc.scale(search_space[:, i].reshape(search_space.shape[0], 1), self.domain_unscaled[i][0], self.domain_unscaled[i][1]).squeeze()

        for point in search_space:
            # set configuration
            config = {
                'conv1_out'     : int(point[0]),    
                'conv1_kernel'  : int(point[1]),
                'pool1_kernel'  : int(point[2]),  
                'pool1_stride'  : int(point[3]),
                'conv2_out'     : int(point[4]),    
                'conv2_kernel'  : int(point[5]),
                'pool2_kernel'  : int(point[6]),  
                'pool2_stride'  : int(point[7]),
                'conv3_out'     : int(point[8]),    
                'conv3_kernel'  : int(point[9]),
                'fc_out_size'   : int(point[10]),
                'learning_rate' : point([11]),
                'batch_size'    : int(point[12]),
                # 'optimizer_Adam': int(point[13] >= 0.5),  # ['Adam' -> 1, 'SGD' -> 0]
            }

            if self.verbose > 0:
                print('Current parameters for training:')
                self.__print_dict(config)

            train_loader = DataLoader(self.train_dataset, batch_size=config['batch_size'], shuffle=True)
            test_loader = DataLoader(self.test_dataset, batch_size=config['batch_size'], shuffle=False)

            model = LeNet5(
                input_size=32,   
                input_channels=3,

                conv1_out=config['conv1_out'],    
                conv1_kernel=config['conv1_kernel'],

                pool1_kernel=config['pool1_kernel'],  
                pool1_stride=config['pool1_stride'],

                conv2_out=config['conv2_out'],    
                conv2_kernel=config['conv2_kernel'],

                pool2_kernel=config['pool2_kernel'],  
                pool2_stride=config['pool2_stride'],

                conv3_out=config['conv3_out'],    
                conv3_kernel=config['conv3_kernel'],

                fc_out_size=config['fc_out_size'],
            ).to(DEVICE)

            # ----- plot the model architecture -----
            if self.plot:
                timestamp = datetime.now().strftime("%d_%H_%M")
                filename = f"cifar10_model_graph_{timestamp}.png"

                model_graph = draw_graph(
                    LeNet5(),
                    input_size=(1, 3, 32, 32),
                    expand_nested=True,
                    save_graph=True,
                    filename=filename
                )

            # ----- model training -----
            # if config['optimizer_Adam'] == 1:
            #     optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            # else:
            #     optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            best_metric = 1e-2

            if self.verbose > 0:
                train_loop = tqdm(range(NUM_EPOCHS), desc='Model training')
            else:
                train_loop = range(NUM_EPOCHS)
            
            for epoch in train_loop:
                model.train()
                for images, labels in train_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                val_loss, accuracy = model.validate(test_loader, criterion, device=DEVICE)
                if accuracy > best_metric:
                    best_metric = accuracy
                
                if self.verbose > 0:
                    desc = f'Model training; Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}'
                    train_loop.set_description(desc)
            results.append(best_metric)
        results = np.array(results).reshape(-1, 1)

        if self.minimise:
            return -1.0 * results
        else:
            return results

    def sample(self, vectors_of_params):
        return vectors_of_params, self.__call__(vectors_of_params)

    def __print_dict(self, d): 
        print('\n'.join(f"{k:<{max(len(str(k)) for k in d)}} : {v}" for k, v in d.items()) if d else "empty")

    def __load_cifar10(self, train_path, test_path):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        ])
        
        train_dataset = CIFAR10(
            root=train_path,
            train=True,
            transform=transform,
            download=True
        )
        test_dataset = CIFAR10(
            root=test_path,
            train=False,
            transform=transform,
            download=True
        )

        return train_dataset, test_dataset
    