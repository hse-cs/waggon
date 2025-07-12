import os
import tempfile
import pickle

from waggon import functions as f

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
from torchview import draw_graph # For visualizing model architecture

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import distance
from datetime import datetime


def print_dict(d): print('\n'.join(f"{k:<{max(len(str(k)) for k in d)}} : {v}" for k, v in d.items()) if d else "empty")

def objective_CIFAR_10(param, logging=False, ray_tune=False):
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

    train_dataset = CIFAR10Dataset(
        root_dir="extracted_data/cifar-10-batches-py",
        train=True,
        transform=transform
    )

    test_dataset = CIFAR10Dataset(
        root_dir="extracted_data/cifar-10-batches-py",
        train=False,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = CIFAR10Model(
        input_size=32,
        input_channels=3,
        conv1_out=param[0],
        conv1_kernel=param[1],
        pool1_kernel=param[2],
        conv2_out=param[3],
        pool2_kernel=param[4],
        fc1_size=param[5],
        num_classes=10
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=param[6])
    best_metric = 1e-2
    
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_loss, accuracy = model.validate(test_loader, criterion, DEVICE)

        if ray_tune:
            with tempfile.TemporaryDirectory() as checkpoint_dir:
            # (Path(checkpoint_dir) / "data.ckpt").write_text(str(step)) 
                tune.report(
                    {"iterations": step, "accuracy": score},
                    checkpoint=tune.Checkpoint.from_directory(checkpoint_dir)
                )

        if accuracy > best_metric:
            best_metric = accuracy
        if logging:
            print(f'Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')


    # if self.minimise:
    #     return -1.0 * best_metric
    # else:
    return best_metric

class CIFAR10Model(nn.Module):
    def __init__(
        self,
        input_size=32,   
        input_channels=3,
        conv1_out=32,    
        conv1_kernel=3,  
        padding_1=1,
        pool1_kernel=2,  
        pool1_stride=2,  
        conv2_out=64,    
        conv2_kernel=3,  
        padding_2=1, 
        pool2_kernel=2,  
        pool2_stride=2,

        batch_size=64, 
        apply_avg_pooling=0,

        fc1_size=512,
        negative_slope=0.15,
        dropout_size=0.1,
        num_classes=10   
    ):
        super().__init__()
        
        self.convolutions = nn.Sequential(
            nn.Conv2d(input_channels, conv1_out, kernel_size=conv1_kernel, padding=padding_1),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.BatchNorm2d(conv1_out),
            nn.Dropout(dropout_size),
            nn.MaxPool2d(pool1_kernel, stride=pool1_stride),
            nn.Conv2d(conv1_out, conv2_out, kernel_size=conv2_kernel, padding=padding_2),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.BatchNorm2d(conv2_out)
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d((1,1)) if apply_avg_pooling == 1 else nn.Identity()
        self.flatten = nn.Flatten()
    
        meta_tensor = torch.zeros(1, input_channels, input_size, input_size)
        meta_tensor = self.convolutions(meta_tensor)
        meta_tensor = self.avg_pooling(meta_tensor)
        meta_tensor = self.flatten(meta_tensor)
        out_features = meta_tensor.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(out_features, fc1_size),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(fc1_size, num_classes)
        )

    def forward(self, x=None):
        x = self.convolutions(x)
        # x = x.view(x.size(0), -1)

        x = self.avg_pooling(x)
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
    def __init__(self, n_obs=1, model=1, minimise=True, logging=False, ray_tune=False, plot=False):
        super(f.Function, self).__init__()

        self.search_params = {
            'conv_1_size': 32,
            'conv_1_kernel': 5,
            'padding_1': 4,
            'maxpool_size': 2,
            'conv2_size': 16,
            'padding_2': 3,
            'conv_2_kernel': 3,
            'fc_layer': 125,
            'learning_rate': 3e-4,
            'batch_size': 64,
            'negative_slope': 0.25,
            'dropout_prob': 0.25,
            'optimizer_Adam': 1, # ['Adam' -> 1, 'SGD' -> 0]
            'apply_avg_pooling': 1 # ['Apply' -> 1, 'use fc instead' -> 0]
        }
        self.domain_unscaled = [
            [   1,  128],  # conv_1_size
            [   1,   18],  # conv_1_kernel
            [   0,   20],  # padding_1
            [   1,    3],  # maxpool_size
            [   1,  128],  # conv2_size
            [   0,   10],  # padding_2
            [   1,    5],  # conv_2_kernel
            [  10, 1024],  # fc_layer
            [1e-6, 1e-2],  # learning_rate
            [   4,  512],  # batch_size
            # [ 0.0, 0.99],  # negative_slope
            # [ 0.0,  0.5],  # dropout_prob
            # [ 0.0,  1.0],  # optimizer_Adam
            # [ 0.0,  1.0],  # apply_avg_pooling
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
        self.model         = model
        self.minimise      = minimise
        self.seed          = 73
        self.ray_tune      = ray_tune
        self.logging       = logging
        self.f_min         = 0  # None
        self.glob_min	   = np.zeros(self.dim).reshape(1, -1)  # None

    def __call__(self, params : np.array): 
        NUM_EPOCHS = 10
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        ])

        # ----- dataset creation -----
        train_dataset = CIFAR10(
            root='./cifar10_train_data',
            train=True,
            transform=transform,
            download=True
        )
        test_dataset = CIFAR10(
            root='./cifar10_test_data',
            train=False,
            transform=transform,
            download=True
        )

        results = []
        search_space = np.copy(params)
        for param in search_space:
            # scaling params to actual values
            for domain_idx in range(len(self.domain)):
                if type(self.domain_unscaled[domain_idx][1]) == int:
                    # print(f'{param[domain_idx]} , {self.domain_unscaled[domain_idx][1]} and {self.domain_unscaled[domain_idx][0]}')
                    param[domain_idx] = int(param[domain_idx] * (self.domain_unscaled[domain_idx][1] - self.domain_unscaled[domain_idx][0]) + self.domain_unscaled[domain_idx][0])
                else:
                    # print(f'{param[domain_idx]} , {self.domain_unscaled[domain_idx][1]} and {self.domain_unscaled[domain_idx][0]}')
                    param[domain_idx] = param[domain_idx] * (self.domain_unscaled[domain_idx][1] - self.domain_unscaled[domain_idx][0]) + self.domain_unscaled[domain_idx][0]

            config = {
                'conv_1_size': int(param[0]),
                'conv_1_kernel': int(param[1]),
                'padding_1': int(param[2]),
                'maxpool_size': int(param[3]),
                'conv2_size': int(param[4]),
                'padding_2': int(param[5]),
                'conv_2_kernel': int(param[6]),
                'fc_layer': int(param[7]),
                'learning_rate': param[8],
                'batch_size': int(param[9]),
                # 'negative_slope': param[10],
                # 'dropout_prob': param[11],
                # 'optimizer_Adam': int(param[12] >= 0.5),  # ['Adam' -> 1, 'SGD' -> 0],
                # 'apply_avg_pooling': int(param[13] >= 0.5)  # ['Apply polling' -> 1, 'Use FC' -> 0]
            }

            if self.logging:
                print_dict(config)

            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

            model = CIFAR10Model(
                input_size=32,
                input_channels=3,

                conv1_out=config['conv_1_size'],
                conv1_kernel=config['conv_1_kernel'],
                padding_1=config['padding_1'],

                pool1_kernel=config['maxpool_size'],
                batch_size=config['batch_size'],
                
                conv2_out=config['conv2_size'],
                conv2_kernel=config['conv_2_kernel'],
                padding_2=config['padding_2'],

                fc1_size=config['fc_layer'],
                # negative_slope=config['negative_slope'],
                # dropout_size=config['dropout_prob'],

                # apply_avg_pooling=config['apply_avg_pooling'],

                num_classes=10
            ).to(DEVICE)

            # ----- plot the model architecture -----
            if self.plot:
                timestamp = datetime.now().strftime("%d_%H_%M")
                filename = f"cifar10_model_graph_{timestamp}.png"

                model_graph = draw_graph(
                    CIFAR10Model(),
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

            for epoch in tqdm(range(NUM_EPOCHS), desc=f'Training'):
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
                
                if self.logging:
                    print(f'Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
            results.append(best_metric)

        results = np.array(results).reshape(-1, 1)

        if self.minimise:
            return -1.0 * results
        else:
            return results

    def sample(self, vectors_of_params):
        # vectors_of_params = vectors_of_params[0]

        return vectors_of_params, self.__call__(vectors_of_params)

        
        # X, y = None, None
        # for n in vectors_of_params:
        #     X_ = np.array(self.n_obs * [n])
        #     n = torch.tensor(n)

        #     x = torch.normal(torch.mean(n), torch.std(n), (self.n_obs, n.shape[-1]))
        #     proba = self.model(x).detach().cpu().numpy()
        #     y_ = self.f([n]) + proba

        #     if X is None:
        #         X, y = X_, y_
        #     else:
        #         X = np.concatenate((X, X_))
        #         y = np.concatenate((y, y_))

        # for n in vectors_of_params:
        #     X_ = np.array(self.n_obs * [n])
        #     n = torch.tensor(n)

        #     x = torch.normal(torch.mean(n), torch.std(n), (self.n_obs, n.shape[-1]))
        #     proba = self.model(x).detach().cpu().numpy()
        #     y_ = self.f([n]) + proba

        #     if X is None:
        #         X, y = X_, y_
        #     else:
        #         X = np.concatenate((X, X_))
        #         y = np.concatenate((y, y_))

        # return X, y

