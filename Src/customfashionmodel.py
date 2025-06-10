import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from torch.utils.data import DataLoader
import torch

class CustomFashionModel(nn.Module):
    def __init__(self, use_bn: bool = False) -> None: 
        super().__init__()
        self.use_bn = False  
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> Tuple[float, float]:
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        return avg_loss, accuracy


    def get_model_parameters(self) -> List[np.ndarray]:
        return [param.cpu().detach().numpy() for param in self.parameters()]

    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.load_state_dict(state_dict)

    def train_epoch_prox(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        mu: float,
        global_params: List[np.ndarray]
    ) -> Tuple[float, float]:
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Convert global params to tensor
        global_tensors = [torch.tensor(p).to(device) for p in global_params]
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self(data)
            
            # Standard loss
            standard_loss = criterion(output, target)
            
            # Proximal term (L2 norm squared)
            proximal_term = 0.0
            for local_param, global_param in zip(self.parameters(), global_tensors):
                proximal_term += (local_param - global_param).norm(2)**2
            
            # Total loss
            loss = standard_loss + (mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
            
            total_loss += standard_loss.item()  # Track standard loss for metrics
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        return avg_loss, accuracy
        
    def train_epoch_scaffold(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        global_control: List[np.ndarray],
        local_control: List[np.ndarray],
        lr: float
    ) -> Tuple[float, float, List[np.ndarray]]:
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Convert controls to tensors
        global_control_tensors = [torch.tensor(c, device=device, dtype=torch.float32) for c in global_control]
        local_control_tensors = [torch.tensor(c, device=device, dtype=torch.float32) for c in local_control]
        
        # Store initial parameters
        initial_params = [param.data.clone() for param in self.parameters()]
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # SCAFFOLD correction term
            correction = 0.0
            for param, c_global, c_local in zip(self.parameters(), global_control_tensors, local_control_tensors):
                if param.grad is not None:
                    correction += (c_local - c_global).dot(param.grad.view(-1))
            
            total_loss += loss.item()
            loss = loss - correction
            
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        # Calculate updated local control
        updated_local_control = []
        for param, init_param, c_global in zip(self.parameters(), initial_params, global_control_tensors):
            delta_param = (init_param - param.data)  # Note the sign change
            updated_control = c_global - delta_param / (lr * len(train_loader))
            updated_local_control.append(updated_control.detach().cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy, updated_local_control
    def test_epoch(
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
    ) -> Tuple[float, float]:
        """Evaluate the model on a test dataset."""
        self.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # Disable gradient calculation
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        return avg_loss, accuracy