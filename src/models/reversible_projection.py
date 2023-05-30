import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    多层感知机
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.layers.insert(0, nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        return self.layers[-1](x)


class ReversibleLayer(nn.Module):
    """
    可逆层，由两个MLP组成
    """
    def __init__(self, f):
        super().__init__()
        self.f = f
        
    def forward(self, x):
        x1, x2 = torch.split(x, x.size(1) // 2, dim=1)
        y2 = x2 + self.f(x1)
        y1 = x1 + y2
        return torch.cat([y1, y2], dim=1)
    
    def backward(self, y):
        y1, y2 = torch.split(y, y.size(1) // 2, dim=1)
        x2 = y2 - self.f(y1)
        x1 = y1 - x2
        return torch.cat([x1, x2], dim=1)


class ReversibleNet(nn.Module):
    """
    可逆神经网络，由多个可逆层和全连接层组成。
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([ReversibleLayer(MLP(hidden_dim, hidden_dim, hidden_dim, num_layers)) for _ in range(num_layers)])
        self.output_layer = MLP(hidden_dim, output_dim, hidden_dim, num_layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)
    
    def backward(self, y):
        x = self.output_layer.backward(y)
        for layer in reversed(self.layers):
            x = layer.backward(x)
        return x


# 测试可逆神经网络
if __name__== '__main__':
    x = torch.randn(3, 6)
    net = ReversibleNet(6, 8)
    y = net(x)
    x_ = net.backward(y)
    print(x)
    print(x_)
    print(torch.allclose(x, x_))