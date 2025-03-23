import torch.nn as nn
from .basics import SpectralConv1d
from .utils import _get_act


class MLP_P2P(nn.Module):
    def __init__(self, layers=None, in_dim=3, out_dim=3, act='relu'):
        super(MLP_P2P, self).__init__()

        # Create the first layer (input to the first hidden layer)
        self.p = nn.Linear(in_dim, layers[0])

        # Create the hidden layers (from layers[i] to layers[i+1])
        self.fcs = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fcs.append(nn.Linear(layers[i], layers[i + 1]))

        # Create the output layer (from the last hidden layer to out_dim)
        self.q = nn.Linear(layers[-1], out_dim)

        # Activation function
        self.act = _get_act(act)

    def forward(self, x):
        # First layer
        x = self.p(x)

        # Hidden layers (dynamic number of layers)
        for fc in self.fcs:
            x = fc(x)
            x = self.act(x)

        # Output layer
        x = self.q(x)

        return x


class MLP_1d(nn.Module):
    def __init__(self, modes, fc_dim=128, layers=None, in_dim=3, out_dim=3, length=100, act='relu'):
        super(MLP_1d, self).__init__()
        linear_width = length * 2
        linear_layers = [linear_width, linear_width, linear_width, linear_width, linear_width]
        self.p1 = nn.Linear(in_dim, layers[0])
        self.p2 = nn.Linear(length, linear_layers[0])
        self.linears = nn.ModuleList([nn.Linear(in_size, out_size)
                                      for in_size, out_size in zip(linear_layers, linear_layers[1:])])

        self.q1 = nn.Linear(layers[-1], out_dim)
        self.q2 = nn.Linear(linear_layers[-1], length)
        self.act = _get_act(act)

    def forward(self, x):
        x = self.p1(x)
        x = x.permute(0, 2, 1)
        x = self.p2(x)

        for linear in self.linears:
            x = linear(x)
            x = self.act(x)

        x = self.q2(x)
        x = x.permute(0, 2, 1)
        x = self.q1(x)

        return x


class MLP_nar(nn.Module):
    def __init__(self, modes, fc_dim=128, layers=None, in_dim=3, out_dim=3, length=100, act='relu'):
        super(MLP_nar, self).__init__()
        linear_width = length * 2
        linear_layers = [linear_width, linear_width, linear_width, linear_width, linear_width]
        self.p1 = nn.Linear(in_dim, layers[0])
        self.p2 = nn.Linear(length, linear_layers[0])
        self.linears = nn.ModuleList([nn.Linear(in_size, out_size)
                                      for in_size, out_size in zip(linear_layers, linear_layers[1:])])

        # Create the output layer (from the last hidden layer to out_dim)
        self.q = nn.Linear(layers[-1], out_dim)
        self.r = nn.Linear(linear_layers[-1], 1)

        # Activation function
        self.act = _get_act(act)

    def forward(self, x):
        # First layer
        x = self.p1(x)
        x = x.permute(0, 2, 1)
        x = self.p2(x)

        # Hidden layers (dynamic number of layers)
        for linear in self.linears:
            x = linear(x)
            x = self.act(x)

        # Output layer
        x = self.r(x)
        x = x.permute(0, 2, 1)
        x = self.q(x)

        return x


class LSTM_1d(nn.Module):
    def __init__(self, modes, width=32, layers=None, fc_dim=128, in_dim=1, out_dim=1, length=100, act='relu'):
        super(LSTM_1d, self).__init__()
        print(layers[0])
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=layers[0], num_layers=1, batch_first=True)
        self.q = nn.Linear(in_features=layers[0], out_features=out_dim)  # output channel_dim is 1: u1(x)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.q(x)

        return x


class LSTM_nar(nn.Module):
    def __init__(self, modes, fc_dim=128, layers=None, in_dim=1, out_dim=1, length=100, act='relu'):
        super(LSTM_nar, self).__init__()
        print(layers[0])
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=layers[0], num_layers=1, batch_first=True)
        self.q = nn.Linear(in_features=layers[0], out_features=out_dim)
        self.r = nn.Linear(in_features=length, out_features=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.r(x)
        x = x.permute(0, 2, 1)
        x = self.q(x)

        return x


class FNO_1d(nn.Module):
    def __init__(self, modes, width=32, layers=None, fc_dim=128, in_dim=1, out_dim=1, length=100, act='relu'):
        super(FNO_1d, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=seq_length, c=feature_dim)
        output: the solution of a later timestep
        output shape: (batchsize, x=seq_length, c=feature_dim)
        """

        self.modes1 = modes
        self.width = width
        if layers is None:
            layers = [width] * 4
        self.fc0 = nn.Linear(in_dim, layers[0])  # input channel is 2: (a(x), x)
        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, num_modes) for in_size, out_size, num_modes in zip(layers, layers[1:], self.modes1)])
        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])
        linear_layers = [length, length, length, length, length]
        self.linears = nn.ModuleList([nn.Linear(in_size, out_size)
                                      for in_size, out_size in zip(linear_layers, linear_layers[1:])])
        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.act = _get_act(act)

    def forward(self, x):
        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w, linear) in enumerate(zip(self.sp_convs, self.ws, self.linears)):
            x1 = speconv(x)
            x2 = w(x)
            x3 = linear(x)
            x = x1 + x2 + x3
            if i != length - 1:
                x = self.act(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class FNO_nar(nn.Module):
    def __init__(self,
                 modes, width=32,
                 layers=None,
                 fc_dim=128,
                 in_dim=1, out_dim=1, length=1000,
                 act='relu'):
        super(FNO_nar, self).__init__()
        self.modes1 = modes
        self.width = width
        if layers is None:
            layers = [width] * 4
        self.fc0 = nn.Linear(in_dim, layers[0])  # input channel is 2: (a(x), x)
        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, num_modes) for in_size, out_size, num_modes in zip(layers, layers[1:], self.modes1)])
        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])
        linear_layers = [length, length, length, length, length]
        self.linears = nn.ModuleList([nn.Linear(in_size, out_size)
                                      for in_size, out_size in zip(linear_layers, linear_layers[1:])])
        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.fc3 = nn.Linear(length, 1)
        self.act = _get_act(act)

    def forward(self, x):
        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w, linear) in enumerate(zip(self.sp_convs, self.ws, self.linears)):
            x1 = speconv(x)
            x2 = w(x)
            x3 = linear(x)
            x = x1 + x2 + x3
            if i != length - 1:
                x = self.act(x)

        x = self.fc3(x)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
