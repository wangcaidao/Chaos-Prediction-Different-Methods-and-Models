import scipy.io
import torch
import numpy as np
from torch.utils.data import Dataset

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        #self.data = mat73.loadmat(self.file_path)
        self.data = scipy.io.loadmat(self.file_path)
        self.old_mat = True

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class DefaultLoader_train(object):
    def __init__(self, datapath):
        dataloader = MatReader(datapath)
        self.x_data = dataloader.read_field('data_ic')
        self.y_data = dataloader.read_field('data_out')
        self.val_x_data = dataloader.read_field('val_ic')
        self.val_y_data = dataloader.read_field('val_out')

    def make_loader(self, n_sample, n_val, batch_size):
        Xs = self.x_data[:n_sample]
        ys = self.y_data[:n_sample]
        val_Xs = self.val_x_data[:n_val]
        val_ys = self.val_y_data[:n_val]

        def maybe_unsqueeze(data):  # to ensure the shape is (batch size, sequence_length, feature_dimension)
            if data.size(-1) > 10:  # feature_dimension in our experiment is 1/2/3, all <10
                return data.unsqueeze(-1)
            return data

        Xs = maybe_unsqueeze(Xs)
        ys = maybe_unsqueeze(ys)
        val_Xs = maybe_unsqueeze(val_Xs)
        val_ys = maybe_unsqueeze(val_ys)

        dataset = torch.utils.data.TensorDataset(Xs, ys)
        val_dataset = torch.utils.data.TensorDataset(val_Xs, val_ys)

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        return loader, val_loader


class DefaultLoader_test(object):
    def __init__(self, datapath):
        dataloader = MatReader(datapath)
        self.x_data = dataloader.read_field('test_ic')
        self.y_data = dataloader.read_field('test_out')

    def make_loader(self, n_sample, batch_size):
        Xs = self.x_data[:n_sample]
        ys = self.y_data[:n_sample]

        def maybe_unsqueeze(data):  # to ensure the shape is (batch size, sequence_length, feature_dimension)
            if data.size(-1) > 10:  # feature_dimension in our experiment is 1/2/3, all <10
                return data.unsqueeze(-1)
            return data

        Xs = maybe_unsqueeze(Xs)
        ys = maybe_unsqueeze(ys)

        dataset = torch.utils.data.TensorDataset(Xs, ys)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader
