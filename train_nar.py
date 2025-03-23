from argparse import ArgumentParser
import yaml

import torch
from models import FNO_nar, LSTM_nar, MLP_nar
from train_utils import Adam
from train_utils.datasets import DefaultLoader_test, DefaultLoader_train
from train_utils.train_1d import train_default
from train_utils.eval_1d import eval_nar


def run(args, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    dataset = DefaultLoader_train(data_config['datapath'])
    train_loader, val_loader = dataset.make_loader(n_sample=data_config['n_sample'], n_val=data_config['n_val'],
                                                   batch_size=config['train']['batchsize'])
    model = LSTM_nar(modes=config['model']['modes1'], fc_dim=config['model']['fc_dim'], layers=config['model']['layers'],
                     in_dim=1, out_dim=1, length=data_config['length'], act=config['model']['act']).to(device)
    def count_parameters(model):
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                print(f'Layer {name} | Parameters: {param_count}')
                total_params += param_count
        print(f'Total Trainable Parameters: {total_params}')
    count_parameters(model)

    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    train_default(model, train_loader, val_loader, optimizer, scheduler, config, device)


def test(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    dataset = DefaultLoader_test(data_config['datapath'])
    dataloader = dataset.make_loader(n_sample=data_config['n_sample'], batch_size=config['test']['batchsize'])
    model = LSTM_nar(modes=config['model']['modes1'], fc_dim=config['model']['fc_dim'], layers=config['model']['layers'],
                     in_dim=1, out_dim=1, length=data_config['length'], act=config['model']['act']).to(device)
    # Load from checkpoint
    if 'ckpt' in config['test']:
        ckpt_path = config['test']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    eval_nar(model, dataloader, config, device)

# i = 1 for training,  i = 2 for testing, test results will be saved as .mat file
if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    i = 1
    if i == 1:
        parser.add_argument('--config_path', type=str, default='configs/NAR/train.yaml', help='Path to the configuration file')
        parser.add_argument('--mode', type=str, default='train', help='train or test')
    if i == 2:
        parser.add_argument('--config_path', type=str, default='configs/NAR/test.yaml', help='Path to the configuration file')
        parser.add_argument('--mode', type=str, default='test', help='train or test')

    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    args = parser.parse_args()
    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.mode == 'train':
        run(args, config)
    else:
        test(config)
