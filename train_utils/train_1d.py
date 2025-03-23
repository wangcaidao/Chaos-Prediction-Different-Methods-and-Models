import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from .utils import save_checkpoint
from .losses import LpLoss


def train_default(model, train_loader, val_loader, optimizer, scheduler, config,
                  device=torch.device('cuda:0'), use_tqdm=True):
    data_weight = config['train']['xy_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    data_loss_history = []
    val_loss_history = []
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    for e in pbar:
        model.train()
        data_l2 = 0.0
        train_loss = 0.0
        val_loss = 0.0
        for data_ic, data_out in train_loader:
            data_ic, data_out = data_ic.to(device), data_out.to(device)
            optimizer.zero_grad()
            pred = model(data_ic).reshape(data_out.shape)
            data_loss = myloss(pred, data_out)
            total_loss = data_loss * data_weight
            total_loss.backward()
            optimizer.step()
            data_l2 += data_loss.item()
            train_loss += total_loss.item()

        for val_ic, val_out in val_loader:
            val_ic, val_out = val_ic.to(device), val_out.to(device)
            pred_out = model(val_ic).reshape(val_out.shape)
            loss_val = myloss(pred_out, val_out)
            val_loss += loss_val.item()

        scheduler.step()

        data_l2 /= len(train_loader)
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        data_loss_history.append(data_l2)
        val_loss_history.append(val_loss)

        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch {e}, total loss: {train_loss:.5f} '
                    f'data loss: {data_l2:.5f} '
                    f'val loss: {val_loss:.5f} '
                )
            )

    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)

    plt.figure()
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.title('Validation Loss over Epochs')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(data_loss_history, label='Data Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Data Loss')
    plt.title('Data Loss over Epochs')
    plt.legend()
    plt.show()
    print('Done!')
