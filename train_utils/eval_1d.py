from tqdm import tqdm
import numpy as np
import torch
import math
import scipy.io


def eval_nar(model, dataloader, device, use_tqdm=True):
    device = torch.device('cuda')
    model.eval()
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader
    pred = []
    correct = []
    with torch.no_grad():
        for index, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            seq_length_ic = x.shape[1]  # Length of the input sequence
            seq_length_out = y.shape[1]  # Length of the output sequence
            data_seq = x.clone()  # Start with data_ic as the initial sequence
            print(data_seq.shape)
            for t in range(seq_length_out):  # Loop over the length of the output sequence (data_out.shape[1])
                input_seq = data_seq[:, -seq_length_ic:, :]
                predicted_value = model(input_seq)
                data_seq = torch.cat((data_seq, predicted_value), dim=1)  # Append the predicted value to data_seq
                if t % 200 == 0:
                    print(t)

            # Store predictions and ground truth
            out = data_seq[:, seq_length_ic:seq_length_ic + seq_length_out, :]
            pred.append(out.cpu().detach().numpy())
            correct.append(y.cpu().detach().numpy())

    pred = np.concatenate(pred, axis=0)
    correct = np.concatenate(correct, axis=0)
    scipy.io.savemat('pred/MG-NAR-short-MLP.mat', mdict={'pred': pred, 'correct': correct})


def eval_s2s_1p_iteration(model, dataloader, device, use_tqdm=True):
    device = torch.device('cuda')
    model.eval()
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader
    pred = []
    correct = []
    with torch.no_grad():
        for index, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            seq_length_ic = x.shape[1]  # Length of the input sequence
            seq_length_out = y.shape[1]  # Length of the output sequence
            data_seq = x.clone()  # Start with data_ic as the initial sequence
            print(data_seq.shape)
            for t in range(seq_length_out):  # Loop over the length of the output sequence (data_out.shape[1])
                input_seq = data_seq[:, -seq_length_ic:, :]  # Extract the last seq_length_ic time steps from data_seq
                predicted_value = model(input_seq)[:, -1:, :]
                data_seq = torch.cat((data_seq, predicted_value), dim=1)  # Append the predicted value to data_seq
                if t % 200 == 0:
                    print(t)
            out = data_seq[:, seq_length_ic:seq_length_ic + seq_length_out, :]
            pred.append(out.cpu().detach().numpy())
            correct.append(y.cpu().detach().numpy())

    # Convert the list of arrays to numpy arrays for final output
    pred = np.concatenate(pred, axis=0)
    correct = np.concatenate(correct, axis=0)
    scipy.io.savemat('pred/MG-S2S1P-short-MLP.mat', mdict={'pred': pred, 'correct': correct})


def eval_p2p_iteration(model, dataloader, device, use_tqdm=True, nt=1000):
    device = torch.device('cuda')
    model.eval()
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader
    pred = []
    correct = []
    with torch.no_grad():
        for index, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            out = torch.zeros(y.shape).to(device)  # Prepare to store predictions for n_sample steps
            predicted_value = x.clone()  # Start with the initial input x as the sequence
            # Generate predictions using the model iteratively
            for t in range(nt):
                predicted_value = model(predicted_value)  # Model outputs a single predicted value
                out[:, t, :] = predicted_value  # Store the predicted value in the output
                predicted_value = predicted_value.detach()
                if t % 200 == 0:
                    print(t)
            pred.append(out.cpu().detach().numpy())
            correct.append(y.cpu().detach().numpy())
    # Convert the list of arrays to numpy arrays for final output
    pred = np.concatenate(pred, axis=0)
    correct = np.concatenate(correct, axis=0)

    # Save the predictions and real values as a .mat file
    scipy.io.savemat('pred/Rossler-P2P-short-MLP.mat', mdict={'pred': pred, 'correct': correct})


def eval_p2p_iteration_timedelay(model, dataloader, device, use_tqdm=True, nt=10000):
    device = torch.device('cuda')
    model.eval()
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader
    pred = []
    correct = []
    tau = 100   # delay time steps
    with torch.no_grad():
        for index, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            data_seq = torch.zeros(y.shape[0], y.shape[1] + tau, y.shape[2]).to(device)
            data_seq[:, :tau, :] = x

            # Generate predictions using the model iteratively
            for t in range(nt):
                input = torch.cat((data_seq[:, tau + t - 1, :], data_seq[:, t, :]), dim=-1)
                predicted_value = model(input)  # Model outputs a single predicted value
                data_seq[:, tau + t, :] = predicted_value  # Store the predicted value in the output
                if t % 200 == 0:
                    print(t)
            out = data_seq[:, -y.shape[1]:, :]
            pred.append(out.cpu().detach().numpy())
            correct.append(y.cpu().detach().numpy())

    # Convert the list of arrays to numpy arrays for final output
    pred = np.concatenate(pred, axis=0)
    correct = np.concatenate(correct, axis=0)
    # Save the predictions and real values as a .mat file
    scipy.io.savemat('pred/MG-P2P-short-MLP.mat', mdict={'pred': pred, 'correct': correct})


def eval_s2s_td_iteration(model, dataloader, device, use_tqdm=True):
    device = torch.device('cuda')
    model.eval()
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader
    pred = []
    correct = []
    tau = 100   # delay time steps
    with torch.no_grad():
        for index, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            seq_length_ic = x.shape[1]  # Length of the input sequence
            seq_length_out = y.shape[1]  # Length of the output sequence
            data_seq = x.clone()  # Start with data_ic as the initial sequence
            print(data_seq.shape)
            for t in range(math.ceil(seq_length_out / tau)):  # Loop over the length of the output sequence (data_out.shape[1])
                input_seq = data_seq[:, -seq_length_ic:]  # Extract the last seq_length_ic time steps from data_seq
                predicted_value = model(input_seq)[:, -tau:]
                print(predicted_value.shape)
                data_seq = torch.cat((data_seq, predicted_value), dim=1)  # Append the predicted value to data_seq
                print(t)

            out = data_seq[:, seq_length_ic:seq_length_ic + seq_length_out]
            pred.append(out.cpu().detach().numpy())
            correct.append(y.cpu().detach().numpy())

    # Convert the list of arrays to numpy arrays for final output
    pred = np.concatenate(pred, axis=0)
    correct = np.concatenate(correct, axis=0)
    # Save the predictions and real values as a .mat file
    scipy.io.savemat('pred/MG-S2STD-long-MLP.mat', mdict={'pred': pred, 'correct': correct})


def eval_onestep(model, dataloader, device, use_tqdm=True):
    device = torch.device('cuda')
    model.eval()
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader
    pred = []
    correct = []

    for index, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        out = model(x)

        # Collect original outputs
        pred.append(out.squeeze(-1).cpu().detach().numpy())
        correct.append(y.squeeze(-1).cpu().detach().numpy())

    # Convert original and flattened lists to numpy arrays for final output
    pred = np.concatenate(pred, axis=0)
    correct = np.concatenate(correct, axis=0)
    # Save the original and flattened data as a .mat file
    scipy.io.savemat('pred/MG-onestep-MLP.mat', mdict={'pred': pred, 'correct': correct})
