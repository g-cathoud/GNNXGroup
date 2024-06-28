import os
import time
import torch
import wandb
import pickle
import random

import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch import nn


def progress_bar_file(filename, epoch, total_epochs, freq=1, time=0, tmse=0, vmse=0, length=20):
    """
    Just for tracking the progress of the training.
    """
    total_time = time*(total_epochs - epoch)/freq
    percent = (epoch / total_epochs)
    filled_length = int(length * percent)
    bar = '█' * filled_length + '-' * (length - filled_length)
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(
            f"|{bar}| {percent*100:.2f}% Train MSE: {tmse:.2e} / Val MSE: {vmse:.2e} - Remaining time: {total_time:.0f} s\n")


def progress_bar_terminal(epoch, total_epochs, batch, total_batches, freq, mse=0, length=20):
    """
    Just for tracking the progress of the training.
    """
    percent1 = (epoch / total_epochs)
    filled_length1 = int(length * percent1)
    bar1 = '█' * filled_length1 + '-' * (length - filled_length1)
    percent2 = (batch / total_batches)
    filled_length2 = int(length * percent2)
    bar2 = '█' * filled_length2 + '-' * (length - filled_length2)
    print(f"\r|{bar1}| {percent1*100:.2f}% |{bar2}| {percent2*100:.2f}% Train MSE: {mse:.2e}", end="")


def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)


def train_model(args, model, prepare_input_function, optimizer, lr_scheduler, train_loader, val_loader, median=0, mad=1):

    epoch_loop_start_time = time.time()
    epochs_start_time = time.time()

    epochs_no_improve = 0
    loss_l1 = nn.L1Loss()
    best_val_loss = float('inf')
    early_stopping_patience = args['patience']

    scaling_params = {
            'median_train': median,
            'mad_train': mad,
        }

    for epoch_idx in range(args['epochs'] + 1):

        model.train()
        total_loss = 0.0

        for batch_idx, data in enumerate(train_loader):

            # Transfer data to the appropriate device
            y = data[args['target']].to(args['device'], args['dtype'])

            # Zero the gradients before running the backward pass
            optimizer.zero_grad()

            # Forward pass: compute the model output
            input = prepare_input_function(data, args)

            pred, _ = model(input)

            loss = loss_l1(pred, (y - median) / mad)
            loss.backward()

            if args['log_gradients']:  # Log gradients
                grad_norms = {}
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        grad_norm = parameter.grad.data.norm(
                            2).item()  # Calculate the L2 norm
                        grad_norms[f'grad_norm/{name}'] = grad_norm
                wandb.log(grad_norms, step=epoch_idx)

            optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()

            progress_bar_terminal(epoch_idx, args['epochs'], batch_idx, len(
                train_loader), args['log_freq'], loss.item())

        # Learning rate scheduler step (if epoch-wise stepping is intended)
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Calculate average loss over the batches
        avg_loss = total_loss / len(train_loader)
        wandb.log({f"{args['target']}/Train_loss": avg_loss}, step=epoch_idx)

        # Logging
        if epoch_idx % args['log_freq'] == 0 or epoch_idx == args['epochs']:

            val_loss = validate_model(
                args, model, prepare_input_function, val_loader, median, mad)
            wandb.log({f"{args['target']}/Val_loss": val_loss}, step=epoch_idx)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'scaling_params': scaling_params
                }, os.path.join(f"./models/Z_trained_models/{args['run_tag']}.pth"))

            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    with open(args['report_file'], 'a', encoding='utf-8') as file:
                        file.write(
                            f"Early stopping triggered at epoch {epoch_idx}\n")
                    break

            if val_loss < 0.95*best_val_loss:
                epochs_no_improve = 0

            epochs_end_time = time.time()
            epochs_total_time = epochs_end_time - epochs_start_time

            progress_bar_file(args['report_file'], epoch_idx, args['epochs'],
                              args['log_freq'], epochs_total_time, avg_loss, val_loss)

            epochs_start_time = time.time()

    epoch_loop_end_time = time.time()
    epoch_loop_total_time = epoch_loop_end_time - epoch_loop_start_time

    with open(args['report_file'], 'a', encoding='utf-8') as file:
        file.write(
            f"++ the time for the epochs loop was {epoch_loop_total_time:.0f} s\n")

    torch.save({'epoch': epoch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'scaling_params': scaling_params},
               f"./models/Z_trained_models/{args['run_tag']}.pth")


def train_model_custom_loss(args, model, prepare_input_function, optimizer, lr_scheduler, train_loader, val_loader, alpha_scheduler, median=0, mad=1):

    epoch_loop_start_time = time.time()
    epochs_start_time = time.time()

    epochs_no_improve = 0
    loss_l1 = nn.L1Loss()
    best_val_loss = float('inf')
    early_stopping_patience = args['patience']

    scaling_params = {
            'median_train': median,
            'mad_train': mad,
        }

    for epoch_idx in range(args['epochs'] + 1):

        model.train()
        total_loss = 0.0

        for batch_idx, data in enumerate(train_loader):

            # Transfer data to the appropriate device
            y = data[args['target']].to(args['device'], args['dtype'])
            true_contributions = data[f"{args['target']}_benson_coefs"].to(
                args['device'], args['dtype'])

            # Zero the gradients before running the backward pass
            optimizer.zero_grad()

            input = prepare_input_function(data, args)

            # Forward pass: compute the model output
            pred, contributions = model(input)

            # Calculate prediction loss
            pred_loss = loss_l1(pred, (y - median) / mad)

            # Calculate node contributions loss
            contributions_loss = loss_l1(contributions, true_contributions)

            # Combine the losses
            alpha = alpha_scheduler.get_alpha()
            loss = (1 - alpha) * pred_loss + alpha * contributions_loss

            # Backpropagation
            loss.backward()
            if args['log_gradients']:
                # Log gradients
                grad_norms = {}
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        grad_norm = parameter.grad.data.norm(
                            2).item()  # Calculate the L2 norm
                        grad_norms[f'grad_norm/{name}'] = grad_norm
                wandb.log(grad_norms, step=epoch_idx)
            optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()

            progress_bar_terminal(epoch_idx, args['epochs'], batch_idx, len(
                train_loader), args['log_freq'], loss.item())

        # Learning rate and alpha scheduler step
        if lr_scheduler is not None:
            lr_scheduler.step()
        if alpha_scheduler is not None:
            alpha_scheduler.step()

        # Calculate average loss over the batches
        avg_loss = total_loss / len(train_loader)
        wandb.log({f"{args['target']}/Train_loss": avg_loss}, step=epoch_idx)

        # Logging
        if epoch_idx % args['log_freq'] == 0 or epoch_idx == args['epochs']:

            val_loss = validate_model_custom_loss(
                args, model, prepare_input_function, val_loader, alpha, median, mad)
            wandb.log({f"{args['target']}/Val_loss": val_loss}, step=epoch_idx)

            epochs_end_time = time.time()
            epochs_total_time = epochs_end_time - epochs_start_time

            progress_bar_file(args['report_file'], epoch_idx, args['epochs'],
                              args['log_freq'], epochs_total_time, avg_loss, val_loss)

            epochs_start_time = time.time()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'scaling_params': scaling_params,
                }, os.path.join(f"./models/Z_trained_models/{args['run_tag']}.pth"))

            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    with open(args['report_file'], 'a', encoding='utf-8') as file:
                        file.write(
                            f"Early stopping triggered at epoch {epoch_idx}\n")
                    break

            if val_loss < 0.95*best_val_loss:
                epochs_no_improve = 0

    epoch_loop_end_time = time.time()
    epoch_loop_total_time = epoch_loop_end_time - epoch_loop_start_time

    with open(args['report_file'], 'a', encoding='utf-8') as file:
        file.write(
            f"++ the time for the epochs loop was {epoch_loop_total_time:.0f} s\n")

    torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'scaling_params': scaling_params}, 
                f"./models/Z_trained_models/{args['run_tag']}.pth")


def validate_model(args, model, prepare_input_function, val_loader, median=0, mad=1):

    model.eval()
    total_loss = 0.0

    loss_l1 = nn.L1Loss()

    for batch_idx, data in enumerate(val_loader):
        # Transfer data to the appropriate device
        y = data[args['target']].to(args['device'], args['dtype'])

        # Making the input according to the model
        input = prepare_input_function(data, args)

        pred, _ = model(input)

        # Compute the loss using MAE
        loss = loss_l1(pred, (y - median) / mad)

        # Accumulate the loss
        total_loss += loss.item()

    # Calculate average loss over the batches
    avg_loss = total_loss / len(val_loader)

    return avg_loss


def validate_model_custom_loss(args, model, prepare_input_function, val_loader, alpha, median=0, mad=1):

    model.eval()
    total_loss = 0.0

    loss_l1 = nn.L1Loss()

    for batch_idx, data in enumerate(val_loader):
        # Transfer data to the appropriate device
        y = data[args['target']].to(args['device'], args['dtype'])
        true_contributions = data[f"{args['target']}_benson_coefs"].to(
            args['device'], args['dtype'])

        # Making the input according to the model
        input = prepare_input_function(data, args)

        pred, contributions = model(input)

        # Calculate prediction loss
        pred_loss = loss_l1(pred, (y - median) / mad)

        # Calculate node contributions loss
        contributions_loss = loss_l1(contributions, true_contributions)

        # Combine the losses
        loss = (1 - alpha) * pred_loss + alpha * contributions_loss

        # Accumulate the loss
        total_loss += loss.item()

    # Calculate average loss over the batches
    avg_loss = total_loss / len(val_loader)

    return avg_loss


def test_model(args, model, prepare_input_function, test_loader, train_median=0, train_mad=1):

    model.eval()
    total_loss = 0.0

    # Lists to store target and predicted values
    all_targets = []
    all_predictions = []

    loss_l1 = nn.L1Loss()

    for batch_idx, data in enumerate(test_loader):
        # Transfer data to the appropriate device
        y = data[args['target']].to(args['device'], args['dtype'])
        all_targets.append(y.cpu().detach())

        # Making the input according to the model
        input = prepare_input_function(data, args)

        # Forward pass: compute the model output
        pred, _ = model(input)
        pred = (pred * train_mad) + train_median 
        all_predictions.append(pred.cpu().detach())

        # Compute the loss using MAE
        loss = loss_l1(pred , y)

        # Accumulate the loss
        total_loss += loss.item()

    # Calculate average loss over the batches
    avg_loss = total_loss / len(test_loader)

    # Concatenate all batches for targets and predictions
    all_targets = torch.cat(all_targets, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    return avg_loss, all_targets, all_predictions


def compute_median_mad(train_data):
    values = torch.tensor(train_data)
    median = torch.median(values)
    ma = torch.abs(values - median)
    mad = torch.mean(ma)
    return median, mad


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AlphaScheduler:
    def __init__(self, alpha_start, alpha_end, num_epochs):
        self.alpha = alpha_start
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.num_epochs = num_epochs
        self.current_epoch = 0

    def step(self):
        # Linear interpolation
        self.alpha = (self.alpha_end - self.alpha_start) * \
            (self.current_epoch / self.num_epochs) + self.alpha_start
        self.current_epoch += 1

    def get_alpha(self):
        return self.alpha
