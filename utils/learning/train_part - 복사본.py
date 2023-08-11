import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import cv2
import matplotlib.pyplot as plt

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.unet import Unet


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type, device):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        input, target, maximum, _, _ = data
        input = input.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        
        mask = np.zeros(target.shape)
        mask[target>5e-5] = 1
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations = 1)
        mask = cv2.dilate(mask, kernel, iterations = 15)
        mask = cv2.erode(mask, kernel, iterations = 14)
        mask = (torch.from_numpy(mask).to(device=device)).type(torch.float)

        output = model(input)
        target = target.cuda(non_blocking=True)
        
        loss = loss_type(output*mask, target*mask, maximum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    inputs = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, _, fnames, slices = data
            input = input.cuda(non_blocking=True)
            output = model(input)

            for i in range(output.shape[0]):
                mask = np.zeros_like(target[i])
                mask[target[i]>5e-5] = 1
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.erode(mask, kernel, iterations = 1)
                mask = cv2.dilate(mask, kernel, iterations = 15)
                mask = cv2.erode(mask, kernel, iterations = 14)            
                
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy() * mask
                targets[fnames[i]][int(slices[i])] = target[i].numpy() * mask
                inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()
                

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
        
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, inputs, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    
    model = Unet(in_chans = args.in_chans, out_chans = args.out_chans)
    model.to(device=device)
    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0

    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)

    epochs = [i for i in range(1,args.num_epochs+1)]
    train_losses = []
    val_losses = []
    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type, device)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)

        train_losses.append(train_loss)
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = args.val_loss_dir / "val_loss_log"
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        val_loss = val_loss / num_subjects
        val_losses.append(val_loss)
        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
    plt.plot(epochs, train_losses)
    plt.plot(epochs, val_losses)
    plt.legend(['train', 'val'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss graph')
    plt.show()
    