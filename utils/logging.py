import torch
import shutil
import os
import torchvision.utils as tvu
import cv2
import numpy as np


def mu_tonemap(hdr_image, mu=5000):
    return np.log(1 + mu * hdr_image) / np.log(1 + mu)

def resume(ldr_image, mu=5000):
    return (np.exp(ldr_image * np.log(1 + mu)) - 1) / mu

def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    psnr_pred = torch.squeeze(img.clone())
    psnr_pred = psnr_pred.data.cpu().numpy().astype(np.float32)
    pred_rgb = psnr_pred.transpose(1, 2, 0)
    pred_rgb = pred_rgb[:, :, ::-1][:1000,:1500]
    pred_rgb = resume(pred_rgb)
    cv2.imwrite(os.path.join(file_directory), pred_rgb)

def save_image_png(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    img = img.cpu().numpy()
    img = torch.from_numpy(img)
    tvu.save_image(img, file_directory)



def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')


def load_checkpoint(path, device):
    if device is None:
        return torch.load(path)
    else:
        return torch.load(path, map_location=device)
