import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image

from src.model import FeatureExtracter
from src.utils import select_mask
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def select_mask(features, mask=None):
    """
    select activation vectors in mask
    features: (c, w, h)
    mask: (w, h)
    """
    assert(len(features.shape)==3)
    if mask is not None:
        activations = torch.masked_select(features, mask).view(features.shape[0], -1)
    else:
        activations = features.view(features.shape[0], -1)
    
    return activations


def prepare_input(data_dir, index):
    # open images with PIL library
    img_style = Image.open(os.path.join(data_dir, f'{index}_target.jpg'))
    img_content = Image.open(os.path.join(data_dir, f'{index}_naive.jpg'))
    img_inter = Image.open(os.path.join(data_dir, f'{index}_stage1_out.jpg'))
    img_mask = Image.open(os.path.join(data_dir, f'{index}_c_mask_dilated.jpg'))

    # transforms used by all pretrained models in pytorch, ref: https://pytorch.org/docs/stable/torchvision/models.html
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std =[0.229, 0.224, 0.225])
    
    tsfm = transforms.Compose([to_tensor, normalize])

    # transform images and expand batch dimension
    tensor_style = tsfm(img_style).unsqueeze_(0)
    tensor_content = tsfm(img_content).unsqueeze_(0)
    tensor_inter = tsfm(img_stage1_out).unsqueeze_(0)

    tensor_batch = torch.cat([tensor_style, tensor_content, tensor_inter], dim=0) # concat style and content images

    return tensor_batch, img_mask


def resize_masks(mask, reference_output):
    result = []
    for l in reference_output:
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize(l.shape[-2:])

        tensor_mask = to_tensor(resize(mask))[0] # apply transforms and drop color channels
        
        result.append(tensor_mask.byte().to(device))
    return result


def get_patches(features, mask=None):
    grid = [(-1, -1), (-1, 0), (-1, 1),
            ( 0, -1), ( 0, 0), ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)]
    c, w, h = features.shape
    if mask is None:
        mask = torch.ones((w, h), dtype=torch.uint8)
    assert(mask.shape == (w, h))
    
    
    features = F.pad(features[None], (1, 1, 1, 1)).squeeze(0)
    mask = F.pad(mask, (2, 2, 2, 2))
    selected = []
    for offset in grid:
        dx, dy = offset
        x0, y0 = 1, 1
        x1, y1 = w+3, h+3
        
        offset_mask = mask[x0+dx : x1+dx , y0+dy : y1+dy].to(device)
        selected.append(select_mask(features, offset_mask))
    patches = torch.cat(selected)
    return patches


def independent_mapping(f_input, f_style, mask):
    patch_input = get_patches(f_input, mask)
    patch_style = get_patches(f_style)
    
    # compute matrix of l2 norm between each pairs of patches
    l2_input = torch.sum(patch_input**2, dim=0).unsqueeze_(1)
    l2_style = torch.sum(patch_style**2, dim=0).unsqueeze_(0)
    
    l2_matrix = -2 * torch.mm(patch_input.t(), patch_style) + l2_input + l2_style # sqrt is omitted since it will have no effect on comparing

    nearest_idx = torch.argmin(l2_matrix, dim=1)
    
    # remap the activation vectors of style features using nearest indices
    mapped_features = torch.cat([row.take(nearest_idx).unsqueeze(0) for row in select_mask(f_style)], dim=0)
    mapped_style = f_style.masked_scatter(mask, mapped_features)
    
    return mapped_style


def gram(inp):
    mat = inp.view(inp.shape[0], -1)
    return torch.mm(mat, mat.t())

def loss_content(output, refer, mask):
    return F.mse_loss(
        select_mask(output, mask),
        select_mask(refer, mask))

def loss_style(output, refer, mask):
    return F.mse_loss(
        gram(select_mask(output, mask)),
        gram(select_mask(refer, mask)))


def stage1_loss(fts_target, fts_refer, masks):
    w_s = 10
    w_c = 1
    s_layers = [2, 3, 4]
    c_layers = [3]

    c_loss = 0
    s_loss = 0
    for i, (target, refer, mask) in enumerate(zip(fts_target, fts_refer, masks)):
        ref_s, ref_c = torch.unbind(refer)
        if i in s_layers:
            s_loss += loss_style(target[0], ref_s, mask) / 3

        if i in c_layers:
            c_loss += loss_content(target[0], ref_c, mask)
    loss = w_s * s_loss + w_c * c_loss
    return loss



if __name__ == '__main__':
    model = FeatureExtracter().to(device)
    image_index = 16
    batch_inputs, img_mask = prepare_input(data_dir='data/', index=image_index)
    batch_inputs = batch_inputs.to(device)

    # print(model)
    features = model(batch_inputs) # python list of activations 
    ref_features = []
    masks = resize_masks(img_mask, features)

    print('Pass1: Begin Independent Mapping')
    with torch.no_grad():
        for i, (feat, mask) in enumerate(zip(features, masks)):
            feat_no_grad = feat
            if i in [2, 3, 4]:
                # mask = mask.byte().to(device)
                f_style = feat[0]
                f_content = feat[1]
                feat[0] = independent_mapping(f_content, f_style, mask)
                feat_no_grad = feat.detach()
            ref_features.append(feat_no_grad)


    print('Pass1: Begin Reconstruction')

    # prepare image to be optimized by copying content image
    content = batch_inputs[1].unsqueeze(0)
    opt_img = torch.zeros_like(content).new_tensor(content.data, device=device, requires_grad=True)

    optimizer = optim.LBFGS([opt_img], lr=1)
    n_iter = 0
    max_iter = 1000
    show_iter = 100
    while  n_iter <= max_iter: 
        def closure():
            optimizer.zero_grad()
            global n_iter
            n_iter += 1
            output = model(opt_img)

            loss = stage1_loss(output, ref_features, masks)
            loss.backward()
            if n_iter % show_iter == 0: 
                print(f'Iteration: {n_iter}, loss: {loss.item()}')
            return loss
        optimizer.step(closure)

    output_path = f'data/{image_index}_stage1_out'
    with torch.no_grad():
        np.save(f'{output_path}.npy', opt_img.data.cpu().numpy())
        # print(opt_img.shape)
        
        #denormalize and save output image
        inv_tsfm = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255]), 
            transforms.ToPILImage()])
        inv_tsfm(opt_img.cpu().data[0]).save(f'{output_path}.jpg')
        