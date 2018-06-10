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
    # features.squeeze_(0)
    # c, w, h = features.shape
    if mask is not None:
        activations = torch.masked_select(features, mask).view(features.shape[0], -1)
    else:
        activations = features.view(features.shape[0], -1)
    
    return activations


def prepare_input(data_dir, index):
    # open images with PIL library
    img_style = Image.open(os.path.join(data_dir, f'{index}_target.jpg'))
    img_content = Image.open(os.path.join(data_dir, f'{index}_naive.jpg'))
    img_mask = Image.open(os.path.join(data_dir, f'{index}_c_mask_dilated.jpg'))

    # transforms used by all pretrained models in pytorch, ref: https://pytorch.org/docs/stable/torchvision/models.html
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std =[0.229, 0.224, 0.225])
    

    tsfm = transforms.Compose([to_tensor, normalize])

    # transform images and expand batch dimension
    tensor_style = tsfm(img_style).unsqueeze_(0)
    tensor_content = tsfm(img_content).unsqueeze_(0)

    tensor_batch = torch.cat([tensor_style, tensor_content], dim=0) # concat style and content images

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
    
    
    features = F.pad(features[None], (1, 1, 1, 1), mode='reflect').squeeze(0) # TODO: try reflection padding
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

    nearest_idx = torch.argmax(l2_matrix, dim=1)
    
    # remap the activation vectors of style features using nearest indices
    mapped_features = torch.cat([row.take(nearest_idx).unsqueeze(0) for row in select_mask(f_style)], dim=0)
    mapped_style = f_style.masked_scatter_(mask, mapped_features)
    
    return mapped_style


def gram(inp):
    mat = inp.view(inp.shape[0], -1)
    return torch.mm(mat, mat.transpose(0, 1))

def loss_content(output, refer, mask):
    return F.mse_loss(
        select_mask(output, mask), 
        select_mask(refer, mask))

def loss_style(output, refer, mask):
    return F.mse_loss(
        gram(select_mask(output, mask)), 
        gram(select_mask(refer, mask)))


def stage1_loss(fts_target, fts_refer, masks):
    w_c = 1
    w_s = 10
    s_layers = [2, 3, 4]
    c_layers = [3]

    # loss = torch.tensor([0.0], device=device)
    # losses = torch.zeros((len(s_layers + c_layers, ), ), dtype=torch.float32)
    # s_loss = torch.tensor([0.0, 0.0, 0.0], device=device)
    # c_loss = torch.tensor([0.0], device=device)

    # ref_s = []
    # ref_c = []
    # for refer in fts_refer:
    #     s, c = torch.unbind(refer)
    #     ref_s.append(s)
    #     ref_c.append(c)
    # s_loss = 0
    # c_loss = 0
    losses = []
    for i, (target, refer, mask) in enumerate(zip(fts_target, fts_refer, masks)):

        # if i <= 2:
        #     continue
        # s_loss += loss_style(target, ref_s, mask) * w_s
        # c_loss += loss_content(target, ref_c, mask) * w_c
        
        ref_s, ref_c = torch.unbind(refer)
        if i in s_layers:
            # loss += s_loss
            losses.append(loss_style(target.squeeze(0), ref_s, mask) * w_s)
            # break
        if i in c_layers:
            # loss += s_loss
            losses.append(loss_content(target.squeeze(0), ref_c, mask) * w_c)
        # else:
        #     s_loss = torch.tensor([0.0], device=device)
        # flag_s = torch.tensor([i in s_layers], device=device, dtype=torch.float32)
        # s_loss += loss_style(target.squeeze(0), ref_s, mask) * w_s * flag_s
        # flag_c = torch.tensor([i in c_layers], device=device, dtype=torch.float32)
        # c_loss += loss_content(target.squeeze(0), ref_c, mask) * w_c * flag_c
            
        # ref_s, ref_c = torch.unbind(refer)
        # s_loss.append(loss_style(target.squeeze(0), ref_s, mask) * w_s)
        # if i == 2:
        #     break
                    # ref_s, ref_c = torch.unbind(refer)
            # loss += c_loss
        
        
            
        # # loss += s_loss + c_loss
        # if i != 2:
        #     loss += loss_style(target, ref_s, mask) * w_s
        # if i == 2:
        #     break
        # target = target.squeeze(0)
        # loss = loss_content(target, ref_c, mask) * w_c
        # if i == 3:
        #     break
    # loss += s_loss + c_loss
    loss = sum(losses)
    # s_loss[0] += loss_style(fts_target[2].squeeze(0), ref_s[2], masks[2])
    # s_loss[1] += loss_style(fts_target[3].squeeze(0), ref_s[3], masks[3])
    # s_loss[2] += loss_style(fts_target[4].squeeze(0), ref_s[4], masks[4])
    # loss = loss_style(fts_target[2].squeeze(0), ref_s[2], masks[2])
    # target = fts_target[2].squeeze(0)
    # ref_s, ref_c = torch.unbind(fts_refer[2])
    # mask = masks[2]
    # loss += loss_style(target, ref_s, mask) * w_s
    # for l in s_loss:
    # loss += s_loss[0]
    # loss += s_loss[1]
    # print(loss)
    return loss



if __name__ == '__main__':
    model = FeatureExtracter().to(device)
    batch_inputs, img_mask = prepare_input(data_dir='data/', index=9)
    batch_inputs = batch_inputs.to(device)


    features = model(batch_inputs) # python list of activations 
    masks = resize_masks(img_mask, features)

    print('Pass1: Begin Independent Mapping')
    with torch.no_grad():
        for i, (feat, mask) in enumerate(zip(features, masks)):
            if i in [2, 3, 4]:
                # mask = mask.byte().to(device)

                f_style = feat[0]
                f_content = feat[1]
                feat[0] = independent_mapping(f_content, f_style, mask)

    print('Pass1: Begin Reconstruction')

    # prepare image to be optimized by copying content image
    content = batch_inputs[1].unsqueeze(0)
    opt_img = torch.zeros_like(content).new_tensor(content.data, device=device, requires_grad=True)

    # opt_img = opt_img.to(device)
    optimizer = optim.LBFGS([opt_img], lr=1)
    # optimizer = optim.Adam([opt_img], lr=0.01, amsgrad=True)
    # for p in optimizer.param_groups.
    #     if isinstance(p, torch.Tensor):
    #         p = p.to(device)
    # optimizer = optimizer.to(device)
    n_iter = 0
    max_iter = 1000
    show_iter = 100
    while  n_iter <= max_iter: 
        # n_iter += 1
        # optimizer.zero_grad()
        # output = model(opt_img)
        # loss = stage1_loss(output, features, masks)
        # loss.backward()
        # optimizer.step()
        # if n_iter % show_iter == 0: 
        #     print(f'Iteration: {n_iter}, loss: {loss.item()}')
        # return loss
    
        def closure():
            optimizer.zero_grad()
            global n_iter
            n_iter += 1
            output = model(opt_img)
            # loss = torch.sum(output[0])
            loss = stage1_loss(output, features, masks)
            loss.backward()
            if n_iter % show_iter == 0: 
                print(f'Iteration: {n_iter}, loss: {loss.item()}')
            print(loss)
            return loss
        optimizer.step(closure)

    np.save(opt_img.cpu().numpy(), 'result.npy')

    inv_tsfm = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255]), 
        transforms.ToPILImage()])

    out_image = inv_tsfm(opt_img)
    save_image(out_image, f'data/{9}_out1.jpg')
