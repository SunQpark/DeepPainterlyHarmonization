import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image, ImageFilter

from src.model import FeatureExtracter
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


def prepare_input(data_dir, index, stage=1):
    # open images with PIL library
    img_style = Image.open(os.path.join(data_dir, f'{index}_target.jpg'))
    img_content = Image.open(os.path.join(data_dir, f'{index}_naive.jpg'))
    img_tight_mask = Image.open(os.path.join(data_dir, f'{index}_c_mask.jpg'))
    img_dil_mask = Image.open(os.path.join(data_dir, f'{index}_c_mask_dilated.jpg'))

    # transforms expected by every pretrained models in pytorch, ref: https://pytorch.org/docs/stable/torchvision/models.html
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])

    tsfm = transforms.Compose([to_tensor, normalize])

    # transform images and expand batch dimension
    tensor_style = tsfm(img_style).unsqueeze_(0)
    tensor_content = tsfm(img_content).unsqueeze_(0)
    tensor_batch = torch.cat([tensor_style, tensor_content.detach()], dim=0) # concat style and content images

    img_tight_mask = img_tight_mask.filter(ImageFilter.GaussianBlur(3))
    # tensor_tight_mask = to_tensor(img_tight_mask)[0].float().to(device)

    if stage == 2:
        img_inter = Image.open(os.path.join(data_dir, f'{index}_stage1_out.jpg'))
        tensor_inter = tsfm(img_inter).unsqueeze_(0)
        tensor_batch = torch.cat([tensor_batch, tensor_inter.detach()], dim=0)
    
    return tensor_batch, img_tight_mask, img_dil_mask


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

def nearest_search(f_input, f_style, mask):
    patch_input = get_patches(f_input, mask)
    patch_style = get_patches(f_style)
    
    # compute matrix of l2 norm between each pairs of patches
    l2_input = torch.sum(patch_input**2, dim=0).unsqueeze_(1)
    l2_style = torch.sum(patch_style**2, dim=0).unsqueeze_(0)
    
    l2_matrix = -2 * torch.mm(patch_input.t(), patch_style) + l2_input + l2_style # sqrt is omitted since it will have no effect on comparing

    return torch.argmin(l2_matrix, dim=1)

def independent_mapping(f_input, f_style, mask):
    
    nearest_idx = nearest_search(f_input, f_style, mask)
    # remap the activation vectors of style features using nearest indices
    mapped_features = torch.cat([row.take(nearest_idx).unsqueeze(0) for row in select_mask(f_style)], dim=0)
    mapped_style = f_style.masked_scatter_(mask, mapped_features)
    
    return mapped_style


def gram(inp):
    mat = inp.view(inp.shape[0], -1)
    return torch.mm(mat, mat.t())

def loss_content(output, refer, mask, weight=10):
    return F.mse_loss(
        select_mask(output, mask),
        select_mask(refer, mask)) * weight

def gram_mse(output, refer, mask=None):
    if mask is not None:
        output = output * mask.float()
        refer = refer * mask.float()
    return F.mse_loss(
        gram(output),
        gram(refer))

def loss_style_stage1(fts_target, fts_style, masks, weight=10):
    s_loss = 0
    for i, (target, ref_s, mask) in enumerate(zip(fts_target, fts_style, masks)):
        s_loss += gram_mse(target.squeeze(0), ref_s, mask)
    return s_loss * weight / 3


def stage1_loss(fts_target, fts_style, fts_content, masks):
    s_loss = loss_style_stage1(fts_target[2:], ref_style[2:], masks[2:], weight=10)
    c_loss = loss_content(fts_target[3].squeeze(0), ref_style[3], masks[3], weight=1)
    loss = s_loss + c_loss
    return loss



if __name__ == '__main__':
    image_index = 0
    model = FeatureExtracter().to(device)
    batch_inputs, img_tight_mask, img_dil_mask = prepare_input(data_dir='data/', index=image_index)
    batch_inputs = batch_inputs.to(device)

    # print(model)
    features = model(batch_inputs) # python list of activations 
    masks = resize_masks(img_dil_mask, features)
    ref_features = []
    ref_style = []
    ref_content = []

    print('Stage1: Begin Mapping')
    with torch.no_grad():
        for i, (feat, mask) in enumerate(zip(features, masks)):
            feat_no_grad = feat
            f_style, f_content= torch.unbind(feat.detach(), dim=0)
            if i in [2, 3, 4]:
                f_style = independent_mapping(f_content, f_style, mask)
            ref_style.append(f_style)
            ref_content.append(f_content) 

    print('Stage1: Begin Reconstruction')

    # prepare image to be optimized by copying content image
    content = batch_inputs[1].unsqueeze(0)
    opt_img = torch.tensor(content, device=device, requires_grad=True)

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

            loss = stage1_loss(output, ref_style, ref_content, masks)
            loss.backward()
            if n_iter % show_iter == 0: 
                print(f'Iteration: {n_iter}, loss: {loss.item()}')
            return loss
        optimizer.step(closure)


    tight_mask = transforms.ToTensor()(img_tight_mask)[0].float().to(device)

    output_path = f'data/{image_index}_stage1_out'
    with torch.no_grad():
        output_img = opt_img * tight_mask + batch_inputs[0] * (1 - tight_mask)

        np.save(f'{output_path}.npy', output_img.data.cpu().numpy())
        # print(opt_img.shape)
        
        #denormalize and save output image
        inv_tsfm = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255]), 
            lambda x: x.cpu().numpy(),
            lambda x: np.clip(x, 0.0, 1.0),
            torch.from_numpy,
            transforms.ToPILImage()])
        inv_tsfm(output_img.cpu().data[0]).save(f'{output_path}.jpg')
        