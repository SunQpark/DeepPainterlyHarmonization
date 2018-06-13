import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image

from src.stage1 import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def apply_offset(idx, offset, size):
    dx, dy = offset
    w, h = size

    result = idx + dx
    # check index is not out of horizontal range 
    if result // w != idx // w:
        result = -1
    else:
        result += w * dy
    # check index is not out of vertical range 
    if result < 0 or result >= w * h:
        result = -1
    return result

def second_match(init_map, f_style, mask, kernel_size=5):
    # init_map = nearest_search(f_input, f_style, mask)

    c, w, h = f_style.shape
    # grid pointing to neighbor
    rad = (kernel_size - 1) // 2
    grid = [np.array([i,j]) for i in range(-rad,rad+1) for j in range(-rad,rad+1)]
    
    f_style = select_mask(f_style).t() # shape: (w*h, c)
    mask = mask.view(w*h, 1)
    result = init_map
    k = -1
    for i in range(f_style.shape[0]):
        if mask[i] == 0:
            continue
        else:
            k += 1
        candidates = set()
        ngb_s = []
        for offset in grid:
            i = apply_offset(i, offset, (w, h))
            if i != -1:
                j = apply_offset(init_map[k].item(), -offset, (w, h))

                if j != -1:
                    candidates.add(j)
                    ngb_s.append(j)
        candidates = torch.tensor(list(candidates), device=device, dtype=torch.int64)
        ngb_s = torch.tensor(ngb_s, device=device, dtype=torch.int64)
        
        if len(candidates) <= 1:
            assert(not candidates.empty())
            assert(candidates[0] == i)
        
        cand_ft = torch.index_select(f_style, 0, candidates)
        ngbr_ft = torch.index_select(f_style, 0, ngb_s)
        l2_dist = ((cand_ft[:, None, :] - ngbr_ft[None,:, :]) ** 2)
        l2_dist = l2_dist.sum(dim=2).sum(dim=1)
        result[k] = candidates[torch.argmin(l2_dist)]

    return result


def upsample(ref_map, ref_size, new_size, mask_ref, mask_new):
    r_h, r_w = ref_size
    n_h, n_w = new_size
    new_map = torch.zeros(mask_new.sum().item(), device=device, dtype=torch.int64)
    ratio_h, ratio_w = n_h/r_h, n_w/r_w
    k = -1
    l = -1

    ref_idx_map = torch.full(mask_ref.shape, -1, device=device, dtype=torch.int64).masked_scatter_(mask_ref, ref_map)
    uniq_mask = torch.zeros_like(mask_new, dtype=torch.uint8) # mask that covers unique elts(activations) only
    uniq_elts = set()
    for i in range(n_h * n_w):
        n_x, n_y = i//n_w, i%n_w
        r_x, r_y = int((0.5+n_x) / ratio_h), int((0.5+n_y)/ratio_w)
        r_x, r_y = min(max(0,r_x),r_h-1), min(max(0,r_y),r_w-1)

        if mask_new[n_x, n_y] == 0 or mask_ref[r_x, r_y] == 0:
            continue
        else: 
            k += 1
            
        style_idx = ref_idx_map[r_x, r_y].item()
        s_x = int(n_x + (style_idx // r_w - r_x) * ratio_h + 0.5)
        s_y = int(n_y + (style_idx % r_w - r_y) * ratio_w + 0.5)
        s_x, s_y = min(max(0,s_x),n_h-1), min(max(0,s_y),n_w-1)
        new_map[k] = s_x * n_w + s_y

        if style_idx not in uniq_elts:
            uniq_elts.add(style_idx)
            uniq_mask[n_x, n_y] = 1

    return new_map, uniq_mask

def hist_mask(ft_style, uniq_mask, n_bins=256):
    ft_style = ft_style.cpu()
    uniq_mask = uniq_mask.cpu()
    masked = torch.masked_select(ft_style, uniq_mask).view(ft_style.shape[0], -1)
    return torch.cat([torch.histc(l, n_bins).unsqueeze(0) for l in masked])

def remap_hist(x, hist_ref, n_bins=256):
    ch, n = x.size()
    sorted_x, sort_idx = x.data.sort(1)
    ymin, ymax = sorted_x[:, 0].unsqueeze(1), sorted_x[:, -1].unsqueeze(1)

    hist = hist_ref / hist_ref.sum(1).unsqueeze(1)# Normalization between the different lengths of masks.
    hist = hist.to(device)
    
    cum_ref = n * hist.cumsum(1).to(device)
    cum_prev = torch.cat([torch.zeros(ch, 1, device=device), cum_ref[:,:-1]],1)
    step = (ymax-ymin)/n_bins

    rng = torch.arange(1,n+1).unsqueeze(0).to(device) # rng: range
    
    idx = torch.cat([(cum_ref < k).sum(1, keepdim=True) for k in range(1, n + 1)], dim=1).long().to(device)
    
    ratio = (rng - torch.take(cum_prev, idx).view(ch, -1)) / (1e-8 + torch.take(hist, idx)).view(ch, -1)
    ratio = ratio.squeeze().clamp(0,1)
    new_x = ymin + (ratio + idx.float()) * step
    new_x[:,-1] = ymax.squeeze(1)
    new_x = torch.take(new_x,idx).view(ch, -1)
    return new_x

def hist_loss(fts_target, fts_hist, masks, indices=[0, 3]):
    h_loss = 0
    for i, (target, mask, ref_hist) in enumerate(zip(fts_target, masks, fts_hist)):
        if i in indices:
            target = select_mask(target.squeeze(0), mask)
            h_loss += F.mse_loss(target, remap_hist(target, ref_hist)) 
    return h_loss / 2

def loss_style_stage2(fts_target, fts_style, masks, tmasks, weight=0.01):
    s_loss = 0
    for target, ref_s, mask, tmask in zip(fts_target, fts_style, masks, tmasks):
        target = select_mask(target.squeeze(0), mask)
        ref_s = select_mask(ref_s, tmask)

        ref_s *= float(np.sqrt(mask.sum() / tmask.sum())) # regularize differences of n_elts of masks
        s_loss += gram_mse(target, ref_s)
    return s_loss * weight / 4

def loss_totvar(out):
    return ((out[:,:-1,:] - out[:,1:,:]) ** 2).sum() + ((out[:,:,:-1] - out[:,:,1:]) ** 2).sum()

def get_med_tv(arr):
    ch, h, w = arr.shape
    arr1 = torch.cat([torch.zeros((ch,w), device=device)[:,None,:], arr], dim=1)
    arr2 = torch.cat([torch.zeros((ch,h), device=device)[:,:,None], arr], dim=2)
    return torch.median((arr1[:,:-1,:] - arr1[:,1:,:]) ** 2 + (arr2[:,:,:-1] - arr2[:,:,1:]) ** 2)

def stage2_loss(opt_img, style_tfm, output, ref_style, ref_content, ref_hist, masks, uniq_masks):
    mtv = get_med_tv(style_tfm)
    w_tv = float(10 / (1 + np.exp(min(mtv * 10**4 -25, 30)))) # TODO: handling overflow here

    c_loss = loss_content(output[3].squeeze(0), ref_content[3], masks[3], )
    s_loss = loss_style_stage2(output, ref_style, masks, uniq_masks)
    h_loss = hist_loss(output, ref_hist, masks, indices=[0, 3])
    t_loss = loss_totvar(opt_img[0])
    # print(c_loss.item(), s_loss.item(), h_loss.item(), (w_tv * t_loss).item())
    return c_loss + s_loss + h_loss + w_tv * t_loss
## copy and pasted parts to here

if __name__ == '__main__':
    model = FeatureExtracter().to(device)
    image_index = 8
    batch_inputs, img_tight_mask, img_dil_mask = prepare_input(data_dir='data/', index=image_index, stage=2)
    batch_inputs = batch_inputs.to(device)

    # print(model)
    features = model(batch_inputs)[:-1] # python list of activations, drop last layer features
    ref_style = []
    ref_content = []
    uniq_masks = []
    masks = resize_masks(img_dil_mask, features)
    tmasks = resize_masks(img_tight_mask, features)

    print('Stage2: Begin Mapping')
    with torch.no_grad():
        for i, (feat, mask) in enumerate(zip(features, masks)):
            feat_no_grad = feat
            f_style, f_content, f_stage1 = torch.unbind(feat.detach(), dim=0)
            if i == 3:
                init_map = nearest_search(f_stage1, f_style, mask)
                sec_map = second_match(init_map, f_style, mask, kernel_size=5) # TODO: try with resized original style image 
                mask_ref = mask
                size_ref = f_style.shape[-2:]
            
            ref_style.append(f_style)
            ref_content.append(f_stage1) 

        for i, (f_style, mask) in enumerate(zip(ref_style, masks)):
            new_map, uniq_mask = upsample(sec_map, size_ref, f_style.shape[-2:], mask_ref, mask)
            uniq_masks.append(uniq_mask)


        ref_hist = []
        for i, (f_style, uniq_mask) in enumerate(zip(ref_style, uniq_masks)):
            if i in [0, 3]:
                hist = hist_mask(f_style, uniq_mask)
                ref_hist.append(hist)
                # print(hist.shape)


    print('Stage2: Begin Reconstruction')

    # prepare image to be optimized by copying stage1 output image
    content = batch_inputs[2].unsqueeze(0)
    opt_img = torch.tensor(content.data, requires_grad=True)

    optimizer = optim.LBFGS([opt_img], lr=0.1)
    n_iter = 0
    max_iter = 200
    show_iter = 10
    while  n_iter <= max_iter: 
        def closure():
            optimizer.zero_grad()
            global n_iter
            n_iter += 1
            output = model(opt_img)

            loss = stage2_loss(opt_img, batch_inputs[1], output, ref_style, ref_content, ref_hist, masks, uniq_masks)
            loss.backward()
            if n_iter % show_iter == 0: 
                print(f'Iteration: {n_iter}, loss: {loss.item()}')
            return loss
        optimizer.step(closure)

    
    output_path = f'data/{image_index}_stage2_out'

    with torch.no_grad():
        tight_mask = tmasks[0].float()
        output_img = opt_img * tight_mask + batch_inputs[0] * (1 - tight_mask)

        np.save(f'{output_path}.npy', output_img.data.cpu().numpy())
        # print(opt_img.shape)
        
        #denormalize and save output image
        inv_tsfm = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255]), 
            lambda x: x.clamp(0, 1),
            transforms.ToPILImage()])
        inv_tsfm(output_img.cpu().data[0]).save(f'{output_path}.jpg')
        