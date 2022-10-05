import torch
import torch.nn.functional as F
import numpy as np
import math
import time

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if classname.find('KernelConv2D') == -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.04)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.04)
        torch.nn.init.constant_(m.bias.data, 0)

def adjust_learning_rate(optimizer, epoch, decay_rate, decay_every, lr_init):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lrs = []
    for param_group in optimizer.param_groups:

        lr = param_group['lr_init'] * (decay_rate ** (epoch // decay_every))
        param_group['lr'] = lr
        lrs.append(lr)

    return lrs

def norm_feat_vis(feat):
    l2norm = torch.sqrt(np.finfo(float).eps + torch.sum(torch.mul(feat,feat),1,keepdim=True))
    feat = feat / (l2norm + np.finfo(float).eps)
    feat = feat.permute(1, 0, 2, 3)
    return feat

def norm_res_vis(res):
    res = res.detach().clone()
    b, c, h, w = res.size()

    res = res.view(res.size(0), -1)
    res = res - res.min(1, keepdim=True)[0]
    res = res / res.max(1, keepdim=True)[0]
    res = res.view(b, c, h, w)

    return res

def OF_vis(OF):
    OF = OF.cpu().numpy().transpose(1, 2, 0)
    OF = flow2img(OF)
    OF = torch.FloatTensor(np.expand_dims(OF.transpose(2, 0, 1), axis = 0))
    return OF


#from DPDD code
def get_psnr2(img1, img2, PIXEL_MAX=1.0):
    mse_ = torch.mean( (img1 - img2) ** 2 )
    return 10 * torch.log10(PIXEL_MAX / mse_)

    # return calculate_psnr(img1, img2)

Backward_tensorGrid = {}
def warp(tensorInput, tensorFlow, padding_mode = 'zeros'):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).to(torch.device('cuda'), non_blocking = True)

    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode=padding_mode, align_corners = True)

def upsample(inp, h = None, w = None, mode = 'bilinear'):
    return F.interpolate(input=inp, size=(int(h), int(w)), mode=mode)

def FM(F1, F2, F3, corr_, pool, flow_prev=None, scale=None):
    # rank = torch.distributed.get_rank()

    # if rank <= 0: print('!!!! 1\n\n')
    if flow_prev is None:
        shape = F1.size()
        channel = shape[1]
        # l2norm = torch.sqrt(torch.sum(torch.mul(F1,F1),1,keepdim=True) + 1e-8)
        # l2norm2 = torch.sqrt(torch.sum(torch.mul(F2,F2),1,keepdim=True) + 1e-8)
        # corr = channel * corr_(F1 / (l2norm + 1e-8), F2 / (l2norm2 + 1e-8))
        F1 = F.normalize(F1, p = 2, dim=1)
        F2 = F.normalize(F2, p = 2, dim=1)
        corr = channel * corr_(F1, F2)

        corr = pool(corr)
        matching_index = torch.argmax(corr, dim=1).type(torch.cuda.FloatTensor)
        # matching_index = torch.argmax(corr, dim=1, dtype=torch.cuda.FloatTensor, device=self.device('cuda')).type(torch.cuda.FloatTensor)
        # print(corr.size(), matching_index.size())

        kernel_size = corr_.max_displacement * 2 + 1
        half_ks = np.floor(kernel_size/(corr_.stride2*2))
        y = ((matching_index//np.floor(kernel_size/float(corr_.stride2)+0.5)) - half_ks) * corr_.stride2
        x = ((matching_index%np.floor(kernel_size/float(corr_.stride2)+0.5)) - half_ks) * corr_.stride2

        flow = torch.cat((torch.unsqueeze(x,1),torch.unsqueeze(y,1)), axis=1)
        if scale is not None:
            shape = F3.size()
            flow = F.interpolate(input=flow, scale_factor=scale, mode='nearest') * scale
    else:
        corr = None
        flow = flow_prev

    n = warp(F3, flow)
    return n, corr, flow

def flow2img(flow_data):
    """
    convert optical flow into color image
    :param flow_data:
    :return: color image
    """
    # print(flow_data.shape)
    # print(type(flow_data))
    u = flow_data[:, :, 1]
    v = flow_data[:, :, 0]

    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return img / 255.

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: horizontal optical flow
    :param v: vertical optical flow
    :return:
    """

    height, width = u.shape
    img = np.zeros((height, width, 3))

    NAN_idx = np.isnan(u) | np.isnan(v)
    u[NAN_idx] = v[NAN_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

    return img

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel
