import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import memory_summary

#### misc #####

def parameter_count(model):
    print('# parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

def gpu_usage():
    print('gpu usage (current/max): {:.2f} / {:.2f} GB'.format(torch.cuda.memory_allocated()*1e-9, torch.cuda.max_memory_allocated()*1e-9))

#### distance #####

def pdist(x, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=2)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist

def pdist2(x, y, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(dim=3)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = (y**2).sum(dim=2).unsqueeze(1)
        dist = xx + yy - 2.0 * torch.bmm(x, y.permute(0, 2, 1))
    return dist

#### filter #####

def filter1D(img, weight, dim, padding_mode='replicate'):
    B, C, D, H, W = img.shape
    N = weight.shape[0]
    
    padding = torch.zeros(6,)
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N//2
    padding = padding.long().tolist()
    
    view = torch.ones(5,)
    view[dim + 2] = -1
    view = view.long().tolist()
    
    return F.conv3d(F.pad(img.view(B*C, 1, D, H, W), padding, mode=padding_mode), weight.view(view)).view(B, C, D, H, W)

def smooth(img, sigma):
    device = img.device
    
    sigma = torch.tensor([sigma]).to(device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1
    
    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N).to(device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()
    
    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)
    
    return img

#### feature #####

def mindssc(img, delta=1, sigma=0.8):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor
    
    device = img.device
    dtype = img.dtype
    
    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]]).long()
    
    # squared distances
    dist = pdist(six_neighbourhood.unsqueeze(0)).squeeze(0)
    
    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))
    
    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).to(dtype).to(device)
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).to(dtype).to(device)
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad = nn.ReplicationPad3d(delta)
    
    # compute patch-ssd
    ssd = smooth(((F.conv3d(rpad(img), mshift1, dilation=delta) - F.conv3d(rpad(img), mshift2, dilation=delta)) ** 2), sigma)
    
    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
    mind /= mind_var
    mind = torch.exp(-mind).to(dtype)
    
    #permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
    
    return mind

#### transform #####

def warp_img(img, disp, mode='bilinear', padding_mode='border'):
    _, _, D, H, W = img.shape
    device = img.device
    
    identity = F.affine_grid(torch.eye(3, 4).unsqueeze(0).to(device), (1, 1, D, H, W), align_corners=True)

    return F.grid_sample(img, identity + disp, mode=mode, padding_mode=padding_mode, align_corners=True)

#### similarity metrics #####

def ssd(kpts_fixed, feat_fixed, feat_moving, orig_shape, disp_radius=16, disp_step=2, patch_radius=2, alpha=1.5, unroll_factor=50):
    _, N, _ = kpts_fixed.shape
    device = kpts_fixed.device
    D, H, W = orig_shape
    C = feat_fixed.shape[1]
    dtype = feat_fixed.dtype
    
    patch_step = disp_step # same stride necessary for fast implementation
    patch = torch.stack(torch.meshgrid(torch.arange(0, 2 * patch_radius + 1, patch_step),
                                       torch.arange(0, 2 * patch_radius + 1, patch_step),
                                       torch.arange(0, 2 * patch_radius + 1, patch_step))).permute(1, 2, 3, 0).contiguous().view(1, 1, -1, 1, 3).float() - patch_radius
    patch = (patch.flip(-1) * 2 / (torch.tensor([W, H, D]) - 1)).to(dtype).to(device)
    
    patch_width = round(patch.shape[2] ** (1.0 / 3))
    
    if patch_width % 2 == 0:
        pad = [(patch_width - 1) // 2, (patch_width - 1) // 2 + 1]
    else:
        pad = [(patch_width - 1) // 2, (patch_width - 1) // 2]
    
    disp = torch.stack(torch.meshgrid(torch.arange(- disp_step * (disp_radius + ((pad[0] + pad[1]) / 2)), (disp_step * (disp_radius + ((pad[0] + pad[1]) / 2))) + 1, disp_step),
                                      torch.arange(- disp_step * (disp_radius + ((pad[0] + pad[1]) / 2)), (disp_step * (disp_radius + ((pad[0] + pad[1]) / 2))) + 1, disp_step),
                                      torch.arange(- disp_step * (disp_radius + ((pad[0] + pad[1]) / 2)), (disp_step * (disp_radius + ((pad[0] + pad[1]) / 2))) + 1, disp_step))).permute(1, 2, 3, 0).contiguous().view(1, 1, -1, 1, 3).float()
    disp = (disp.flip(-1) * 2 / (torch.tensor([W, H, D]) - 1)).to(dtype).to(device)
    
    disp_width = disp_radius * 2 + 1
    ssd = torch.zeros(1, N, disp_width ** 3).to(dtype).to(device)
    split = np.array_split(np.arange(N), unroll_factor)
    for i in range(unroll_factor):
        feat_fixed_patch = F.grid_sample(feat_fixed, kpts_fixed[:, split[i], :].view(1, -1, 1, 1, 3).to(dtype) + patch, padding_mode='border', align_corners=True)
        feat_moving_disp = F.grid_sample(feat_moving, kpts_fixed[:, split[i], :].view(1, -1, 1, 1, 3).to(dtype) + disp, padding_mode='border', align_corners=True)        
        corr = F.conv3d(feat_moving_disp.view(1, -1, disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1]), feat_fixed_patch.view(-1, 1, patch_width, patch_width, patch_width), groups=C * split[i].shape[0]).view(C, split[i].shape[0], -1)
        patch_sum = (feat_fixed_patch ** 2).squeeze(0).squeeze(3).sum(dim=2, keepdims=True)
        disp_sum = (patch_width ** 3) * F.avg_pool3d((feat_moving_disp ** 2).view(C, -1, disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1]), patch_width, stride=1).view(C, split[i].shape[0], -1)
        ssd[0, split[i], :] = ((- 2 * corr + patch_sum + disp_sum)).sum(0)
    
    ssd *= (alpha / (patch_width ** 3))
    
    return ssd

#### keypoints #####

def kpts_pt(kpts_world, shape):
    device = kpts_world.device
    D, H, W = shape
    return (kpts_world.flip(-1) / (torch.tensor([W, H, D]).to(device) - 1)) * 2 - 1

def kpts_world(kpts_pt, shape):
    device = kpts_pt.device
    D, H, W = shape
    return ((kpts_pt.flip(-1) + 1) / 2) * (torch.tensor([D, H, W]).to(device) - 1)

def farthest_point_sampling(kpts, num_points):
    _, N, _ = kpts.size()
    ind = torch.zeros(num_points).long()
    ind[0] = torch.randint(N, (1,))
    dist = torch.sum((kpts - kpts[:, ind[0], :]) ** 2, dim=2)
    for i in range(1, num_points):
        ind[i] = torch.argmax(dist)
        dist = torch.min(dist, torch.sum((kpts - kpts[:, ind[i], :]) ** 2, dim=2))
       
    return kpts[:, ind, :], ind

def structure_tensor(img, sigma):
    B, C, D, H, W = img.shape
    device = img.device
    
    struct = []
    for i in range(C):
        for j in range(i, C):
            struct.append(smooth((img[:, i, ...] * img[:, j, ...]).unsqueeze(1), sigma))

    return torch.cat(struct, dim=1)

def invert_structure_tensor(struct):
    a = struct[:, 0, ...]
    b = struct[:, 1, ...]
    c = struct[:, 2, ...]
    e = struct[:, 3, ...]
    f = struct[:, 4, ...]
    i = struct[:, 5, ...]

    A =   e*i - f*f
    B = - b*i + c*f
    C =   b*f - c*e
    E =   a*i - c*c
    F = - a*f + b*c
    I =   a*e - b*b

    det = (a*A + b*B + c*C).unsqueeze(1)

    struct_inv = (1./det) * torch.stack([A, B, C, E, F, I], dim=1)

    return struct_inv

def foerstner_kpts(img, mask, sigma=1.4, d=9, thresh=1e-8, num_points=None):
    _, _, D, H, W = img.shape
    device = img.device
    dtype = img.dtype
    
    filt = torch.tensor([1.0 / 12.0, -8.0 / 12.0, 0.0, 8.0 / 12.0, -1.0 / 12.0]).to(dtype).to(device)
    grad = torch.cat([filter1D(img, filt, 0),
                      filter1D(img, filt, 1),
                      filter1D(img, filt, 2)], dim=1)
    
    struct_inv = invert_structure_tensor(structure_tensor(grad, sigma))
    
    distinctiveness = 1. / (struct_inv[:, 0, ...] + struct_inv[:, 3, ...] + struct_inv[:, 5, ...]).unsqueeze(1)
    
    pad1 = d//2
    pad2 = d - pad1 - 1
    
    maxfeat = F.max_pool3d(F.pad(distinctiveness, (pad2, pad1, pad2, pad1, pad2, pad1)), d, stride=1)
    
    structure_element = torch.tensor([[[0., 0,  0],
                                       [0,  1,  0],
                                       [0,  0,  0]],
                                      [[0,  1,  0],
                                       [1,  0,  1],
                                       [0,  1,  0]],
                                      [[0,  0,  0],
                                       [0,  1,  0],
                                       [0,  0,  0]]]).to(device)
    
 
    mask_eroded = (1 - F.conv3d(1 - mask.to(dtype), structure_element.unsqueeze(0).unsqueeze(0), padding=1).clamp_(0, 1)).bool()
    
    kpts = torch.nonzero(mask_eroded & (maxfeat == distinctiveness) & (distinctiveness >= thresh)).unsqueeze(0).to(dtype)[:, :, 2:]
    
    if not num_points is None:
        kpts = farthest_point_sampling(kpts, num_points)[0]
    
    return kpts_pt(kpts, (D, H, W))

def knn_graph(kpts, k, include_self=False):
    B, N, D = kpts.shape
    device = kpts.device
    
    dist = pdist(kpts)
    ind = (-dist).topk(k + (1 - int(include_self)), dim=-1)[1][:, :, 1 - int(include_self):]
    A = torch.zeros(B, N, N).to(device)
    A[:, torch.arange(N).repeat(k), ind[0].t().contiguous().view(-1)] = 1
    A[:, ind[0].t().contiguous().view(-1), torch.arange(N).repeat(k)] = 1
    
    return ind, dist*A, A


