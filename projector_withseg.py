""" Projecting input images into latent spaces. """

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import mrcfile

import dnnlib
import legacy

import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

from camera_utils import LookAtPoseSampler
 
def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size


def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    target_back: torch.Tensor,
    c: torch.Tensor,
    c_back: torch.Tensor,
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    optimize_noise             = False,
    verbose                    = False,
    device: torch.device,
    target_img,
    target_fname
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)#随机生成
    camera_lookat_point = torch.tensor([0, 0, 0.0], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)

    # back
    cam2world_pose_back = LookAtPoseSampler.sample(3.14/2 + 3.14, 3.14/2, camera_lookat_point, radius=2.7, device=device)



    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    # back
    c_samples_back = torch.cat([cam2world_pose_back.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    c_samples_combined = torch.cat([c_samples, c_samples_back], dim=0)
    z_samples_combined = torch.cat([torch.from_numpy(z_samples).to(device), torch.from_numpy(z_samples).to(device)], dim=0)
    
    # back
    w_samples = G.mapping(z_samples_combined, c_samples_combined.repeat(w_avg_samples,1))





    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]




    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]



    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    print(w_samples.shape)


    # fix delta_c

    delta_c = G.t_mapping(torch.from_numpy(np.mean(z_samples, axis=0, keepdims=True)).to(device), c[:1], truncation_psi=1.0, truncation_cutoff=None, update_emas=False)
    delta_c = torch.squeeze(delta_c, 1)
    print(delta_c) #tensor([[-0.0001,  0.0055,  0.0113]], device='cuda:0')
    c[:,3] += delta_c[:,0]
    c[:,7] += delta_c[:,1]
    c[:,11] += delta_c[:,2]

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
    target_images_perc = (target_images + 1) * (255/2)
    if target_images_perc.shape[2] > 256:
        target_images_perc = F.interpolate(target_images_perc, size=(256, 256), mode='area')
    target_features = vgg16(target_images_perc, resize_images=False, return_lpips=True)

    # back
    target_images_back = target_back.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
    target_images_perc_back = (target_images_back + 1) * (255/2)
    if target_images_perc_back.shape[2] > 256:
        target_images_perc_back = F.interpolate(target_images_perc_back, size=(256, 256), mode='area')
    target_features_back = vgg16(target_images_perc_back, resize_images=False, return_lpips=True)

    w_avg = torch.tensor(w_avg, dtype=torch.float32, device=device).repeat(1, G.backbone.mapping.num_ws, 1)
    w_opt = w_avg.detach().clone()
    w_opt.requires_grad = True
    
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device="cpu")
    if optimize_noise:
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)#优化w
    else:
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    if optimize_noise:
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise
        synth_images = G.synthesis(ws, c=c, noise_mode='const')['image']
        
        # back
        synth_images_back = G.synthesis(ws, c=c_back, noise_mode='const')['image']

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_perc = (synth_images + 1) * (255/2)
        if synth_images_perc.shape[2] > 256:
            synth_images_perc = F.interpolate(synth_images_perc, size=(256, 256), mode='area')

        # back
        synth_images_perc_back = (synth_images_back + 1) * (255/2)
        if synth_images_perc_back.shape[2] > 256:
            synth_images_perc_back = F.interpolate(synth_images_perc_back, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images_perc, resize_images=False, return_lpips=True)
        perc_loss = (target_features - synth_features).square().sum(1).mean()
        mse_loss = (target_images - synth_images).square().mean()

        w_norm_loss = (w_opt-w_avg).square().mean()

        #back   
        synth_features_back = vgg16(synth_images_perc_back, resize_images=False, return_lpips=True)
        perc_loss_back = (target_features_back - synth_features_back).square().sum(1).mean()
        mse_loss_back = (target_images_back - synth_images_back).square().mean()



        # Noise regularization.
        reg_loss = 0.0
        if optimize_noise:
            for v in noise_bufs.values():
                noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
        
        # back
        loss = 0.5 * (  mse_loss + 0.3 * mse_loss_back) + ( perc_loss + 0.3 * perc_loss_back) + 1.0 * w_norm_loss +  reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step: {step+1:>4d}/{num_steps} mse: {mse_loss:<4.4f} perc: {perc_loss:<4.4f} mse_back: {mse_loss_back:<4.4f} w_norm: {w_norm_loss:<4.4f}  noise: {float(reg_loss):<5.4f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach().cpu()[0]



        # Normalize noise.
        if optimize_noise:
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

    if w_out.shape[1] == 1:
        w_out = w_out.repeat([1, G.mapping.num_ws, 1])


    return w_out, c, c_back 


def project_pti(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    target_back: torch.Tensor,
    w_pivot: torch.Tensor,
    c: torch.Tensor,
    c_back: torch.Tensor,
    *,
    num_steps                  = 1000,
    initial_learning_rate      = 3e-4,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    verbose                    = False,
    device: torch.device,
    target_img,
    target_fname
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).train().requires_grad_(True).to(device) # type: ignore

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
    target_images_perc = (target_images + 1) * (255/2)
    if target_images_perc.shape[2] > 256:
        target_images_perc = F.interpolate(target_images_perc, size=(256, 256), mode='area')
    target_features = vgg16(target_images_perc, resize_images=False, return_lpips=True)

    # back
    target_images_back = target_back.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
    target_images_perc_back = (target_images_back + 1) * (255/2)
    if target_images_perc_back.shape[2] > 256:
        target_images_perc_back = F.interpolate(target_images_perc_back, size=(256, 256), mode='area')
    target_features_back = vgg16(target_images_perc_back, resize_images=False, return_lpips=True)

    w_pivot = w_pivot.to(device).detach()
    optimizer = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999), lr=initial_learning_rate)#优化gan参数

    out_params = []

    for step in range(num_steps):
        # Synth images from opt_w.
        synth_images = G.synthesis(w_pivot, c=c, noise_mode='const')['image']

        # back
        synth_images_back = G.synthesis(w_pivot, c=c_back, noise_mode='const')['image']

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_perc = (synth_images + 1) * (255/2)
        if synth_images_perc.shape[2] > 256:
            synth_images_perc = F.interpolate(synth_images_perc, size=(256, 256), mode='area')

        # back
        synth_images_perc_back = (synth_images_back + 1) * (255/2)
        if synth_images_perc_back.shape[2] > 256:
            synth_images_perc_back = F.interpolate(synth_images_perc_back, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images_perc, resize_images=False, return_lpips=True)
        perc_loss = (target_features - synth_features).square().sum(1).mean()#感知损失
        mse_loss = (target_images - synth_images).square().mean()#均方误差

        # back
        synth_features_back = vgg16(synth_images_perc_back, resize_images=False, return_lpips=True)
        perc_loss_back = (target_features_back - synth_features_back).square().sum(1).mean()
        mse_loss_back = (target_images_back - synth_images_back).square().mean()


        # back
        loss = 0.5 * (  mse_loss + 0.3 * mse_loss_back) + (  perc_loss + 0.3 * perc_loss_back) 


        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step: {step+1:>4d}/{num_steps} mse: {mse_loss:<4.4f} perc: {perc_loss:<4.4f} mse_back: {mse_loss_back:<4.4f}')

        if step == num_steps - 1 or step % 10 == 0:
            out_params.append(copy.deepcopy(G).eval().requires_grad_(False).cpu())

    return out_params

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# @click.option('--target', 'target_fname',       help='Target image file to project to', required=True, metavar='FILE|DIR')
@click.option('--target_img', 'target_img',       help='Target image folder', required=True, metavar='FILE|DIR')
@click.option('--target_img_back', 'target_img_back',       help='Target image folder', required=True, metavar='FILE|DIR')
@click.option('--target_seg', 'target_seg',       help='Target segmentation folder', required=False, metavar='FILE|DIR')
@click.option('--idx',                    help='index from dataset', type=int, default=0,  metavar='FILE|DIR')
@click.option('--num_steps',              help='Number of optimization steps', type=int, default=500, show_default=True)
@click.option('--num-steps-pti',          help='Number of optimization steps for pivot tuning', type=int, default=500, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=666, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--fps',                    help='Frames per second of final video', default=30, show_default=True)
@click.option('--shapes', type=bool, help='Gen shapes for shape interpolation', default=False, show_default=True)

def run_projection(
    network_pkl: str,
    # target_fname: str,
    target_img:str,
    target_img_back:str,
    target_seg:str,
    idx: int,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    num_steps_pti: int,
    fps: int,
    shapes: bool,
):
    """Project given image to the latent space of pretrained network pickle.

    """
    np.random.seed(seed)
    torch.manual_seed(seed)


    # Render debug output: optional video and projected image and W vector.
    outdir = os.path.join(outdir, os.path.basename(network_pkl), str(idx))
    os.makedirs(outdir, exist_ok=True)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with dnnlib.util.open_url(network_pkl) as fp:
        network_data = legacy.load_network_pkl(fp)
        G = network_data['G_ema'].requires_grad_(False).to(device) # type: ignore
    
    G.rendering_kwargs["ray_start"] = 2.35

    if target_img is not None:
        # we actually do not need the seg in this step
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=target_img, use_labels=True, max_size=None, xflip=False)
        # dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.MaskLabeledDataset', img_path=target_img, seg_path=target_seg, use_labels=True, max_size=None, xflip=False)
        dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        # target_fname = dataset._path + "/" + dataset._image_fnames[idx]
        target_fname = dataset._path + "/" + dataset._image_fnames[idx]
        c = torch.from_numpy(dataset._get_raw_labels()[idx:idx+1]).to(device)
        # print(c)
        # back
        dataset_kwargs_back = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=target_img_back, use_labels=True, max_size=None, xflip=False)
        dataset_back = dnnlib.util.construct_class_by_name(**dataset_kwargs_back) # Subclass of training.dataset.Dataset.
        target_fname_back = dataset_back._path + "/" + dataset_back._image_fnames[idx]
        c_back = torch.from_numpy(dataset_back._get_raw_labels()[idx:idx+1]).to(device)
        # print(c_back)

        print(f"projecting: [{idx}] {target_fname}")
        print(f"camera matrix: {c.shape}")
    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # back
    target_pil_back = PIL.Image.open(target_fname_back).convert('RGB')
    w_, h_ = target_pil_back.size
    s_ = min(w_, h_)
    target_pil_back = target_pil_back.crop(((w_ - s_) // 2, (h_ - s_) // 2, (w_ + s_) // 2, (h_ + s_) // 2))
    target_pil_back = target_pil_back.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8_back = np.array(target_pil_back, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps, c, c_back= project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        target_back=torch.tensor(target_uint8_back.transpose([2, 0, 1]), device=device),
        c=c,
        c_back=c_back,
        num_steps=num_steps,
        device=device,
        verbose=True,
        target_img=target_img,
        target_fname=target_fname
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')
    G_steps = project_pti(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        target_back=torch.tensor(target_uint8_back.transpose([2, 0, 1]), device=device),
        w_pivot=projected_w_steps[-1:],
        c=c,
        c_back= c_back,
        num_steps=num_steps,
        device=device,
        verbose=True,
        target_img=target_img,
        target_fname=target_fname
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]

    with open('/mnt/backup/yiheng/PanoHead/latent/projected_w.pkl', 'wb') as f:
        pickle.dump(projected_w, f)

    G_final = G_steps[-1].to(device)
    synth_image = G_final.synthesis(projected_w.unsqueeze(0).to(device), c=c, noise_mode='const')['image']
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

    #left
    camera_lookat_point_left = torch.tensor([0, 0, 0.0], device=device)
    cam2world_pose_left = LookAtPoseSampler.sample(0, 3.14/2, camera_lookat_point_left, radius=2.7, device=device)
    intrinsics_left = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c_left = torch.cat([cam2world_pose_left.reshape(-1, 16), intrinsics_left.reshape(-1, 9)], 1)
    synth_image_left = G_final.synthesis(projected_w.unsqueeze(0).to(device), c=c_left[0:1], noise_mode='const')['image']
    synth_image_left = (synth_image_left + 1) * (255/2)
    synth_image_left = synth_image_left.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image_left, 'RGB').save(f'{outdir}/proj_left.png')

    #right
    camera_lookat_point_right = torch.tensor([0, 0, 0.0], device=device)
    cam2world_pose_right = LookAtPoseSampler.sample(3.14, 3.14/2, camera_lookat_point_right, radius=2.7, device=device)
    intrinsics_right = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c_right = torch.cat([cam2world_pose_right.reshape(-1, 16), intrinsics_right.reshape(-1, 9)], 1)
    synth_image_right = G_final.synthesis(projected_w.unsqueeze(0).to(device), c=c_right[0:1], noise_mode='const')['image']
    synth_image_right = (synth_image_right + 1) * (255/2)
    synth_image_right = synth_image_right.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image_right, 'RGB').save(f'{outdir}/proj_right.png')

    #back
    synth_image_back = G_final.synthesis(projected_w.unsqueeze(0).to(device), c=c_back[0:1], noise_mode='const')['image']
    synth_image_back = (synth_image_back + 1) * (255/2)
    synth_image_back = synth_image_back.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image_back, 'RGB').save(f'{outdir}/proj_back.png')

    #left_back
    camera_lookat_point_lback = torch.tensor([0, 0, 0.0], device=device)
    cam2world_pose_lback = LookAtPoseSampler.sample(3.14 + 3.14/2 +3.14/4, 3.14/2, camera_lookat_point_lback, radius=2.7, device=device)
    intrinsics_lback = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c_lback = torch.cat([cam2world_pose_lback.reshape(-1, 16), intrinsics_lback.reshape(-1, 9)], 1)
    synth_image_lback = G_final.synthesis(projected_w.unsqueeze(0).to(device), c=c_lback[0:1], noise_mode='const')['image']
    synth_image_lback = (synth_image_lback + 1) * (255/2)
    synth_image_lback = synth_image_lback.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image_lback, 'RGB').save(f'{outdir}/proj_lback.png')

    #right_back
    camera_lookat_point_rback = torch.tensor([0, 0, 0.0], device=device)
    cam2world_pose_rback = LookAtPoseSampler.sample(3.14 + 3.14/4, 3.14/2, camera_lookat_point_rback, radius=2.7, device=device)
    intrinsics_rback = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c_rback = torch.cat([cam2world_pose_rback.reshape(-1, 16), intrinsics_rback.reshape(-1, 9)], 1)
    synth_image_rback = G_final.synthesis(projected_w.unsqueeze(0).to(device), c=c_rback[0:1], noise_mode='const')['image']
    synth_image_rback = (synth_image_rback + 1) * (255/2)
    synth_image_rback = synth_image_rback.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image_rback, 'RGB').save(f'{outdir}/proj_rback.png')


    f_path = os.path.join(outdir, 'front')
    b_path = os.path.join(outdir, 'back')
    os.makedirs(f_path, exist_ok=True)
    os.makedirs(b_path, exist_ok=True)

    camera_lookat_point = torch.tensor([0, 0, 0.0], device=device)
    radius = 2.7

    #left_front
    camera_lookat_point_lfront = torch.tensor([0, 0, 0.0], device=device)
    cam2world_pose_lfront = LookAtPoseSampler.sample(3.14/4, 3.14/2, camera_lookat_point_lfront, radius=2.7, device=device)
    intrinsics_lfront = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c_lfront = torch.cat([cam2world_pose_lfront.reshape(-1, 16), intrinsics_lfront.reshape(-1, 9)], 1)
    synth_image_lfront = G_final.synthesis(projected_w.unsqueeze(0).to(device), c=c_lfront[0:1], noise_mode='const')['image']
    synth_image_lfront = (synth_image_lfront + 1) * (255/2)
    synth_image_lfront = synth_image_lfront.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image_lfront, 'RGB').save(f'{outdir}/proj_lfront.png')

    #right_front
    camera_lookat_point_rfront = torch.tensor([0, 0, 0.0], device=device)
    cam2world_pose_rfront = LookAtPoseSampler.sample(3.14/2 + 3.14/4, 3.14/2, camera_lookat_point_rfront, radius=2.7, device=device)
    intrinsics_rfront = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c_rfront = torch.cat([cam2world_pose_rfront.reshape(-1, 16), intrinsics_rfront.reshape(-1, 9)], 1)
    synth_image_rfront= G_final.synthesis(projected_w.unsqueeze(0).to(device), c=c_rfront[0:1], noise_mode='const')['image']
    synth_image_rfront = (synth_image_rfront + 1) * (255/2)
    synth_image_rfront = synth_image_rfront.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image_rfront, 'RGB').save(f'{outdir}/proj_rfront.png')

    # Save geometry
    max_batch = 10000000
    voxel_resolution = 512
    if shapes:
        # generate shapes
        # print('Generating shape for frame %d / %d ...' % (frame_idx, num_keyframes * w_frames))
        
        samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'])
        samples = samples.to(device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        with tqdm(total = samples.shape[1]) as pbar:
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    sigma = G_final.sample_mixed(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], projected_w.unsqueeze(0).to(device), truncation_psi=1, noise_mode='const')['sigma']
                    sigmas[:, head:head+max_batch] = sigma
                    head += max_batch
                    pbar.update(max_batch)

        sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
        sigmas = np.flip(sigmas, 0)
        
        pad = int(30 * voxel_resolution / 256)
        pad_top = int(38 * voxel_resolution / 256)
        sigmas[:pad] = 0
        sigmas[-pad:] = 0
        sigmas[:, :pad] = 0
        sigmas[:, -pad_top:] = 0
        sigmas[:, :, :pad] = 0
        sigmas[:, :, -pad:] = 0

        output_ply = False
        if output_ply:
            from shape_utils import convert_sdf_samples_to_ply
            convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, 'geometry.ply'), level=10)
        else: # output mrc
            with mrcfile.new_mmap(os.path.join(outdir, 'geometry.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                mrc.data[:] = sigmas

    with open(f'{outdir}/fintuned_generator.pkl', 'wb') as f:
        network_data["G_ema"] = G_final.eval().requires_grad_(False).cpu()
        pickle.dump(network_data, f)
#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
