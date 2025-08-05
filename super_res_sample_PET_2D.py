"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os
import os.path

import nibabel as nib
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import pdb

from guided_diffusion import dist_util, logger
from guided_diffusion import gaussian_diffusion
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from torch.utils.data import DataLoader
from dataloader_scripts.dataset_PET_2D import PET_Test
from guided_diffusion.gaussian_diffusion import GaussianDiffusion
from scipy.io import savemat

th.manual_seed(66)

def testing_dataloader_wrapper(dataset):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return data_loader


def main():
    args = create_argparser().parse_args()

    # args.large_size = 256
    # args.t_start = 200
    # args.num_path = 20
    # args.use_average = True
    # save_path = './test_results_PET_2D_t200_path20_avg/'

    # args.large_size = 256
    # args.t_start = 400
    # args.num_path = 20
    # args.use_average = True
    # save_path = './test_results_PET_2D_t400_path20_avg/'

    # args.large_size = 256
    # args.t_start = 600
    # args.num_path = 20
    # args.use_average = True
    # save_path = './test_results_PET_2D_t600_path20_avg/'

    args.large_size = 256
    args.t_start = 200
    args.num_path = 20
    args.use_average = False
    save_path = './test_results_PET_2D_t200_path20_nonavg/'

    # args.large_size = 256
    # args.t_start = 200
    # args.num_path = 1
    # args.use_average = False
    # save_path = './test_results_PET_2D_t200_path1_nonavg/'

    # args.large_size = 256
    # args.t_start = 940
    # args.num_path = 20
    # args.use_average = False
    # save_path = './test_results_PET_2D_t940_path20_nonavg/'

    # args.large_size = 256
    # args.t_start = 940
    # args.num_path = 3
    # args.use_average = False
    # args.t_start_average = 30
    # args.average_inv = None
    # save_path = './test_results_PET_2D_t940_path2_nonavg/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    training_data = PET_Test()
    data = testing_dataloader_wrapper(training_data)

    logger.log("creating samples... number_samples", args.num_samples)

    input_gaussian_noise = th.randn(size=(args.batch_size, 1, args.large_size, args.large_size))

    data_iter = iter(data)

    for i, batch in enumerate(data_iter):
        # high_res, model_kwargs = batch
        high_res, model_kwargs, prior = batch

        t_start_th = th.tensor([args.t_start])

        for ii in range(args.num_path):
            if ii == 0:
                prior_diffused = diffusion.q_sample(prior.clone(), t=t_start_th, noise=input_gaussian_noise).cuda()
            else:
                noise_flattened = input_gaussian_noise.view(-1)
                perm = th.randperm(noise_flattened.nelement())
                noise_flattened_shuffled = noise_flattened[perm]
                input_gaussian_noise_shuffled = noise_flattened_shuffled.view_as(input_gaussian_noise)
                prior_diffused = th.cat((prior_diffused, diffusion.q_sample(prior.clone(), t=t_start_th, noise=input_gaussian_noise_shuffled).cuda()), 0)

        print("testing case - ", i, ' - ', model_kwargs['low_res'].shape)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample = diffusion.ddim_sample_loop(
            model,
            (args.batch_size, 1, args.large_size, args.large_size),
            noise=prior_diffused,
            start_t=args.t_start,
            end_t=args.t_end,
            num_path=args.num_path,
            clip_denoised=args.clip_denoised,
            use_average=args.use_average,
            average_inv=args.average_inv,
            model_kwargs=model_kwargs,
        )

        img_pred = np.squeeze(np.mean(sample.detach().cpu().numpy(), 0))
        img_gt = np.squeeze(high_res.detach().cpu().numpy())
        img_input = np.squeeze(model_kwargs['low_res'].detach().cpu().numpy())
        img_prior = np.squeeze(prior.detach().cpu().numpy())
        img_prior_diffused = np.squeeze(prior_diffused.detach().cpu().numpy())

        img_pred_nib = nib.Nifti1Image(img_pred, affine=np.eye(4))
        img_pred_fullfile = os.path.join(save_path, str(i) + "_img_pred.nii")
        nib.save(img_pred_nib, img_pred_fullfile)

        img_gt_nib = nib.Nifti1Image(img_gt, affine=np.eye(4))
        img_gt_fullfile = os.path.join(save_path, str(i) + "_img_gt.nii")
        nib.save(img_gt_nib, img_gt_fullfile)

        img_input_nib = nib.Nifti1Image(img_input, affine=np.eye(4))
        img_input_fullfile = os.path.join(save_path, str(i) + "_img_input.nii")
        nib.save(img_input_nib, img_input_fullfile)

        img_prior_nib = nib.Nifti1Image(img_prior, affine=np.eye(4))
        img_prior_fullfile = os.path.join(save_path, str(i) + "_img_prior.nii")
        nib.save(img_prior_nib, img_prior_fullfile)

        img_prior_diffused_nib = nib.Nifti1Image(img_prior_diffused, affine=np.eye(4))
        img_prior_diffused_fullfile = os.path.join(save_path, str(i) + "_img_prior_diffused.nii")
        nib.save(img_prior_diffused_nib, img_prior_diffused_fullfile)

        mdic = {"img_pred": img_pred, "img_gt": img_gt, "img_input": img_input, "img_prior": img_prior, 
        "img_prior_diffused": img_prior_diffused,
        "sample": sample.detach().cpu().numpy()}

        savemat(os.path.join(save_path, str(i) + "_results.mat"), mdic)


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=1000,
        batch_size=1,
        use_ddim=True,
        base_samples="",
        model_path="./model_PET_2D/model400000.pt",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
