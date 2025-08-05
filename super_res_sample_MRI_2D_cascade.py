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
from dataloader_scripts.dataset_MRI_2D import MRI_Test
from guided_diffusion.gaussian_diffusion import GaussianDiffusion
from scipy.io import savemat


sss = [111]

weight_avg = [0.5, 0.45, 0.4, 0.35, 0.3]

def testing_dataloader_wrapper(dataset):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return data_loader


def main():
    args = create_argparser().parse_args()

    # args.large_size = 256
    # args.t_start = 200
    # args.t_end=None
    # args.num_path = 20
    # args.use_average = False
    # args.average_inv = None
    # args.num_cascade = 3
    # save_path = './test_results_MRI_2D_t200_path20_nonavg_cascade3/'

    # args.large_size = 256
    # args.t_start = 250
    # args.t_end=None
    # args.num_path = 20
    # args.use_average = False
    # args.average_inv = None
    # args.num_cascade = 3
    # save_path = './test_results_MRI_2D_t250_path20_nonavg_cascade3/'

    # args.large_size = 256
    # args.t_start = 300
    # args.t_end=None
    # args.num_path = 20
    # args.use_average = False
    # args.average_inv = None
    # args.num_cascade = 3
    # save_path = './test_results_MRI_2D_t300_path20_nonavg_cascade3/'

    # args.large_size = 256
    # args.t_start = 940
    # args.t_end=None
    # args.num_path = 20
    # args.use_average = False
    # args.average_inv = None
    # args.num_cascade = 1
    # save_path = './test_results_MRI_2D_t940_path20_nonavg_cascade1/'

    # args.large_size = 256
    # args.t_start = 600
    # args.t_end=None
    # args.num_path = 20
    # args.use_average = False
    # args.average_inv = None
    # args.num_cascade = 1
    # save_path = './test_results_MRI_2D_t600_path20_nonavg_cascade1/'

    args.large_size = 256
    args.t_start = 700
    args.t_end=None
    args.num_path = 6
    args.use_average = False
    args.average_inv = None
    args.num_cascade = 1
    save_path = './test_results_MRI_2D_t700_path6_nonavg_cascade1_random111/'


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
    training_data = MRI_Test()
    data = testing_dataloader_wrapper(training_data)

    logger.log("creating samples... number_samples", args.num_samples)

    input_gaussian_noise = th.randn(size=(args.batch_size, 1, args.large_size, args.large_size))

    data_iter = iter(data)

    for i, batch in enumerate(data_iter):
        if i > 10:
            # high_res, model_kwargs = batch
            high_res, model_kwargs, prior = batch

            t_start_th = th.tensor([args.t_start])

            for i_cas in range(args.num_cascade):
                th.manual_seed(sss[i_cas])

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
                print("cascade - ", i_cas)

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

                prior_new = th.mean(sample, 0, True).clone()
                
                img_pred_mean = np.squeeze(np.mean(sample.detach().cpu().numpy(), 0))
                img_pred_std = np.squeeze(np.std(sample.detach().cpu().numpy(), 0))
                img_gt = np.squeeze(high_res.detach().cpu().numpy())
                img_input = np.squeeze(model_kwargs['low_res'].detach().cpu().numpy())
                img_prior = np.squeeze(prior.detach().cpu().numpy())
                img_prior_diffused = np.squeeze(prior_diffused.detach().cpu().numpy())

                img_pred_nib = nib.Nifti1Image(img_pred_mean, affine=np.eye(4))
                img_pred_fullfile = os.path.join(save_path, str(i) + "_img_pred_cas" + str(i_cas) + ".nii")
                nib.save(img_pred_nib, img_pred_fullfile)

                img_pred_std_nib = nib.Nifti1Image(img_pred_std, affine=np.eye(4))
                img_pred_std_fullfile = os.path.join(save_path, str(i) + "_img_pred_std_cas" + str(i_cas) + ".nii")
                nib.save(img_pred_std_nib, img_pred_std_fullfile)

                if i_cas+1 == args.num_cascade:
                    img_pred_nib = nib.Nifti1Image(img_pred_mean, affine=np.eye(4))
                    img_pred_fullfile = os.path.join(save_path, str(i) + "_img_pred.nii")
                    nib.save(img_pred_nib, img_pred_fullfile)

                    img_pred_std_nib = nib.Nifti1Image(img_pred_std, affine=np.eye(4))
                    img_pred_std_fullfile = os.path.join(save_path, str(i) + "_img_pred_std.nii")
                    nib.save(img_pred_std_nib, img_pred_std_fullfile)

                img_gt_nib = nib.Nifti1Image(img_gt, affine=np.eye(4))
                img_gt_fullfile = os.path.join(save_path, str(i) + "_img_gt.nii")
                nib.save(img_gt_nib, img_gt_fullfile)

                img_input_nib = nib.Nifti1Image(img_input, affine=np.eye(4))
                img_input_fullfile = os.path.join(save_path, str(i) + "_img_input.nii")
                nib.save(img_input_nib, img_input_fullfile)

                if i_cas == 0:
                    img_prior_nib = nib.Nifti1Image(img_prior, affine=np.eye(4))
                    img_prior_fullfile = os.path.join(save_path, str(i) + "_img_prior.nii")
                    nib.save(img_prior_nib, img_prior_fullfile)

                    img_prior_diffused_nib = nib.Nifti1Image(img_prior_diffused, affine=np.eye(4))
                    img_prior_diffused_fullfile = os.path.join(save_path, str(i) + "_img_prior_diffused.nii")
                    nib.save(img_prior_diffused_nib, img_prior_diffused_fullfile)

                else:
                    img_prior_nib = nib.Nifti1Image(img_prior, affine=np.eye(4))
                    img_prior_fullfile = os.path.join(save_path, str(i) + "_img_prior_cas" + str(i_cas+1) + ".nii")
                    nib.save(img_prior_nib, img_prior_fullfile)

                    img_prior_diffused_nib = nib.Nifti1Image(img_prior_diffused, affine=np.eye(4))
                    img_prior_diffused_fullfile = os.path.join(save_path, str(i) + "_img_prior_diffused_cas" + str(i_cas+1) + ".nii")
                    nib.save(img_prior_diffused_nib, img_prior_diffused_fullfile)

                # mdic = {"img_pred": img_pred, "img_gt": img_gt, "img_input": img_input, "img_prior": img_prior, 
                # "img_prior_diffused": img_prior_diffused,
                # "sample": sample.detach().cpu().numpy()}

                # savemat(os.path.join(save_path, str(i) + "_results.mat"), mdic)

                if i_cas == 0:
                    prior_init = prior.clone()
                prior = (1 - weight_avg[i_cas]) * prior_new.detach().cpu() + weight_avg[i_cas] * prior_init
                # prior = (prior_new.detach().cpu() + prior.clone()) / 2


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=1000,
        batch_size=1,
        use_ddim=True,
        base_samples="",
        # model_path="./model_MRI_2D/model300000.pt",
        model_path="./model_MRI_2D/model280000.pt",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
