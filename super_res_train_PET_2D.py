"""
Train a super-resolution model.
combine all noise-levels together for training
train on 2D slices
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import argparse
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from dataloader_scripts.dataset_PET_2D import PET_Train

from guided_diffusion.train_util import TrainLoop
from torch.utils.data import DataLoader


def training_dataloader_wrapper(dataset, args):
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    while True:
        yield from data_loader

def main():
    os.environ['OPENAI_LOGDIR'] = "./model_PET_2D/"  # set the logdir

    args = create_argparser().parse_args()
    args.large_size = 256

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)


    logger.log("training...")
    trainloop=TrainLoop(
        model=model,
        diffusion=diffusion,
        data=None,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    )

    while True:
        logger.log("creating data loader...")
        training_data = PET_Train()

        loader = training_dataloader_wrapper(training_data, args)
        trainloop.data = loader
        trainloop.run_loop()


def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        # batch_size=10,
        batch_size=8,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
