# Cascaded diffusion models for medical image translation 
This is the code for [Cascaded Multi-path Shortcut Diffusion Model for Medical Image Translation](https://www.sciencedirect.com/science/article/pii/S1361841524002251). The code is based on the [guided-diffusion](https://github.com/openai/guided-diffusion) by OpenAI.

## Directory
- dataloader_scripts: the sample dataloader script for training the model. Due to restriction on dataset, we couldn't share these private datasets, but the sample dataloader scripts are provided for reference. 
- Evaluate: the matlab code to evaluate the outputs on NMSE, SSIM, PSNR and uncertainty

## Training the diffusion model and sampling from the diffusion model
- To train a diffusion model, please using the following script:
```
CUDA_VISIBLE_DEVICES=0  python super_res_train_DE_2D.py
```
- To sample from the diffusion model using multi-path cascaded diffusion, please using the following script:
```
CUDA_VISIBLE_DEVICES=0  python super_res_sample_DE_2D.py
```