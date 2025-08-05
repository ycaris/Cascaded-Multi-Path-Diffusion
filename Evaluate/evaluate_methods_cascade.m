%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluate Umap Generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc

result_methods_dir = {'./test_results_SVCT_2D_t300_path20_nonavg_cascade3/'
                     };

%% Evaluate Results
l_sample = length(dir(fullfile(result_methods_dir{1},'*_gt.nii')));
l_sample = 1;

% save to nii & compute the metrics
ssim_all = zeros(l_sample,3 + 1);
psnr_all = zeros(l_sample,3 + 1);
nmse_all = zeros(l_sample,3 + 1);

fname_all = cell(l_sample, 1);

for i = 0:2
    fprintf('Cascade --- ' + string(i) + '\n')
    result_method_dir = result_methods_dir{1};
    
    S = dir(fullfile(result_method_dir, '*_gt.nii'));
    [~, idx] = sort([S.datenum],'ascend');
    S = S(idx);
    
    for j = 1:l_sample
        fprintf('Sample --- ' + string(j) + '---' + S(j).name + '\n')

        gt_filefullname = fullfile(result_method_dir, S(j).name);
        input_filefullname = fullfile(result_method_dir, strrep(S(j).name, '_gt.nii', '_input.nii'));
        prior_filefullname = fullfile(result_method_dir, strrep(S(j).name, '_gt.nii', '_prior.nii'));
        if i == 0
            pred_filefullname = fullfile(result_method_dir, strrep(S(j).name, '_gt.nii', '_pred_cas0.nii'));
        end
        if i == 1
            pred_filefullname = fullfile(result_method_dir, strrep(S(j).name, '_gt.nii', '_pred_cas1.nii'));
        end
        if i == 2
            pred_filefullname = fullfile(result_method_dir, strrep(S(j).name, '_gt.nii', '_pred_cas2.nii'));
        end
        
        vol_gt = niftiread(gt_filefullname);
        vol_input = niftiread(input_filefullname);
        vol_prior = niftiread(prior_filefullname);
        vol_pred = niftiread(pred_filefullname);
        
        diff_vol_pred = (vol_gt - vol_pred);
        diff_vol_prior = (vol_gt - vol_prior);
        
        diff_pred_filefullname = fullfile(result_method_dir, strrep(S(j).name, '_gt.nii', '_pred_diff'));
        diff_prior_filefullname = fullfile(result_method_dir, strrep(S(j).name, '_gt.nii', '_prior_diff'));
        
%         niftiwrite(diff_vol_pred, diff_pred_filefullname);
%         niftiwrite(diff_vol_prior, diff_prior_filefullname); 
        
        r = 1.2;
        ssim_all(j, 1) = ssim(vol_gt, vol_prior, "DynamicRange", r);
        psnr_all(j, 1) = psnr(vol_gt, vol_prior, r);
        nmse_all(j, 1) = norm(vol_gt(:) - vol_prior(:))^2 / norm(vol_gt(:))^2;
        
        ssim_all(j, i+2) = ssim(vol_gt, vol_pred, "DynamicRange", r);
        psnr_all(j, i+2) = psnr(vol_gt, vol_pred, r);
        nmse_all(j, i+2) = norm(vol_gt(:) - vol_pred(:))^2 / norm(vol_gt(:))^2; 

        fname_all{j, 1} = S(j).name;

    end
end

fprintf('RMSE -- \n')
mean(nmse_all,1)
std(nmse_all,1)

fprintf('SSIM -- \n')
mean(ssim_all,1)
std(ssim_all,1)

fprintf('PSNR -- \n')
mean(psnr_all,1)
std(psnr_all,1)

% save('./eval_results_drf10_FDG.mat', 'nmse_all', 'ssim_all', 'psnr_all', 'fname_all')
% save('./eval_results_drf25_FDG.mat', 'nmse_all', 'ssim_all', 'psnr_all', 'fname_all')
% save('./eval_results_drf10_DOTA.mat', 'nmse_all', 'ssim_all', 'psnr_all', 'fname_all')
% save('./eval_results_drf25_DOTA.mat', 'nmse_all', 'ssim_all', 'psnr_all', 'fname_all')
