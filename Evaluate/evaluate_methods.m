%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluate Umap Generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc

% result_methods_dir = {'./test_results_DE_2D_t200_path20_nonavg_cascade3/', ...
%                       './test_results_DE_2D_t250_path20_nonavg_cascade3/', ...
%                       './test_results_DE_2D_t300_path20_nonavg_cascade3/', ...
%                       './test_results_DE_2D_t940_path20_nonavg_cascade1/', 
%                      };

% result_methods_dir = {'./test_results_SVCT_2D_t200_path20_nonavg_cascade3/', ...
%                       './test_results_SVCT_2D_t250_path20_nonavg_cascade3/', ...
%                       './test_results_SVCT_2D_t300_path20_nonavg_cascade3/', ...
%                       './test_results_SVCT_2D_t940_path20_nonavg_cascade1/'
%                      };

result_methods_dir = {'./test_results_MRI_2D_t200_path20_nonavg_cascade3/', ...
                      './test_results_MRI_2D_t250_path20_nonavg_cascade3/', ...
                      './test_results_MRI_2D_t300_path20_nonavg_cascade3/', ...
                      './test_results_MRI_2D_t940_path20_nonavg_cascade1/'
                     };

% result_methods_dir = {'./test_results_SVCT_2D_tstart200_tend180_path20_avginv2/', ...
%                       './test_results_SVCT_2D_t200_path20_nonavg/'
%                      };

%% Evaluate Results
l_method = length(result_methods_dir);
l_sample = length(dir(fullfile(result_methods_dir{1},'*_gt.nii')));
% l_sample = 5;

% save to nii & compute the metrics
ssim_all = zeros(l_sample,l_method + 1);
psnr_all = zeros(l_sample,l_method + 1);
nmse_all = zeros(l_sample,l_method + 1);

fname_all = cell(l_sample, 1);

for i = 1:l_method
    fprintf('Method --- ' + string(i) + '\n')
    result_method_dir = result_methods_dir{1};
    
    S = dir(fullfile(result_method_dir, '*_gt.nii'));
    [~, idx] = sort([S.datenum],'ascend');
    S = S(idx);
    
    for j = 1:l_sample
%     for j = 5:6
        fprintf('Sample --- ' + string(j) + '---' + S(j).name + '\n')

        gt_filefullname = fullfile(result_methods_dir{i}, S(j).name);
        input_filefullname = fullfile(result_methods_dir{i}, strrep(S(j).name, '_gt.nii', '_input.nii'));
        prior_filefullname = fullfile(result_methods_dir{i}, strrep(S(j).name, '_gt.nii', '_prior.nii'));
        pred_filefullname = fullfile(result_methods_dir{i}, strrep(S(j).name, '_gt.nii', '_pred.nii'));
        
        vol_gt = niftiread(gt_filefullname);
        vol_input = niftiread(input_filefullname);
        vol_prior = niftiread(prior_filefullname);
        vol_pred = niftiread(pred_filefullname);
        
        infooo = niftiinfo(gt_filefullname);
        diff_vol_pred = (vol_gt - vol_pred);
        diff_vol_prior = (vol_gt - vol_prior);
        
        diff_pred_filefullname = fullfile(result_methods_dir{i}, strrep(S(j).name, '_gt.nii', '_pred_diff.nii'));
        diff_pred_abs_filefullname = fullfile(result_methods_dir{i}, strrep(S(j).name, '_gt.nii', '_pred_diff_abs.nii'));
        diff_prior_filefullname = fullfile(result_methods_dir{i}, strrep(S(j).name, '_gt.nii', '_prior_diff.nii'));
        
        niftiwrite(diff_vol_pred, diff_pred_filefullname, infooo);
        niftiwrite(abs(diff_vol_pred), diff_pred_abs_filefullname, infooo);
        niftiwrite(diff_vol_prior, diff_prior_filefullname, infooo); 

        ssim_all(j, 1) = ssim(vol_gt, vol_prior, "DynamicRange", 1);
        psnr_all(j, 1) = psnr(vol_gt, vol_prior, 1);
%         nmse_all(j, 1) = norm(vol_gt(:) - vol_prior(:))^2 / norm(vol_gt(:))^2;
        nmse_all(j, 1) = mean(abs(vol_gt(:) - vol_prior(:))) * 100;
        
        ssim_all(j, i+1) = ssim(vol_gt, vol_pred, "DynamicRange", 1);
        psnr_all(j, i+1) = psnr(vol_gt, vol_pred, 1);
%         nmse_all(j, i+1) = norm(vol_gt(:) - vol_pred(:))^2 / norm(vol_gt(:))^2; 
        nmse_all(j, i+1) = mean(abs(vol_gt(:) - vol_pred(:))) * 100; 

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
