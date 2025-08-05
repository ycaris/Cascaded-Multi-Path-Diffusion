%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluate Umap Generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc

% result_methods_dir = {'./test_results_DE_2D_t200_path20_nonavg_cascade3/'
%                      };

result_methods_dir = {'./test_results_MRI_2D_t200_path20_nonavg_cascade3/'
                     };

% result_methods_dir = {'./test_results_SVCT_2D_t200_path20_nonavg_cascade3/'
%                      };

%% Evaluate Results
l_method = length(result_methods_dir);
l_sample = length(dir(fullfile(result_methods_dir{1},'*_gt.nii')));
% l_sample = 5;

% save to nii & compute the metrics
R_all = zeros(l_sample,l_method);

fname_all = cell(l_sample, 1);

for i = 1:l_method
    fprintf('Method --- ' + string(i) + '\n')
    result_method_dir = result_methods_dir{1};
    
    S = dir(fullfile(result_method_dir, '*_gt.nii'));
    [~, idx] = sort([S.datenum],'ascend');
    S = S(idx);
    
    for j = 1:l_sample
%     for j = 7:7
        fprintf('Sample --- ' + string(j) + '---' + S(j).name + '\n')

        gt_filefullname = fullfile(result_methods_dir{i}, S(j).name);
        input_filefullname = fullfile(result_methods_dir{i}, strrep(S(j).name, '_gt.nii', '_input.nii'));
        prior_filefullname = fullfile(result_methods_dir{i}, strrep(S(j).name, '_gt.nii', '_prior.nii'));
        pred_filefullname = fullfile(result_methods_dir{i}, strrep(S(j).name, '_gt.nii', '_pred.nii'));
        pred_std_filefullname = fullfile(result_methods_dir{i}, strrep(S(j).name, '_gt.nii', '_pred_std.nii'));
        
        vol_gt = niftiread(gt_filefullname);
        vol_input = niftiread(input_filefullname);
        vol_prior = niftiread(prior_filefullname);
        vol_pred = niftiread(pred_filefullname);
        vol_pred_std = niftiread(pred_std_filefullname); 
        
        infooo = niftiinfo(gt_filefullname);
        diff_vol_pred = (vol_gt - vol_pred);
        diff_vol_prior = (vol_gt - vol_prior);
        
        figure,
        x = vol_pred_std(find(abs(diff_vol_pred) > 0.010));
        y = diff_vol_pred(find(abs(diff_vol_pred) > 0.010)) * 0.75;
%         x = vol_pred_std;
%         y = abs(diff_vol_pred) * 0.5;
        x = x(:);
        y = abs(y(:));
        plot(x, y, 'ro', 'MarkerSize', 4, 'LineWidth', 2);
        grid on;
%         fontSize = 20;
%         xlabel('X', 'FontSize', fontSize);
%         ylabel('Y', 'FontSize', fontSize);
%         title('Linear Fit', 'FontSize', fontSize);
        
        linearCoefficients = polyfit(x, y, 1);
        xFit = linspace(0, 0.13, 2);
        yFit = polyval(linearCoefficients, xFit);
        hold on;
%         plot(xFit, yFit, 'b-', 'MarkerSize', 15, 'LineWidth', 1);
        plot(xFit, yFit, 'b', 'LineWidth', 5);
%         legend('Training Set', 'Fit', 'Location', 'Northwest');
        
        R = corrcoef(x,y);
        R_all(j,1) = R(2,1);
        

    end
end
