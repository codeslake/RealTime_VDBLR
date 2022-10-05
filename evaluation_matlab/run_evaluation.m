%% init
clear all; close all;

addpath('evaluation_code');
addpath('evaluation_code\image_quality_algorithms');
addpath('evaluation_code\image_quality_algorithms\metrix_mux');
addpath('evaluation_code\image_quality_algorithms\metrix_mux\metrix');
addpath('evaluation_code\image_quality_algorithms\metrix_mux\metrix\ssim');

delete(gcp('nocreate'));
parpool(4);

%%
test_offset = '[data_offset]'; % [config.data_offset] in configs/config.py
result_offset = '[log_offset]\PG2022_RealTime_VDBLR'; % [config.log_offset] in configs/config.py
test_dataset = 'DVD'; % 'nah', 'REDS'
test_model = 'MTU1_DVD';
%%
result_root = 'result\eval\[ckpt_name]\DVD\[datetime]';
folder_result = fullfile(result_offset, test_model, result_root, 'png/output');
[psnr_mean, ssim_mean] = evaluation(test_model, folder_result, test_dataset, test_offset);
fprintf('PSNR: %f, SSIM: %f', psnr_mean, ssim_mean);