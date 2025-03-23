clear;
file_name = 'Ikeda-onestep-FNO';
load(['pred/' file_name '.mat']);
rmse_1p     = sqrt(mean((correct(:,1,:) - pred(:,1,:)).^2, 'all'));
rmse_100p	= sqrt(mean((correct(:,1:100,:) - pred(:,1:100,:)).^2, 'all'));
time_steps  = size(correct, 2);
for i = 1:size(pred,1)
    sample_num = i;
    figure('Position', [500, 100, 400, 300]); % [left, bottom, width, height]
    pred_sample = pred(sample_num, :);          % 预测值 predict value
    correct_sample = correct(sample_num, :);	% 真值 correct value
    error = abs(pred_sample - correct_sample);	% 绝对值误差 absolute error
    
    subplot(2,1,1);
    plot(1:time_steps, correct_sample, 'b-', 1:time_steps, pred_sample, 'r-');
    xlabel('Time Step');
    ylabel('x(t)');    
    
    subplot(2,1,2);
    plot(1:time_steps, error, 'k-');
    xlabel('Time Step');
    ylabel('AE of x(t)');
end