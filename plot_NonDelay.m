clear;
file_name = 'Lorenz-NAR-short-MLP';
load(['pred/' file_name '.mat']);
rmse_1p     = sqrt(mean((correct(:,1,:) - pred(:,1,:)).^2, 'all'));
rmse_100p	= sqrt(mean((correct(:,1:100,:) - pred(:,1:100,:)).^2, 'all'));
time_steps  = size(correct, 2);
for i = 1:size(pred,1)
    sample_num = i;
    figure('Position', [500, 100, 1000, 500]);
    pred_sample = pred(sample_num, :, :);  % 预测值 predict value
    correct_sample = correct(sample_num, :, :);  % 真值 correct value
    
    pred_x = squeeze(pred_sample(:, :, 1));  % 预测的 x 值 predict x
    pred_y = squeeze(pred_sample(:, :, 2));  % 预测的 y 值 predict y
    pred_z = squeeze(pred_sample(:, :, 3));  % 预测的 z 值 predict z
    
    correct_x = squeeze(correct_sample(:, :, 1));  % 真值的 x 值 correct x
    correct_y = squeeze(correct_sample(:, :, 2));  % 真值的 y 值 correct y
    correct_z = squeeze(correct_sample(:, :, 3));  % 真值的 z 值 correct z
    
    error_x = abs(pred_x - correct_x); % 绝对值误差 absolute error x
    error_y = abs(pred_y - correct_y); % 绝对值误差 absolute error y
    error_z = abs(pred_z - correct_z); % 绝对值误差 absolute error z
    
    % X dimension
    subplot(3, 2, 1);
    plot(1:time_steps, correct_x, 'b-', 1:time_steps, pred_x, 'r-');
    axis([1 1000 -1 1]);
    set(gca, 'YTick', [-1 0 1]);
    xlabel('Time Step');
    ylabel('x(t)');
    
    subplot(3, 2, 2);
    plot(1:time_steps, error_x, 'k-');
    axis([1 1000 0 2]);
    set(gca, 'YTick', [0 1 2]);
    xlabel('Time Step');
    ylabel('AE of x(t)');
    
    % Y dimension
    subplot(3, 2, 3);
    plot(1:time_steps, correct_y, 'b-', 1:time_steps, pred_y, 'r-');
    axis([1 1000 -1 1]);
    set(gca, 'YTick', [-1 0 1]);
    xlabel('Time Step');
    ylabel('y(t)');
    
    subplot(3, 2, 4);
    plot(1:time_steps, error_y, 'k-');
    axis([1 1000 0 2]);
    set(gca, 'YTick', [0 1 2]);
    xlabel('Time Step');
    ylabel('AE of y(t)');
    
    % Z dimension
    subplot(3, 2, 5);
    plot(1:time_steps, correct_z, 'b-', 1:time_steps, pred_z, 'r-');
    axis([1 1000 0 1]);
    set(gca, 'YTick', [0 0.5 1]);
    xlabel('Time Step');
    ylabel('z(t)');
    
    subplot(3, 2, 6);
    plot(1:time_steps, error_z, 'k-');
    axis([1 1000 0 2]);
    set(gca, 'YTick', [0 1 2]);
    xlabel('Time Step');
    ylabel('AE of z(t)');
end