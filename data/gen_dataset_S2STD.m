clear;
load('Time Series/MG_N800000_dt1to1s_a0.2_b0.1_c10_tau100.mat');
dim = size(u, 2);
tau = N0; % N0 is the delayed time steps. If there is no N0 in your own data, you need to input the delay time steps here.
data_length = 200;      % window size L
n_data_dataset = 20000; % training sets
n_val_dataset = 100;	% validation sets
n_test_dataset = 10;    % short_term test sets
test_length = 2000;     % short_term test length
test_long = 100000;     % long_term test length
L = length(u);

n_all = n_data_dataset + n_val_dataset;

all_ic = zeros(n_all, data_length, dim);	% inputs
all_out = zeros(n_all, data_length, dim);   % labels

% Make training sets & validation sets
for i = 1:n_all
    start_idx	= i;
    end_idx     = start_idx + data_length - 1;
    all_ic(i, :, :)     = u(start_idx:end_idx, :);
    all_out(i, :, :)	= u(start_idx+tau:end_idx+tau, :);
end
train_end = end_idx + 1;

% Make training sets & validation sets
data_ic = all_ic(1:n_data_dataset, :, :);           % 训练集输入 training inputs
data_out = all_out(1:n_data_dataset, :, :);         % 训练集标签 training labels
val_ic = all_ic(n_data_dataset + 1:n_all, :, :);    % 评估集输入 validation inputs
val_out = all_out(n_data_dataset + 1:n_all, :, :);	% 评估集标签 validation labels

% Make long_term test sets
long_start = train_end + 5000; % 防止测试集和训练集混叠 Preventing overlap between the training sets and the test sets
test_ic(1, :, :) = u(long_start-data_length+1:long_start, :);
test_out(1, :, :) = u(long_start + 1:long_start + test_long, :);
save('Datasets/MG_test_long.mat','test_ic','test_out');

% Make short_term test sets
test_ic = zeros(n_test_dataset, data_length, dim);	% inputs
test_out = zeros(n_test_dataset, test_length, dim); % labels

% 从时间序列的末尾往前取 Take the test sets from the end of the time series moving backward
for i = 1:n_test_dataset
    start_idx	= L - 2 * i * test_length + 1;
    test_ic(i, :, :)	= u(start_idx - data_length:start_idx - 1, :);
    test_out(i, :, :)	= u(start_idx:start_idx + test_length - 1, :);
end
test_start = start_idx;
assert(test_start > train_end, 'You need to ensure test_start > train_end to prevent overlap between the training sets and the test sets');

save('Datasets/MG_S2STD_train.mat','data_ic','data_out','val_ic','val_out');
save('Datasets/MG_test_short.mat','test_ic','test_out');