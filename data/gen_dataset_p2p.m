clear;
load('Time Series/Rossler_N800000_dt0.01to0.1s.mat');  % your input data

n_train = 20000;        % training sets
n_val = 100;            % validation sets
n_test = 10;            % short_term test sets
test_length = 2000;     % short_term test length
test_long = 100000;   % long_term test length
L = length(u);

all_ic = u(1 : n_train + n_val,:);      % inputs
all_out = u(2 : n_train + n_val + 1,:); % labels

% Make training sets & validation sets
data_ic = all_ic(1:n_train, :);    % 训练集输入 training inputs
data_out = all_out(1:n_train, :);  % 训练集标签 training labels
val_ic = all_ic(n_train + 1:n_train + n_val, :);   % 评估集输入 validation inputs
val_out = all_out(n_train + 1:n_train + n_val, :); % 评估集标签 validation labels
train_end = n_train + n_val + 1;

% Make long_term test sets
long_start = n_train + n_val + 5000; % 防止测试集和训练集混叠 Preventing overlap between the training sets and the test sets
test_ic(1, :) = u(long_start, :);
test_out(1, :, :) = u(long_start + 1:long_start + test_long, :);
save('data/Rossler_P2P_test_long.mat','test_ic','test_out');

% Make short_term test sets
test_ic = zeros(n_test, 3);                 % inputs
test_out = zeros(n_test, test_length, 3);	% labels

% 从时间序列的末尾往前取 Take the test sets from the end of the time series moving backward
for i = 1:n_test
    start_idx = L - 2 * i * test_length + 1;
    test_ic(i, :)       = u(start_idx - 1, :);
    test_out(i, :, :)	= u(start_idx:start_idx + test_length - 1, :);
end
test_start = start_idx;
assert(test_start > n_train + n_val, 'You need to ensure test_start > train_end to prevent overlap between the training sets and the test sets');

save(['Datasets/Rossler_P2P_train_N' num2str(n_train) '.mat'],'data_ic','data_out','val_ic','val_out');
save('Datasets/Rossler_P2P_test_short.mat','test_ic','test_out');
