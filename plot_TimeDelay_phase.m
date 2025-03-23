clear;
file_name = 'MG-P2P-long-MLP';
load(['pred/' file_name '.mat']);
figure('Position', [200, 200, 1200, 500]);
N0 = 100;	% N0 = tau / h
subplot(1, 2, 1);
plot(pred(1:end-N0),pred(N0+1:end), 'LineWidth', 0.5);
axis([0 1.5 0 1.5]);
set(gca, 'XTick', [0 0.5 1 1.5]);
set(gca, 'YTick', [0 0.5 1 1.5]);

subplot(1, 2, 2);
plot(correct(1:end-N0),correct(N0+1:end),'LineWidth', 0.5);
axis([0 1.5 0 1.5]);
set(gca, 'XTick', [0 0.5 1 1.5]);
set(gca, 'YTick', [0 0.5 1 1.5]);