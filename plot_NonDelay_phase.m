clear;
file_name = 'Lorenz-NAR-long-MLP';
load(['pred/' file_name '.mat']);
figure('Position', [200, 200, 1000, 500]);

subplot(1, 2, 1);
plot3(pred(:,:,1),pred(:,:,2),pred(:,:,3), 'LineWidth', 0.5);
axis([-1 1 -1 1 0 1]);
view(45, 30);
title('Predict Result');

subplot(1, 2, 2);
plot3(correct(:,:,1),correct(:,:,2),correct(:,:,3),'LineWidth', 0.5);
axis([-1 1 -1 1 0 1]);
view(45, 30);
title('True');