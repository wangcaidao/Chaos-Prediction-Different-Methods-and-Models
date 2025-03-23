clear;
% Lorenz 系统参数设置
sigma = 10;
rho = 28;
beta = 8/3;

% 时间设置
T = 10000;
dt = 0.01;
sub_dt = 0.01;
sub = sub_dt / dt;
num_steps = T / dt;  % 时间步数

% 初始化状态变量
x = zeros(1, num_steps);
y = zeros(1, num_steps);
z = zeros(1, num_steps);

% 初始条件
x(1) = 1.0;
y(1) = 1.0;
z(1) = 1.0;

% Lorenz 系统的微分方程
lorenz = @(x, y, z) [sigma*(y-x); x*(rho-z)-y; x*y - beta*z];

% 四阶龙格-库塔法求解
for i = 1:num_steps-1
    X = [x(i); y(i); z(i)];
    k1 = lorenz(X(1), X(2), X(3)) * dt;
    k2 = lorenz(X(1) + k1(1)/2, X(2) + k1(2)/2, X(3) + k1(3)/2) * dt;
    k3 = lorenz(X(1) + k2(1)/2, X(2) + k2(2)/2, X(3) + k2(3)/2) * dt;
    k4 = lorenz(X(1) + k3(1), X(2) + k3(2), X(3) + k3(3)) * dt;
    X_next = X + (k1 + 2*k2 + 2*k3 + k4) / 6;
    x(i+1) = X_next(1);
    y(i+1) = X_next(2);
    z(i+1) = X_next(3);
end
u = cat(2, x', y', z');
u = u(0.2*end+1:sub:end,:);
N = 800000;
u = u(end-N+1:end,:);
u = u./max(abs(u));
save(['Time Series/Lorenz_N' num2str(N) '_dt' num2str(dt) 'to' num2str(sub_dt) 's.mat'], 'u');

% 绘制结果
figure(1);
plot3(u(:,1), u(:,2), u(:,3));
grid on;
xlabel('x');
ylabel('y');
zlabel('z');
title('phase diagram');

num_to_plot = 2000;  % 要绘制的点数
figure(2);

% 绘制 x 的时间序列
subplot(3,1,1);
plot(u(end-num_to_plot+1:end,1), 'r');
grid on;
xlabel('Time');
ylabel('x');
title('x(t) - Last 2000 Points');

% 绘制 y 的时间序列
subplot(3,1,2);
plot(u(end-num_to_plot+1:end,2), 'g');
grid on;
xlabel('Time');
ylabel('y');
title('y(t) - Last 2000 Points');

% 绘制 z 的时间序列
subplot(3,1,3);
plot(u(end-num_to_plot+1:end,3), 'b');
grid on;
xlabel('Time');
ylabel('z');
title('z(t) - Last 2000 Points');