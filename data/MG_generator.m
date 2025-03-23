clear
clc

param.tau = 100;     % 延迟时间tau(可修改)
param.a = 0.2;        % 系数a(可修改)
param.b = 0.1;        % 系数b(可修改)
param.c = 10;       % 系数c(可修改)
T_total = 1000000;    % 时间序列总长度(可修改)
dt = 1;           % 数值模拟时的时间间隔
sub = 1;            % 下采样倍数
N = round(T_total/dt);
x = zeros(1,N);
x(1:param.tau/dt) = rand(1,param.tau/dt);   % 生成随机值用做初始条件
y = MG_RK4(x,dt,N,param);
raw_data = y(1:sub:end);
dt_origin = dt;

clear x y
tau = param.tau;
a = param.a;
b = param.b;
c = param.c;
dt = dt * sub;
N  = round(T_total/dt);
N0 = round(tau/dt);
N_points = 800000;   % 从后往前取多少个点存成时间序列（因为开头的点还没进入混沌状态，需要舍去。）一般建议存后80%的点，即0.8*N
u = raw_data(end-N_points+1:end)';
save(['Time Series/MG_N' num2str(N_points) '_dt' num2str(dt_origin) 'to' num2str(dt) 's_a' num2str(a) '_b' num2str(b) '_c' num2str(c) '_tau' num2str(tau) '.mat'], 'u','a','b','c','tau','dt','N','N0');
N_plot = 1000;  % 把时间序列的最后N_plot个点画出来（如果把整个时间序列（raw_data）都画出来，会看不清）（可修改）
figure(1);plot(raw_data(end-N_plot+1:end));

% 绘制结果
figure(2);
plot(u(1:end-N0), u(N0+1:end));
grid on;
xlabel('x(t)');
ylabel('x(t+tau)');
title('phase diagram');