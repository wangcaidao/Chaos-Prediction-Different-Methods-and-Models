clear
clc

param.tau = 5;      % 延迟时间tau(可修改)
param.b = 6;        % 系数b(可修改)
T_total = 50000;	% 时间序列总长度(可修改)
dt = 0.05;          % 数值模拟时的时间间隔
sub = 1;            % 下采样倍数
N = round(T_total/dt);
x = zeros(1,N);
x(1:param.tau/dt) = rand(1,param.tau/dt);   % 生成随机值用做初始条件
y = Ikeda_RK4(x,dt,N,param);
raw_data = y(1:sub:end);
dt_origin = dt;

clear x y
tau	= param.tau;
b	= param.b;
dt	= dt * sub;
N	= round(T_total/dt);
N0	= round(tau/dt);
N_points = 800000;  % 从后往前取多少个点存成时间序列（因为开头的点还没进入混沌状态，需要舍去。）一般建议存后80%的点，即0.8*N
u = raw_data(end-N_points+1:end)';
save(['Time Series/Ikeda_N' num2str(N_points) '_dt' num2str(dt_origin) 'to' num2str(dt) 's_b' num2str(b) '_tau' num2str(tau) '.mat'], 'u','b','tau','dt','N','N0');
N_plot = 5000;      % 把时间序列的最后N_plot个点画出来（如果把整个时间序列（raw_data）都画出来，会看不清）（可修改）
figure(1);plot(raw_data(end-N_plot+1:end));

% 绘制结果
figure(2);
plot(u(1:end-N0), u(N0+1:end));
grid on;
xlabel('x(t)');
ylabel('x(t+tau)');
title('phase diagram');