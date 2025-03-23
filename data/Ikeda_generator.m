clear
clc

param.tau = 5;      % �ӳ�ʱ��tau(���޸�)
param.b = 6;        % ϵ��b(���޸�)
T_total = 50000;	% ʱ�������ܳ���(���޸�)
dt = 0.05;          % ��ֵģ��ʱ��ʱ����
sub = 1;            % �²�������
N = round(T_total/dt);
x = zeros(1,N);
x(1:param.tau/dt) = rand(1,param.tau/dt);   % �������ֵ������ʼ����
y = Ikeda_RK4(x,dt,N,param);
raw_data = y(1:sub:end);
dt_origin = dt;

clear x y
tau	= param.tau;
b	= param.b;
dt	= dt * sub;
N	= round(T_total/dt);
N0	= round(tau/dt);
N_points = 800000;  % �Ӻ���ǰȡ���ٸ�����ʱ�����У���Ϊ��ͷ�ĵ㻹û�������״̬����Ҫ��ȥ����һ�㽨����80%�ĵ㣬��0.8*N
u = raw_data(end-N_points+1:end)';
save(['Time Series/Ikeda_N' num2str(N_points) '_dt' num2str(dt_origin) 'to' num2str(dt) 's_b' num2str(b) '_tau' num2str(tau) '.mat'], 'u','b','tau','dt','N','N0');
N_plot = 5000;      % ��ʱ�����е����N_plot���㻭���������������ʱ�����У�raw_data�������������ῴ���壩�����޸ģ�
figure(1);plot(raw_data(end-N_plot+1:end));

% ���ƽ��
figure(2);
plot(u(1:end-N0), u(N0+1:end));
grid on;
xlabel('x(t)');
ylabel('x(t+tau)');
title('phase diagram');