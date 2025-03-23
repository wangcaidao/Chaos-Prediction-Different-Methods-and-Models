function yout=Ikeda_RK4(x, dt, N, param)
b	= param.b;
tau	= param.tau;
N0  = round(tau/dt);
yout= x;
for i = N0+1:N-1
    nd = round(i-N0);	%	t(i)-tau的位置
    xnow = yout(i);     %	此刻的值
    xlag = yout(nd);    %   延迟值
    k1 = FuncIkeda(xnow,xlag,b);
    k2 = FuncIkeda(xnow+k1*dt/2,	xlag, b);
    k3 = FuncIkeda(xnow+k2*dt/2,	xlag, b);
    k4 = FuncIkeda(xnow+k3*dt,      xlag, b);
    yout(i+1) = xnow + (k1 + 2*k2 + 2*k3 + k4)*dt/6;
end
    function xdot=FuncIkeda(x, xlag, b)
        xdot = b * sin(xlag) - x;
    end
end