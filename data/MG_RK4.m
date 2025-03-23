function yout=MG_RK4(x, dt, N, param)
a   = param.a;
b   = param.b;
c   = param.c;
tau	= param.tau;
N0  = round(tau/dt);
yout= x;
for i = N0+1:N-1
    nd = round(i-N0);	% t(i)-tau的位置
    xnow = yout(i);     % 此刻的值
    xlag = yout(nd);	%延迟值
    k1 = FuncMG(xnow,xlag,a,b,c);
    k2 = FuncMG(xnow+k1*dt/2,xlag,a,b,c);
    k3 = FuncMG(xnow+k2*dt/2,xlag,a,b,c);
    k4 = FuncMG(xnow+k3*dt,xlag,a,b,c);
    yout(i+1) = xnow + (k1 + 2*k2 + 2*k3 + k4)*dt/6;
end
    function xdot=FuncMG(x, xlag, a, b, c)
        xdot = (a*xlag) / (1+xlag.^c) - b * x;
    end
end