clear all; close all; clc;

period=1;  omega=2*pi/period;
e=[0,1/4,1/2,3/4]; color=['r','g','b','c'];
a=1./(1-e); b=sqrt((1+e)./(1-e));
t=linspace(0,period,1000);
x=zeros(length(t),length(e)); y=zeros(length(t),length(e));
for j=1:length(e)
    for i=1:length(t)
        E=fzero(@(E) E-e(j)*sin(E)-omega*t(i) ,0); % add anonymous function for root finding.  Make use of the variables e(j) and t(i) and omega.
        x(i,j)= a(j)*(e(j)-cos(E)); % assign x-coordinate.  Make use of the variables a(j), e(j) and E.
        y(i,j)= b(j)*sin(E); % assign y-coordinate.  Make use of the variables b(j) and E. 
    end
end
for j=1:length(e)
    plot(x(:,j),y(:,j),color(j),'Linewidth', 2); 
    axis equal; 
    hold on;
end
plot(0,0,'xk') %mark the origin
grid on
xlabel('$x$', 'Interpreter', 'latex', 'FontSize',14)
ylabel('$y$', 'Interpreter', 'latex', 'FontSize',14)
legend('e=0','e=1/4','e=1/2','e=3/4','Location','NorthEast')
title('Planetary Orbits','Interpreter','latex','FontSize',16)