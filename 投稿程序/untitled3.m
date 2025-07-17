t=0:0.01:500;
a1=1+cos(pi*t/500);
a2=2*(1-t/500);
plot(t,a1,t,a2,'-.','LineWidth',3);
xlabel('Iteration');
ylabel('a');
legend('a1','a2')
