
function x=CauchyInverCumDist(dim)
%% Cauchy inverse cumulative distribution
% dim: Î¬Êý
a=1;
b=0.01;

p=randn(1,dim);
x=a+b*tan(pi*(p-1/2));

end