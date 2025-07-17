dim=10;
x=zeros(10,10);
for i=1:dim
    x(i,:)=CauchyInverCumDist(dim);
end
function x=CauchyInverCumDist(dim)
%% Cauchy inverse cumulative distribution
% dim: 维数
a=1;
b=0.01;

p=randn(1,dim);
x=a+b*tan(pi*(p-1/2));

end