%  Red-billed Blue Magpie Optimizer (RBMO)
%
%  Source codes demo version 1.0 using matlab2023a 
% 
%  Author and programmer: Shengwei Fu  e-Mail: shengwei_fu@163.com 
%                                                                                                     
%  Paper : Red-billed blue magpie optimizer: a novel metaheuristic algorithm for 2D/3D UAV path planning and engineering design problems
% 
%  Journal : Artificial Intelligence Review
%
%  DOI   : https://doi.org/10.1007/s10462-024-10716-3                                                                                                                                                                                                                                          
%_______________________________________________________________________________________________
% You can simply define your cost function in a seperate file and load its handle to fobj 
% The initial parameters that you need are:
%__________________________________________
% fobj = @YourCostFunction
% D = number of your variables
% T = maximum number of iterations
% N = number of search agents
% Xmin is the lower bound
% Xmax is the upper bound
% func_id is Function Index (F1-F30)
% FES is the number of evaluations
function [fit, X, fit_old, X_old] = Food_storage(fit, X, fit_old, X_old)
    Inx = (fit_old < fit);
    Indx = repmat(Inx, 1, size(X, 2));
    X = Indx .* X_old + ~Indx .* X;
    fit = Inx .* fit_old + ~Inx .* fit;
    fit_old = fit;
    X_old = X;
end