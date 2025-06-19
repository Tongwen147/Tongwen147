% %% 反向+莱维自适应+差分进化+精英引导策略
% function [T,L,BestSol , BestCost]=HALA2(param   , model ,  CostFunction,dim)
% tic
% nVar = model.nVar ;       %  节点个数
% VarSize=[1 nVar];   %  决策变量维度
% 
% VarMin = 0 ; % 自变量取值   低阶
% VarMax= 1 ;% 自变量取值   高阶
% 
% nPop =  param.nPop  ; % 种群规模
% MaxIt =  param.MaxIt ; % 最大迭代次数
% X=zeros(nPop,dim);
% empty_bee.Position=[];
% empty_bee.Cost=[];
% empty_bee.sol=[];
%  vec_flag=[1,-1]; %% Directional flag 方向旗            
% % Initialize Population Array 
% pop=repmat(empty_bee,nPop,1);
% 
% 
% 
% % Initialize Best Solution Ever Found
% BestSol.Cost=inf;
% % Create Initial Population
% for i=1:nPop
%     pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
%     [ pop(i).Cost,  pop(i).sol ] =CostFunction(pop(i).Position);
%     if pop(i).Cost<=BestSol.Cost
%         BestSol=pop(i);
%     end
% end
% 
% %% 反向学习
% for i = 1 : nPop
% Y(i,:)=pop(i,:).Position;
%  X_opposite(i,:)=(VarMax+VarMin)-Y(i,:);
% end
% X_combined = [Y; X_opposite];
% [~, idx] = sort(rand(size(X_combined, 1), 1)); 
%     S = X_combined(idx(1:nPop), :);
% for i = 1 : nPop
% pop(i,:).Position=S(i,:);
% end
% %% 反向学习
% % Array to Hold Best Cost Values
% BestCost=zeros(MaxIt,1);
% 
% for it=1:MaxIt
% RB=randn(nPop,dim);  % Brownian motion布朗运动
%     F=vec_flag(floor(2*rand()+1)); % Random directional flag随机方向旗
%     theta=2*atan(1-it/MaxIt); % Time-varying parameter时变参数(从 pi/2单调递减到 0)非线性
%    pFit =  [pop.Cost] ; %  每个解的 目标函数值
%     
%     newpop=repmat(empty_bee,nPop,1);
%      for i = 1 : nPop
%        newpop( i  ) = pop( i  );
%      end
% 
%         
%         [ ~ , sortIndex ] = sort(     pFit   );
%         second_best=newpop( sortIndex(2),: ).Position;
%         third_best=newpop( sortIndex(3),: ).Position;
%         [ fMin, bestI ] = min( pFit );      % fMin denotes the global optimum fitness value
%          bestX = newpop( bestI, : ).Position;             % bestX denotes the global optimum position corresponding to fMin 
%   %% 以上无误
% for i=1:nPop
%   %  for j=1:dim
% %for i=1:nPop
% E=2*log(1/rand)*theta;
% if E>1
%  %% 探索
% if rand<0.3
%                r1 = 2 * rand(1,dim) - 1;
% % newpop( i,:   ).Position =  bestX  +F.*RB(i,:).*(r1.*( bestX-newpop( i,:   ).Position))+(1-r1).* (newpop( i,:   ).Position-newpop(randi (nPop),:   ).Position);%eq1长距离迁徙          
%              newpop( i,:   ).Position=0.5*rand*(bestX-newpop( i,:   ).Position)+0.5*rand*(newpop(randi (nPop),:   ).Position-newpop( i,:   ).Position)
%   newpop( i,:   ).Position =  VarMin + mod(     newpop(  i ).Position , VarMax-VarMin  ) ;
% else
%                r2 = rand ()* (1 + sin(0.5 * it));
%                newpop( i,:   ).Position =  newpop( i,:   ).Position+F.* r2*(bestX-newpop(randi (nPop),:   ).Position);%eq2挖洞策略
%  newpop( i,:   ).Position =  VarMin + mod(     newpop(  i ).Position , VarMax-VarMin  ) ;
%                %display(newpop( i,:   ).Position)
%                %F.* r2*(bestX-X(randi(nPop),:));%eq2挖洞策略
%           end
%         else
%             %% 开发
% if rand<0.5
%                radius = sqrt(sum((bestX- newpop( i,:   ).Position).^2));
%                r3=rand();
%                spiral=radius*(sin(2*pi*r3)+cos(2*pi*r3));
%               % newpop( i,:   ).Position=bestX+ F.* newpop( i,:   ).Position.*spiral*rand;%eq3觅食
%              %% EGLP精英扰动
% s = 0.1 * (1 - it/MaxIt);  % 逐步减小扰动幅度
% newpop( i,:   ).Position = bestX + s * randn(1,dim); % 以精英个体为中心进行局部扰动
% % for i=1:nPop
% %     pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
% %     [ pop(i).Cost,  pop(i).sol ] =CostFunction(pop(i).Position);
% %     if pop(i).Cost<=BestSol.Cost
% %         BestSol=pop(i);
% %     end
% % end
% 
% 
%              %% EGLPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
%                 newpop( i,:   ).Position =  VarMin + mod(     newpop(  i ).Position , VarMax-VarMin  ) ;% Xnew(i,:) =bestX + F.* X(i,:).*spiral*rand;%eq3觅食
%            else
%                G=2*(sign(rand-0.5))*(1-it/MaxIt);   
%                newpop( i,:   ).Position = bestX + F.* G*Levy(dim).* (bestX - newpop( i,:   ).Position) ;%eq4躲避天敌
%                %%%
%              newpop( i,:   ).Position =  VarMin + mod(     newpop(  i ).Position , VarMax-VarMin  ) ;
%                % Xnew(i,:) = bestX + F.* G*Levy(dim).* (bestX - X(i,:)) ;%eq4躲避天敌
%             end
% % newpop( i,:   ).Position =  VarMin + mod(     newpop(  i ).Position , VarMax-VarMin  ) ;
% 
% end
% %end
% end
%            
%     %%  Boundary check and evaluation
%   
%  for i = 1 : nPop
%       [  newpop( i   ).Cost  ,   newpop(  i   ).sol ]=CostFunction(     newpop(  i  ).Position );
%             
%         if    newpop( i  ).Cost < pop( i  ).Cost
%             pop(i)  =    newpop( i  ) ;
%             
%             if pop( i  ).Cost<BestSol.Cost
%                 BestSol =    pop(i)  ;
%             end
%         end
% 
%     end
%     L=BestSol.sol.Jpath;
%     
%     % Store Best Cost Ever Found
%     BestCost(it)=BestSol.Cost;
%     
%     
% %     % 显示迭代信息
% %     if BestSol.sol.IsFeasible
% %         Flag=' *';
% %     else
% %         Flag=[', Violation = ' num2str(  BestSol.sol.Violation)];
% %     end
% %     disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it)) Flag]);
% 
% 
% 
% 
% 
% end
% 
% toc
% T=toc;
% 
% function o=Levy(d)
% beta=1.5;
% sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
% u=randn(1,d)*sigma;
% v=randn(1,d);
% step=u./abs(v).^(1/beta);
% %% 自适应调整步长 ---
%     scaleFactor = 0.1 + 0.9 * (1 - it / MaxIt);  % 非线性递减步长，前期探索强，后期稳定
%     weight = rand();  % 随机权重，增加多样性
% 
%     step = weight * scaleFactor .* step;  % 计算最终步长
% o=step;
% end
% end
% 
%% 佳点集+差分DE+局部扰动的混合变异
function [T,L,BestSol , BestCost]=HALA2(param   , model ,  CostFunction,dim)
tic
nVar = model.nVar ;       %  节点个数
VarSize=[1 nVar];   %  决策变量维度

VarMin = 0 ; % 自变量取值   低阶
VarMax= 1 ;% 自变量取值   高阶

nPop =  param.nPop  ; % 种群规模
MaxIt =  param.MaxIt ; % 最大迭代次数
X=zeros(nPop,dim);
empty_bee.Position=[];
empty_bee.Cost=[];
empty_bee.sol=[];
 vec_flag=[1,-1]; %% Directional flag 方向旗            
% Initialize Population Array 
pop=repmat(empty_bee,nPop,1);



% Initialize Best Solution Ever Found
BestSol.Cost=inf;
% Create Initial Population
for i=1:nPop
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    [ pop(i).Cost,  pop(i).sol ] =CostFunction(pop(i).Position);
    if pop(i).Cost<=BestSol.Cost
        BestSol=pop(i);
    end
end

% %% 反向学习
% for i = 1 : nPop
% Y(i,:)=pop(i,:).Position;
%  X_opposite(i,:)=(VarMax+VarMin)-Y(i,:);
% end
% X_combined = [Y; X_opposite];
% [~, idx] = sort(rand(size(X_combined, 1), 1)); 
%     S = X_combined(idx(1:nPop), :);
% for i = 1 : nPop
% pop(i,:).Position=S(i,:);
% end
% %% 反向学习
%% 佳点初始化
% 选择一个与 N 互质的素数 p
    p = 31; 
    while gcd(p, nPop) ~= 1
        p = p + 2; % 选择下一个奇数素数
    end
% 初始化种群矩阵
for i=1:nPop
pop(i,:).Position=zeros(1, dim);
end
 for i = 1: nPop
        for j = 1:dim
            pop(i,:).Position(:,j) = VarMin + (VarMax - VarMin) * mod(i * p / nPop, 1);
        end
    end

%% 佳点初始化
% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

for it=1:MaxIt
RB=randn(nPop,dim);  % Brownian motion布朗运动
    F=vec_flag(floor(2*rand()+1)); % Random directional flag随机方向旗
    theta=2*atan(1-it/MaxIt); % Time-varying parameter时变参数(从 pi/2单调递减到 0)非线性
   pFit =  [pop.Cost] ; %  每个解的 目标函数值
    
    newpop=repmat(empty_bee,nPop,1);
     for i = 1 : nPop
       newpop( i  ) = pop( i  );
     end

        
        [ ~ , sortIndex ] = sort(     pFit   );
%         second_best=newpop( sortIndex(2),: ).Position;
%         third_best=newpop( sortIndex(3),: ).Position;
        [ fMin, bestI ] = min( pFit );      % fMin denotes the global optimum fitness value
         bestX = newpop( bestI, : ).Position;             % bestX denotes the global optimum position corresponding to fMin 
  %% 以上无误
for i=1:nPop
  %  for j=1:dim
%for i=1:nPop
E=2*log(1/rand)*theta;
if E>1
 %% 探索
%% 探索
   %% 差分DE 
 r1 = randi([1 nPop]);
        r2 = randi([1 nPop]);
        r3 = randi([1 nPop]);
         while r1 == i
            r1 = randi([1 nPop]); % 确保 r1 不是 i 本身
         end
          while r2 == r1 || r2 == i
            r2 = randi([1 nPop]); % 确保 r2 不是 r1 或 i 本身
        end
        while r3 == r2 || r3 == r1 || r3 == i
            r3 = randi([1 nPop]); % 确保 r3 不是 r2, r1 或 i 本身
        end
        % 差分变异策略
       % F = 0.5; % 缩放因子
        F = 0.5 + rand * 0.3; % 让 F 介于 [0.5, 0.8]
       V=newpop((r1),:   ).Position+F*(newpop((r2),:   ).Position-newpop((r3), :   ).Position);
       % 交叉操作
        CR = 0.9; % 交叉概率
        j_rand = randi(dim); % 确保至少有一个维度会变异
         for j = 1:dim
            if rand > CR && j ~= j_rand
                V(j) = newpop( i,:   ).Position(:,j); % 部分继承原解
            end
         end
          newpop( i,:   ).Position = V; % 生成新个体
 newpop( i,:   ).Position =  VarMin + mod(     newpop(  i ).Position , VarMax-VarMin  ) ;
 %% 差分DE
        else
            %% 开发
if rand<0.5
    %% 选取不同个体进行变异
    %% 局部扰动的混合变异
 gauss_perturb = randn(1, dim) * 0.05;   
    cauchy_perturb = tan(pi * (rand(1, dim) - 0.5)) * 0.01;  
    radius = sqrt(sum((bestX- newpop( i,:   ).Position).^2));
        r3=rand();
     spiral=radius*(sin(2*pi*r3)+cos(2*pi*r3));
  %  newpop( i,:   ).Position =  bestX + F.*(newpop( i,:   ).Position-bestX)+gauss_perturb+cauchy_perturb ;% %eq3觅食
  newpop( i,:   ).Position=bestX+ F.* newpop( i,:   ).Position.*spiral*rand+gauss_perturb+cauchy_perturb;
           else
               G=2*(sign(rand-0.5))*(1-it/MaxIt);   
               newpop( i,:   ).Position = bestX + F.* G*Levy(dim).* (bestX - newpop( i,:   ).Position) ;%eq4躲避天敌
               %%%
             
               % Xnew(i,:) = bestX + F.* G*Levy(dim).* (bestX - X(i,:)) ;%eq4躲避天敌
            end

 newpop( i,:   ).Position =  VarMin + mod(     newpop(  i ).Position , VarMax-VarMin  ) ;

end
%end
end
           
    %%  Boundary check and evaluation
  
 for i = 1 : nPop
      [  newpop( i   ).Cost  ,   newpop(  i   ).sol ]=CostFunction(     newpop(  i  ).Position );
            
        if    newpop( i  ).Cost < pop( i  ).Cost
            pop(i)  =    newpop( i  ) ;
            
            if pop( i  ).Cost<BestSol.Cost
                BestSol =    pop(i)  ;
            end
        end

    end
    L=BestSol.sol.Jpath;
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    
%     % 显示迭代信息
%     if BestSol.sol.IsFeasible
%         Flag=' *';
%     else
%         Flag=[', Violation = ' num2str(  BestSol.sol.Violation)];
%     end
%     disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it)) Flag]);





end

toc
T=toc;

function o=Levy(d )
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,d)*sigma;v=randn(1,d);step=u./abs(v).^(1/beta);
o=step;
end
end





