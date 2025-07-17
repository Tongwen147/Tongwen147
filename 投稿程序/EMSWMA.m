%% 佳点集+（莱维飞行）探索阶段+Cauchy（开发阶段）
%%  Whale Migration Algorithm (WMA) in MATLAB
function [Destination_fitness,Destination_position,Convergence_curve]=HWMA(N,Max_iteration,lb,ub,dim,fobj)


CostFunction = @(x) fobj(x);

%% Problem Definition
nVar = dim;          % Number of Variables
VarSize = [1 nVar];  % 向量维度
VarMin = lb;         %决策变量最小值
VarMax =ub ;         %决策变量最大值

%% Whale Migration Algorithm Parameters
MaxIt = Max_iteration;        % 最大迭代次数
nPop = N;                     % 总鲸鱼数
NL=round(nPop/2);             %领导鲸鱼的数量(总数的一半)


%% Initialization初始化

% Empty Whale Structure
Whale.Position=[];
Whale.Cost=[];

% Initialize Population Array
pop=repmat(Whale,nPop,1);   % 创建空结构体数组，每个元素代表一只鲸鱼
%repmat(10,3,2)创建一个所有元素的值均为 10 的 3×2 矩阵。
% Initialize Best Solution Ever Found
BestSol.Cost=inf;  % 初始化全局最优解，适应度为无穷大

% Create Initial Whales
% for i=1:nPop
%     pop(i).Position=unifrnd(VarMin,VarMax,VarSize);% 在上下界之间均匀随机生成位置
%     %在VarMin和VarMax之间随机生成
%     pop(i).Cost=CostFunction(pop(i).Position); % 计算适应度
% 
%     if pop(i).Cost<=BestSol.Cost% 更新全局最优解
%         BestSol=pop(i);
%     end
% 
% end
  %% 使用佳点集初始化种群
    X = good_init(nPop, nVar, VarMax, VarMin);
    for i = 1:nPop
        pop(i).Position = X(i, :);
        pop(i).Cost = CostFunction(pop(i).Position);
        if pop(i).Cost <= BestSol.Cost
            BestSol = pop(i);
        end
    end
it=0;
FEs=nPop;% 当前函数评估次数（函数调用次数）N
while it<MaxIt
    it=it+1;
    % Calculate Leader Whales Mean 计算领头鲸平均值
    [~, SortOrder]=sort([pop.Cost]);% 按适应度升序排序
    pop=pop(SortOrder); % 排序后的种群
    Mean = 0; % 初始化均值
    for i=1:NL
        Mean = Mean + pop(i).Position; % 累加前 NL 个个体的位置
    end
    Mean = Mean/NL; % 取平均值
    %% 前 NL 个为 Leader（探索行为）后 nPop - NL 个为 Follower（开发行为）
    for i=1:nPop
        %% 跟随者鲸鱼更新公式
          Cauchy = 0.01 * CauchyInverCumDist(nVar); 
        if i>NL
            %% pop(i-1).Position - pop(i).Position：局部引导（从前一只鲸鱼学习）
            %% BestSol.Position - Mean：全局引导（从整体最优解学习）
        pLevy = 0.3 * (1 - it / MaxIt);  % 迭代后期逐步减弱 Levy 使用概率
  %% 计算 Levy 步长（提前准备）
        LevyStep = 0.01 * Levy(nVar);   % 缩放因子可调整
        %% 原始更新向量（包含局部+全局引导）
delta_local  = rand(VarSize) .* (pop(i-1).Position - pop(i).Position);
delta_global = rand(VarSize) .* (BestSol.Position - Mean);
%% 是否加入 Levy 跳跃
if rand < pLevy
    delta_global = delta_global + LevyStep;  % 在全局引导部分加上跳跃
end

% 更新位置
newsol.Position = Mean + delta_local + delta_global;

           % newsol.Position=Mean+(rand(VarSize)).*(pop(i-1).Position-pop(i).Position)+(rand(VarSize)).*(BestSol.Position-Mean);%%%m
            %% 边界检查和适应度计算
            newsol.Position=max(newsol.Position,VarMin);
            newsol.Position=min(newsol.Position,VarMax);
            newsol.Cost=CostFunction(newsol.Position);
            FEs=FEs+1;%FEs=nPop
            if newsol.Cost<=pop(i).Cost
                pop(i)=newsol;
            end
            if newsol.Cost<=BestSol.Cost
                BestSol=newsol;
            end
        end

        %%%%%%%%%% Movement the Leader Whales
%% 领头鲸鱼更新公式
        if   i<=NL

            newsol.Position=pop(i).Position+(rand).*unifrnd(1*VarMin,1*VarMax,VarSize)+ Cauchy;

            newsol.Position=max(newsol.Position,VarMin);
            newsol.Position=min(newsol.Position,VarMax);
            newsol.Cost=CostFunction(newsol.Position);
            FEs=FEs+1;
            if newsol.Cost<=pop(i).Cost
                pop(i)=newsol;
            end
            if newsol.Cost<=BestSol.Cost
                BestSol=newsol;
            end
        end
    end

    [~, SortOrder]=sort([pop.Cost]);
    pop=pop(SortOrder);
    BestCost(it)=BestSol.Cost;
    Convergence_curve(it)=BestSol.Cost;

    % % Show Iteration Information
    % if mod(it,100)==0
    %     disp(['NFEs ' num2str(FEs) ': Best Cost = ' num2str(BestCost(it))]);
    % end


end
Destination_fitness=BestSol.Cost;
Destination_position=BestSol.Position;
function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1+beta)*sin(pi*beta/2) / ...
            (gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u = randn(1,d) * sigma;
    v = randn(1,d);
    step = u ./ abs(v).^(1/beta);
    o = step;
end
end
