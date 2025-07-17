function X = good_point_set_init(N, dim, lb, ub)
% 选择一个与 N 互质的素数 p
    p = 31; 
    while gcd(p, N) ~= 1
        p = p + 2; % 选择下一个奇数素数
    end
    
    % 初始化种群矩阵
    X = zeros(N, dim);
    
    for i = 1:N
        for j = 1:dim
            X(i,j) = lb(j) + (ub(j) - lb(j)) * mod(i * p / N, 1);
        end
    end
end