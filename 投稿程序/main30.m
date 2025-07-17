close all
clear 
clc

SearchAgents_no=50; % 种群数
Max_iteration=500; % 最大迭代次数
dim = 10; % 可选 2, 10, 30, 50, 100%%2,10,20(2022)

Function_name=5; % 函数名： 1 - 30%%1-12(2022)
[lb,ub,dim,fobj] = Get_Functions_cec2022(Function_name,dim);

cos1=ones(20,1);
cos2=ones(20,1);
cos3=ones(20,1);
cos4=ones(20,1);
cos5=ones(20,1);
cos6=ones(20,1);
cos7=ones(20,1);

for v=1:20
%     tic
    [Alpha_score,Alpha_pos,HALA_cg_curve]=EMSWMA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%     t1=toc;
% disp(['NCPO运行时间：',num2str(toc)]);
%     tic;
    [Best_score2,Best_pos2,ZOA_cg_curve2]=ZOA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%     t2=toc;
% disp(['ZOA运行时间：',num2str(toc)]);
%     tic;
    [Best_score3,Best_pos3,ALA_cg_curve3]=WMA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%     t3=toc;
% disp(['CPO运行时间：',num2str(toc)]);
%     tic;
    [Best_score4,Best_pos4,BWO_cg_curve4]=BWO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%     t4=toc;
% disp(['BWO运行时间：',num2str(toc)]);
%     tic;
 
%     t5=toc;
% disp(['Chimp运行时间：',num2str(toc)]);
%     tic;
    [Best_score6,Best_pos6,DBO_cg_curve6]=SAO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%     t6=toc;
% disp(['DBO运行时间：',num2str(toc)]);
%     tic;
    [Best_score7,Best_pos7,DMOA_cg_curve7]=SCSO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);

   % t7=toc;
%     disp(['SCSO运行时间：',num2str(toc)]);
    cos1(v,1)=Alpha_score;
    cos2(v,1)=Best_score2;
    cos3(v,1)=Best_score3;
    cos4(v,1)=Best_score4;
    cos5(v,1)=Best_score5;
    cos6(v,1)=Best_score6;
    cos7(v,1)=Best_score7;
end

HALAmean=mean(cos1);
ZOAmean=mean(cos2);
ALAmean=mean(cos3);
BWOmean=mean(cos4);

DBOmean=mean(cos6);
SCSOmean=mean(cos7);


HALAstd=std(cos1);
ZOAstd=std(cos2);
ALAstd=std(cos3);
BWOstd=std(cos4);

DBOstd=std(cos6);
SCSOstd=std(cos7);

HALAbest=min(cos1);
ZOAbest=min(cos2);
ALAbest=min(cos3);
BWObest=min(cos4);

DBObest=min(cos6);
SCSObest=min(cos7);

HALAworst=max(cos1);
ZOAworst=max(cos2);
ALAworst=max(cos3);
BWOworst=max(cos4);

DBOworst=max(cos6);
SCSOworst=max(cos7);



