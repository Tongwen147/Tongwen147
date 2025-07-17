clc
clear                           
close all
%%
SearchAgents_no=50; % 种群数

Max_iteration=500; % 最大迭代次数

dim = 100; % 可选 2, 10, 30, 50, 100

%%  选择函数

Function_name=1; % 函数名： 1 - 30
[lb,ub,dim,fobj] = Get_Functions_cec2017(Function_name,dim);

%% 调用算法
tic
%[Alpha_score,Alpha_pos,iGWO_cg_curve]=iGWO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%[Best_score2,Best_pos2,GJO_cg_curve2]=GJO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%[Best_score3,Best_pos3,POA_cg_curve3]=POA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_score4,Best_pos4,GWO_cg_curve4]=GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_score5,Best_pos5,Chimp_cg_curve5]=Chimp(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_score6,Best_pos6,SCA_cg_curve6]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_score7,Best_pos7,DMOA_cg_curve7]=DMOA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);

%% plot
figure('Position',[400 200 300 250])
semilogy(iGWO_cg_curve,'Color','r','Linewidth',1.5,'Marker','o','MarkerSize',6,'MarkerIndices',(1:50:500))
hold on
semilogy(GJO_cg_curve2,'Color','c','Linewidth',1.5,'Marker','x','MarkerSize',6,'MarkerIndices',(1:50:500))
hold on
semilogy(POA_cg_curve3,'Color','k','Linewidth',1.5,'Marker','+','MarkerSize',6,'MarkerIndices',(1:50:500))
hold on
semilogy(GWO_cg_curve4,'Color','b','Linewidth',1.5,'Marker','s','MarkerSize',6,'MarkerIndices',(1:50:500))
hold on
semilogy(Chimp_cg_curve5,'Color','g','Linewidth',1.5,'Marker','V','MarkerSize',6,'MarkerIndices',(1:50:500))
hold on
semilogy(SCA_cg_curve6,'Color','y','Linewidth',1.5,'Marker','H','MarkerSize',6,'MarkerIndices',(1:50:500))
hold on
semilogy(DMOA_cg_curve7,'Color','m','Linewidth',1.5,'Marker','P','MarkerSize',6,'MarkerIndices',(1:50:500))


%     plot(cg_curve,'Color','r','Linewidth',1)
title(['Convergence curve, Dim=' num2str(dim)])
xlabel('Iteration');
ylabel(['Best score F' num2str(Function_name) ]);
axis tight
grid on
box on
set(gca,'color','none')
legend('iGWO','GJO','POA','GWO','Chimp','SCA','DMOA')

display(['The best solution obtained by iGWO is : ', num2str(Alpha_pos)]);
display(['The best optimal value of the objective funciton found by iGWO is : ', num2str(Alpha_score)]);

