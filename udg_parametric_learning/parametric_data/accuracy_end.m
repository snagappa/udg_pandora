function accuracy_end
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

clear all;
close all;

%load demos
listDemos_1 = [0,1];
nbDemos_1 = length(listDemos_1);

listDemos_2 = [3,7];
nbDemos_2 = length(listDemos_2);

%d = struct('data', [], 'textdata', cell(11), 'colheaders', cell(11))

demos_1 = repmat(struct('data',[],'data_norm',[],'data_norm_time',[]), nbDemos_1, 1);
% Average point, end
avg_goal_1 = zeros(1,10);
% Average Trajectory
avg_traj_goal_1 = zeros(400,11);
len_demos_1 = zeros(1,nbDemos_1);

for n=1:nbDemos_1
    ni = listDemos_1(n) ;
    sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    demos_1(n).data = sample.data;
    demos_1(n).data(:,1) = demos_1(n).data(:,1) - demos_1(n).data(1,1) ;
    if ni == 20
        demos_1(n).data(:,11) = sample.data(:,11)-1.6;
    elseif ni == 30
        demos_1(n).data(:,11) = sample.data(:,11)+1.4;
    %else
        %plot(sample.data(:,1)-sample.data(1,1), sample.data(:,11)-0.1,'color',[0,0,0]) ;
    end
    
    avg_goal_1 = avg_goal_1 + sample.data(end,2:11)/nbDemos_1 ;
    %plot(sample.data(:,1)-sample.data(1,1), sample.data(:,2),'color',[0,0,0]) ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %normalitzar les dades
    nbDataTmp = size(sample.data,1);
    xx = linspace(1,nbDataTmp,400);
    demos_1(n).data_norm = zeros(400,11);
    for m = 1:11
        demos_1(n).data_norm(:,m) = spline(1:nbDataTmp, demos_1(n).data(:,m), xx);
    end
    avg_traj_goal_1 = avg_traj_goal_1 + demos_1(n).data_norm/nbDemos_1;
    %len_demos(n) = nbDataTmp;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %data normalization in the same time as the reproduction but with
    %stable intervals
    last_time = demos_1(n).data(end,1);
    nbDataTmp = size(sample.data,1);
    intervals = ceil(last_time / 0.2);
    len_demos_1(n) = intervals;
    xx = linspace(1,nbDataTmp,intervals);
    demos_1(n).data_norm_time = zeros(intervals,11);
    for m = 1:11
        demos_1(n).data_norm_time(:,m) = spline(1:nbDataTmp, demos_1(n).data(:,m), xx);
    end   
end

display('Average position where the demosntrations finish')
display(avg_goal_1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% Second group %%%%%%%%%%%%%%%%%%%%%%%%%

demos_2 = repmat(struct('data',[],'data_norm',[],'data_norm_time',[]), nbDemos_1, 1);
% Average point, end
avg_goal_2 = zeros(1,10);
% Average Trajectory
avg_traj_goal_2 = zeros(400,11);
len_demos_2 = zeros(1,nbDemos_2);

for n=1:nbDemos_2
    ni = listDemos_2(n) ;
    sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    demos_2(n).data = sample.data;
    demos_2(n).data(:,1) = demos_2(n).data(:,1) - demos_2(n).data(1,1) ;
    if ni == 20
        demos_2(n).data(:,11) = sample.data(:,11)-1.6;
    elseif ni == 30
        demos_2(n).data(:,11) = sample.data(:,11)+1.4;
    %else
        %plot(sample.data(:,1)-sample.data(1,1), sample.data(:,11)-0.1,'color',[0,0,0]) ;
    end
    
    avg_goal_2 = avg_goal_2 + sample.data(end,2:11)/nbDemos_2 ;
    %plot(sample.data(:,1)-sample.data(1,1), sample.data(:,2),'color',[0,0,0]) ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %normalitzar les dades
    nbDataTmp = size(sample.data,1);
    xx = linspace(1,nbDataTmp,400);
    demos_2(n).data_norm = zeros(400,11);
    for m = 1:11
        demos_2(n).data_norm(:,m) = spline(1:nbDataTmp, demos_2(n).data(:,m), xx);
    end
    avg_traj_goal_2 = avg_traj_goal_2 + demos_2(n).data_norm/nbDemos_2;
    %len_demos(n) = nbDataTmp;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %data normalization in the same time as the reproduction but with
    %stable intervals
    last_time = demos_2(n).data(end,1);
    nbDataTmp = size(sample.data,1);
    intervals = ceil(last_time / 0.2);
    len_demos_2(n) = intervals;
    xx = linspace(1,nbDataTmp,intervals);
    demos_2(n).data_norm_time = zeros(intervals,11);
    for m = 1:11
        demos_2(n).data_norm_time(:,m) = spline(1:nbDataTmp, demos_2(n).data(:,m), xx);
    end   
end

display('Average position where the demosntrations finish')
display(avg_goal_2)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Compute Max and Min of the demonstration
all_finish = 0;
n = 1;
sorted_len_1 = sort(len_demos_1);
len = sorted_len_1(end-1);
max_limit_1 = zeros(len,11);
max_time_limit_1 = zeros(len,11);
min_limit_1 = zeros(len,11);
min_time_limit_1 = zeros(len,11);
step_time = 0.2;
while all_finish < nbDemos_1-1
    all_finish = 0;
    values_array = zeros(nbDemos_1,11);
    %copy values of the demos with data
    for m=1:nbDemos_1
        if n <= len_demos_1(m)
            values_array(m,:) = demos_1(m).data_norm_time(n,:);
        else
            all_finish = all_finish + 1;
            if all_finish ~= 0
                %display('Comenxa aqui');
            end
        end        
    end
    %compute average time
    if all_finish ~= nbDemos_1-1
    %if all_finish ~= nbDemos
        % Time looks strange, dut to I use the average not the one which
        % correspont with the max of the trajectory. But since the maximum 
        % can be different in every degree of freedom this should be okey. 
        time_avg = sum(values_array(:,1))/ (nbDemos_1 - all_finish);
        max_limit_1(n,1) = time_avg;
        min_limit_1(n,1) = time_avg;
        %dinf max and min aboiding zeros
        for m=1:10
            max_limit_1(n,m+1) = max(values_array(values_array(:,m+1)~=0.0, m+1)) ;        
            min_limit_1(n,m+1) = min(values_array(values_array(:,m+1)~=0.0, m+1)) ;
            index_max = values_array(:,m+1)==max_limit_1(n,m+1);
            index_min = values_array(:,m+1)==min_limit_1(n,m+1);
            max_time_limit_1(n,m+1) = values_array(index_max,1);
            min_time_limit_1(n,m+1) = values_array(index_min,1);
        end
        n = n + 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% Second Group %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
all_finish = 0;
n = 1;
sorted_len_2 = sort(len_demos_2);
len = sorted_len_2(end-1);
max_limit_2 = zeros(len,11);
max_time_limit_2 = zeros(len,11);
min_limit_2 = zeros(len,11);
min_time_limit_2 = zeros(len,11);
step_time = 0.2;
while all_finish < nbDemos_2-1
    all_finish = 0;
    values_array = zeros(nbDemos_2,11);
    %copy values of the demos with data
    for m=1:nbDemos_2
        if n <= len_demos_2(m)
            values_array(m,:) = demos_2(m).data_norm_time(n,:);
        else
            all_finish = all_finish + 1;
            if all_finish ~= 0
                %display('Comenxa aqui');
            end
        end        
    end
    %compute average time
    if all_finish ~= nbDemos_2-1
    %if all_finish ~= nbDemos
        % Time looks strange, dut to I use the average not the one which
        % correspont with the max of the trajectory. But since the maximum 
        % can be different in every degree of freedom this should be okey. 
        time_avg = sum(values_array(:,1))/ (nbDemos_2 - all_finish);
        max_limit_2(n,1) = time_avg;
        min_limit_2(n,1) = time_avg;
        %dinf max and min aboiding zeros
        for m=1:10
            max_limit_2(n,m+1) = max(values_array(values_array(:,m+1)~=0.0, m+1)) ;        
            min_limit_2(n,m+1) = min(values_array(values_array(:,m+1)~=0.0, m+1)) ;
            index_max = values_array(:,m+1)==max_limit_2(n,m+1);
            index_min = values_array(:,m+1)==min_limit_2(n,m+1);
            max_time_limit_2(n,m+1) = values_array(index_max,1);
            min_time_limit_2(n,m+1) = values_array(index_min,1);
        end
        n = n + 1;
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Reproductions Lists %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

listSamples = [0,1,5];
nbSamples = length(listSamples);

%load reproductions
% listSamples = (0:64);
% nbSamples = length(listSamples);

% Plot Demos by Current Groups

% zero_current = [22,27,49,62:64]; %30:47,
% fifteen_current = [57];
% twentyfive_current = [0:5,55,56];
% thirty_current = [6:9,48,50];
% thirty_five_current = [10:13,51:54,58:61];
% forty_current = [14:18,23,28];
% forty_five_current = [19:21];
% fifty_current = [24,29];
% fifty_five_current = [25];
% sixty_current = [26];
% 
% nb_groups_len = zeros(1,6);
% group_0 = [zero_current, fifteen_current];
% nb_groups_len(1) = length(group_0);
% group_1 = [twentyfive_current, thirty_current]; 
% nb_groups_len(2) = length(group_1);
% group_2 = [thirty_five_current, forty_current, forty_five_current];
% nb_groups_len(3) = length(group_2);
% group_3 = [fifty_current, fifty_five_current, sixty_current];
% nb_groups_len(4) = length(group_3);
% 
% success = [3:5,7:10,12:14,18,20:25,27,29,30,31,32,34,35:38,39,40,42:50,52:54,56:58,61:64];
% nb_groups_len(5) = length(success);
% fail = [0:2,6,11,15:17,19,26,28,33,37,41,51,55,59,60];
% nb_groups_len(6) = length(fail);
% 
% %We have to add one because matlab works with index starting to 1 not 0 as
% % the files we generate
% group_0 = group_0 +1;
% group_1 = group_1 +1;
% group_2 = group_2 +1;
% group_3 = group_3 +1;
% 
% success = success + 1;
% fail = fail + 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Reproductions Lists %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% x_auv_end = zeros(nbSamples, 1);
% y_auv_end = zeros(nbSamples, 1);
% z_auv_end = zeros(nbSamples, 1);
% yaw_auv_end = zeros(nbSamples, 1);
% 
% 
% x_ee_end = zeros(nbSamples,1);
% y_ee_end = zeros(nbSamples,1);
% z_ee_end = zeros(nbSamples,1);
% roll_ee_end = zeros(nbSamples,1);

repro = repmat(struct('data',[],'data_norm',[],'error_traj', [],'square_error_traj',[]), nbSamples, 1);
%Distance between the end position and the avg end position of the demos
repro_dist_zero = zeros(nbSamples,10);
%Standard desvitation of the distance between the avg end pose and the trajectory 
repro_std_dev = zeros(1,10);
%Euclidian distance of the end position of all repro with the end effector
euclidian_dist_zero = zeros(nbSamples,2);
%Average of the square error along the trajecoties
avg_square_err = zeros(1,11);
%Average of the euclidian distance
avg_euclidian_dist_zero = zeros(1,2);
%Average of the end distance of the quadratic error
avg_end_dist_zero = zeros(1,10);

% Standard desvitation of the distance between the avg end pose and the trajectory 
repro_std_dev_group = zeros(3,10);
%Average of the square error along the trajecoties
avg_square_err_group = zeros(3,11);
%Average of the euclidian distance
avg_euclidian_dist_zero_group = zeros(3,2);
%Average of the end distance of the quadratic error
avg_end_dist_zero_group = zeros(3,10);


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['trajectory_played_sim_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    repro(n).data = sample.data;
    repro(n).data(:,1) = repro(n).data(:,1) - repro(n).data(1,1) ;
    % TODO: Aqui i poden haver problemes mirar com ho faig això
    repro_dist_zero(n,:) = avg_goal_1 - sample.data(end,2:11);
    euclidian_dist_zero(n,1) = sqrt(sum(repro_dist_zero(n,2:4).^2));
    euclidian_dist_zero(n,2) = sqrt(sum(repro_dist_zero(n,5:7).^2));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     ni = ni +1;
%     if ~isempty(find(group_0==ni, 1))
%         index_group = 1;
%     elseif ~isempty(find(group_1==ni, 1))
%         index_group = 2;
%     elseif ~isempty(find(group_2==ni, 1))
%         index_group = 3;
%     elseif ~isempty(find(group_3==ni, 1))
%         index_group = 4;
%     else
%        display('ERROR group not found');
%        display(ni);
%     end
%     
%     if ~isempty(find(success==ni, 1))
%         success_index = 5;
%     elseif ~isempty(find(fail==ni, 1))
%         success_index = 6;
%     else
%         display('Error in the Success');
%         display(ni);
%     end
    
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %normalitzar les dades
    nbDataTmp = size(sample.data,1);
    xx = linspace(1,nbDataTmp,400);
    repro(n).data_norm = zeros(400,11);
    repro(n).error_traj = zeros(400,11);
    for m = 1:11
        repro(n).data_norm(:,m) = spline(1:nbDataTmp, sample.data(:,m), xx);
    end
%     repro(n).error_traj = sqrt((avg_traj_goal - repro(n).data_norm).^2.0);
%     repro(n).square_error_traj = zeros(1,11);
%     for m =2:11
%         repro(n).square_error_traj(1,m) = (1/400)*(sum(((avg_traj_goal(:,m) - repro(n).data_norm(:,m)).^2.0)));
%     end
%     avg_square_err = avg_square_err + repro(n).square_error_traj/nbSamples;
%     avg_euclidian_dist_zero(1,1) = avg_euclidian_dist_zero(1,1) + euclidian_dist_zero(n,1)/nbSamples;
%     avg_euclidian_dist_zero(1,2) = avg_euclidian_dist_zero(1,2) + euclidian_dist_zero(n,2)/nbSamples;
%     avg_end_dist_zero = avg_end_dist_zero + repro_dist_zero(n,:)/nbSamples;
%     
%     avg_square_err_group(index_group,:) = avg_square_err_group(index_group,:) + repro(n).square_error_traj/nb_groups_len(index_group);
%     avg_square_err_group(success_index,:) = avg_square_err_group(success_index,:) + repro(n).square_error_traj/nb_groups_len(success_index);
% 
%     avg_euclidian_dist_zero_group(index_group,1) = avg_euclidian_dist_zero_group(index_group,1) + euclidian_dist_zero(n,1)/nb_groups_len(index_group);
%     avg_euclidian_dist_zero_group(index_group,2) = avg_euclidian_dist_zero_group(index_group,2) + euclidian_dist_zero(n,2)/nb_groups_len(index_group);
%    
%     avg_euclidian_dist_zero_group(success_index,1) = avg_euclidian_dist_zero_group(success_index,1) + euclidian_dist_zero(n,1)/nb_groups_len(success_index);
%     avg_euclidian_dist_zero_group(success_index,2) = avg_euclidian_dist_zero_group(success_index,2) + euclidian_dist_zero(n,2)/nb_groups_len(success_index);
%     
%     avg_end_dist_zero_group(index_group,:) = avg_end_dist_zero_group(index_group,:) + repro_dist_zero(n,:)/nb_groups_len(index_group);
%     avg_end_dist_zero_group(success_index,:) = avg_end_dist_zero_group(success_index,:) + repro_dist_zero(n,:)/nb_groups_len(success_index);

end

%Compute the Standard Deviation
% std_deviation = zeros(6,2);
% std_deviation(1,1) = std(euclidian_dist_zero(group_0,1)); 
% std_deviation(1,2) = std(euclidian_dist_zero(group_0,2)); 
% std_deviation(2,1) = std(euclidian_dist_zero(group_1,1)); 
% std_deviation(2,2) = std(euclidian_dist_zero(group_1,2)); 
% std_deviation(3,1) = std(euclidian_dist_zero(group_2,1)); 
% std_deviation(3,2) = std(euclidian_dist_zero(group_2,2));
% std_deviation(4,1) = std(euclidian_dist_zero(group_3,1)); 
% std_deviation(4,2) = std(euclidian_dist_zero(group_3,2));
% std_deviation(5,1) = std(euclidian_dist_zero(success,1)); 
% std_deviation(5,2) = std(euclidian_dist_zero(success,2)); 
% std_deviation(6,1) = std(euclidian_dist_zero(fail,1)); 
% std_deviation(6,2) = std(euclidian_dist_zero(fail,2)); 
% 
% display('Avg Square Error along all the trajectories')
% display(avg_square_err);
% display('Avg Euclidian Distance betwen Last position and AUV and End-effector') 
% display(avg_euclidian_dist_zero);
% display('Avg End-effector distance')
% display(avg_end_dist_zero);
% 
% display('Avg Euclidian Distance Zero AUV');
% display(avg_euclidian_dist_zero_group(:,1));
% display('Avg Euclidian Distance Zero EE');
% display(avg_euclidian_dist_zero_group(:,2));
% 
% display('Standard Deviation');
% display(std_deviation);

%repro_std_dev = repro_dist_zero.^2;
% for m=1:10
%     avg_dist = sum(repro_dist_zero(:,m))/nbSamples;
%     repro_std_dev(1,m) = sqrt(sum((avg_dist-repro_dist_zero(:,m)).^2)/nbSamples);
%     
%     avg_dist_group_0 = sum(repro_dist_zero(group_0,m))/nbSamples;
%     avg_dist_group_1 = sum(repro_dist_zero(group_1,m))/nbSamples;
%     avg_dist_group_2 = sum(repro_dist_zero(group_2,m))/nbSamples;
%     avg_dist_group_3 = sum(repro_dist_zero(group_3,m))/nbSamples;
%     avg_dist_success = sum(repro_dist_zero(success,m))/nbSamples;
%     avg_dist_fail = sum(repro_dist_zero(fail,m))/nbSamples;
% 
%     repro_std_dev_group(1,m) = sqrt(sum((avg_dist_group_0-repro_dist_zero(group_0,m)).^2)/nb_groups_len(1));
%     repro_std_dev_group(2,m) = sqrt(sum((avg_dist_group_1-repro_dist_zero(group_1,m)).^2)/nb_groups_len(2));
%     repro_std_dev_group(3,m) = sqrt(sum((avg_dist_group_2-repro_dist_zero(group_2,m)).^2)/nb_groups_len(3));
%     repro_std_dev_group(4,m) = sqrt(sum((avg_dist_group_3-repro_dist_zero(group_3,m)).^2)/nb_groups_len(4));
%     repro_std_dev_group(5,m) = sqrt(sum((avg_dist_success-repro_dist_zero(success,m)).^2)/nb_groups_len(5));
%     repro_std_dev_group(6,m) = sqrt(sum((avg_dist_fail-repro_dist_zero(fail,m)).^2)/nb_groups_len(6));
% end
% 
% display('Standard desviation of the error');
% display(repro_std_dev);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% Create tables of succes %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% percent = [ 0, 15, 25, 30, 35, 40, 45, 50, 55, 60 ];
% success_rate = [0.86, 1.0, 0.5, 0.83, 0.66, 0.42, 0.66, 1.0, 1.0, 0.0];
% 
% graph = [0, 1, 2, 3, 4];
% % 0 -> 0-15, 1 -> 20-30, 2 -> 35-45, 3 -> 50-60, 4 -> 70-100
% success_rate_2 = [0.87, 0.64, 0.59, 0.5, 0.0];

% figure(1)
% hold on;
% title('Raw success','FontSize',20);
% xlabel('Current (%)','FontSize',20);
% ylabel('Success Rate','FontSize',20);
% plot(percent, success_rate);
% hold off;
% 
% figure()
% hold on;
% title('Grupped Logically','FontSize',20);
% xlabel('Current Intervals (%)','FontSize',20);
% ylabel('Success Rate','FontSize',20);
% plot(graph, success_rate_2);
% hold off;
%display(repro_dist_zero(5,:));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Demos %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Second %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure()


subplot(4,2,1)
hold on ;
%title('Demos','FontSize',20);
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;

% listDemos = [70, 71,72,20,30];
% nbDemos = length(listDemos);

%AUV
% X
plot(max_time_limit_2(:,2), max_limit_2(:,2),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,2), min_limit_2(:,2),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;


for n=1:nbDemos_2
    plot(demos_2(n).data(:,1), demos_2(n).data(:,2),'LineWidth', 2.0,'color',[0,0,0]) ;
end

plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,2),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 -0.1 0.7])
hold off;

% Y
subplot(4,2,3)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;

plot(max_time_limit_2(:,3), max_limit_2(:,3),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,3), min_limit_2(:,3),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;

for n=1:nbDemos_2
    plot(demos_2(n).data(:,1), demos_2(n).data(:,3),'LineWidth', 2.0,'color',[0,0,0]) ;
end
plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,3),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 -0.3 0.3])

hold off;

% Z
subplot(4,2,5)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Z (m)', 'FontSize',20)
grid on;

plot(max_time_limit_2(:,4), max_limit_2(:,4),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,4), min_limit_2(:,4),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;

for n=1:nbDemos_2
    plot(demos_2(n).data(:,1), demos_2(n).data(:,4),'LineWidth', 2.0,'color',[0,0,0]) ;
end
plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,4),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 0.8 2.5])
hold off;

% Yaw
subplot(4,2,7)
hold on ;
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Yaw (rad)', 'FontSize',20)
grid on;

plot(max_time_limit_2(:,5), max_limit_2(:,5),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,5), min_limit_2(:,5),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;

for n=1:nbDemos_2
    plot(demos_2(n).data(:,1), demos_2(n).data(:,5),'LineWidth', 2.0,'color',[0,0,0]) ;
end
plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,5),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 -0.8 0.1])
hold off;

% End-effector
% X
subplot(4,2,2)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;

plot(max_time_limit_2(:,6), max_limit_2(:,6),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,6), min_limit_2(:,6),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;

for n=1:nbDemos_2
    plot(demos_2(n).data(:,1), demos_2(n).data(:,6),'LineWidth', 2.0,'color',[0,0,0]) ;
end
plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,6),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 -0.5 0.1])
hold off;

% Y
subplot(4,2,4)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;

plot(max_time_limit_2(:,7), max_limit_2(:,7),'LineStyle', '--', 'LineWidth',6.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,7), min_limit_2(:,7),'LineStyle', '--', 'LineWidth',6.0,'color',[0.2,0.4,0.2]) ;

for n=1:nbDemos_2
    plot(demos_2(n).data(:,1), demos_2(n).data(:,7),'LineWidth', 2.0,'color',[0,0,0]) ;
end
plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,7),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 -1.0 0.1])
hold off;

% Z
subplot(4,2,6)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Z (m)', 'FontSize',20)
grid on;

plot(max_time_limit_2(:,8), max_limit_2(:,8),'LineStyle', '--', 'LineWidth',6.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,8), min_limit_2(:,8),'LineStyle', '--', 'LineWidth',6.0,'color',[0.2,0.4,0.2]) ;
for n=1:nbDemos_2
    plot(demos_2(n).data(:,1), demos_2(n).data(:,8),'LineWidth', 2.0,'color',[0,0,0]) ;
end

plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,8),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 -0.2 1.5])
hold off;

% Roll
subplot(4,2,8)
hold on ;
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Roll (rad)', 'FontSize',20)
grid on;

plot(max_time_limit_2(:,11), max_limit_2(:,11),'LineStyle', '--', 'LineWidth',6.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,11), min_limit_2(:,11),'LineStyle', '--', 'LineWidth',6.0,'color',[0.2,0.4,0.2]) ;
for n=1:nbDemos_2
    plot(demos_2(n).data(:,1), demos_2(n).data(:,11),'LineWidth', 2.0,'color',[0,0,0]) ;
end
plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,11),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 -0.9 0.5])
hold off;


subplot(4,2,1)
hold on ;
%title('Demos','FontSize',20);
%xlabel( 'Time (s)', 'FontSize',20)
%ylabel( 'X (m)', 'FontSize',20)
grid on;

% listDemos = [70, 71,72,20,30];
% nbDemos = length(listDemos);

%AUV
% X
plot(max_time_limit_1(:,2), max_limit_1(:,2),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,2), min_limit_1(:,2),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;


for n=1:nbDemos_1
    plot(demos_1(n).data(:,1), demos_1(n).data(:,2),'LineWidth', 2.0,'color',[0,0,0]) ;
end

plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,2),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;

hold off;

% Y
subplot(4,2,3)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
%ylabel( 'Y (m)', 'FontSize',20)
grid on;

plot(max_time_limit_1(:,3), max_limit_1(:,3),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,3), min_limit_1(:,3),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;

for n=1:nbDemos_1
    plot(demos_1(n).data(:,1), demos_1(n).data(:,3),'LineWidth', 2.0,'color',[0,0,0]) ;
end
plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,3),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;

hold off;

% Z
subplot(4,2,5)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
%ylabel( 'Z (m)', 'FontSize',20)
grid on;

plot(max_time_limit_1(:,4), max_limit_1(:,4),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,4), min_limit_1(:,4),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;

for n=1:nbDemos_1
    plot(demos_1(n).data(:,1), demos_1(n).data(:,4),'LineWidth', 2.0,'color',[0,0,0]) ;
end
plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,4),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;

hold off;

% Yaw
subplot(4,2,7)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
%ylabel( 'Yaw (rad)', 'FontSize',20)
grid on;

plot(max_time_limit_1(:,5), max_limit_1(:,5),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,5), min_limit_1(:,5),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;

for n=1:nbDemos_1
    plot(demos_1(n).data(:,1), demos_1(n).data(:,5),'LineWidth', 2.0,'color',[0,0,0]) ;
end
plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,5),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;

hold off;

% End-effector
% X
subplot(4,2,2)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
%ylabel( 'X (m)', 'FontSize',20)
grid on;

plot(max_time_limit_1(:,6), max_limit_1(:,6),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,6), min_limit_1(:,6),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;

for n=1:nbDemos_1
    plot(demos_1(n).data(:,1), demos_1(n).data(:,6),'LineWidth', 2.0,'color',[0,0,0]) ;
end
plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,6),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;
hold off;

% Y
subplot(4,2,4)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
%ylabel( 'Y (m)', 'FontSize',20)
grid on;

plot(max_time_limit_1(:,7), max_limit_1(:,7),'LineStyle', '--', 'LineWidth',6.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,7), min_limit_1(:,7),'LineStyle', '--', 'LineWidth',6.0,'color',[0.4,0.6,1]) ;

for n=1:nbDemos_1
    plot(demos_1(n).data(:,1), demos_1(n).data(:,7),'LineWidth', 2.0,'color',[0,0,0]) ;
end
plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,7),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;
hold off;

% Z
subplot(4,2,6)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
%ylabel( 'Z (m)', 'FontSize',20)
grid on;

plot(max_time_limit_1(:,8), max_limit_1(:,8),'LineStyle', '--', 'LineWidth',6.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,8), min_limit_1(:,8),'LineStyle', '--', 'LineWidth',6.0,'color',[0.4,0.6,1]) ;
for n=1:nbDemos_1
    plot(demos_1(n).data(:,1), demos_1(n).data(:,8),'LineWidth', 2.0,'color',[0,0,0]) ;
end

plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,8),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;
hold off;

% Roll
subplot(4,2,8)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
%ylabel( 'Roll (rad)', 'FontSize',20)
grid on;

plot(max_time_limit_1(:,11), max_limit_1(:,11),'LineStyle', '--', 'LineWidth',6.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,11), min_limit_1(:,11),'LineStyle', '--', 'LineWidth',6.0,'color',[0.4,0.6,1]) ;
for n=1:nbDemos_1
    plot(demos_1(n).data(:,1), demos_1(n).data(:,11),'LineWidth', 2.0,'color',[0,0,0]) ;
end
plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,11),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;
hold off;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Second %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure()
%set(gca, 'FontSize', 18)
subplot(4,2,1)
hold on ;
%title('Results','FontSize',20);
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;

% listDemos = [70, 71,72,20,30];
% nbDemos = length(listDemos);

%AUV
% X
plot(max_time_limit_2(:,2), max_limit_2(:,2),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,2), min_limit_2(:,2),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;


plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,2),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 -0.1 0.7])
hold off;

% Y
subplot(4,2,3)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;

plot(max_time_limit_2(:,3), max_limit_2(:,3),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,3), min_limit_2(:,3),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;

plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,3),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 -0.3 0.3])
hold off;

% Z
subplot(4,2,5)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Z (m)', 'FontSize',20)
grid on;

plot(max_time_limit_2(:,4), max_limit_2(:,4),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,4), min_limit_2(:,4),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;

plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,4),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 0.8 2.5])
hold off;

% Yaw
subplot(4,2,7)
hold on ;
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Yaw (rad)', 'FontSize',20)
grid on;

plot(max_time_limit_2(:,5), max_limit_2(:,5),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,5), min_limit_2(:,5),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;

plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,5),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 -0.8 0.1])
hold off;

% End-effector
% X
subplot(4,2,2)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;

plot(max_time_limit_2(:,6), max_limit_2(:,6),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,6), min_limit_2(:,6),'LineStyle', '--', 'LineWidth',5.0,'color',[0.2,0.4,0.2]) ;

plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,6),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 -0.5 0.1])
hold off;

% Y
subplot(4,2,4)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;

plot(max_time_limit_2(:,7), max_limit_2(:,7),'LineStyle', '--', 'LineWidth',6.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,7), min_limit_2(:,7),'LineStyle', '--', 'LineWidth',6.0,'color',[0.2,0.4,0.2]) ;

plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,7),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 -1.0 0.1])
hold off;

% Z
subplot(4,2,6)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Z (m)', 'FontSize',20)
grid on;

plot(max_time_limit_2(:,8), max_limit_2(:,8),'LineStyle', '--', 'LineWidth',6.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,8), min_limit_2(:,8),'LineStyle', '--', 'LineWidth',6.0,'color',[0.2,0.4,0.2]) ;

plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,8),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 -0.2 1.5])
hold off;

% Roll
subplot(4,2,8)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Roll (rad)', 'FontSize',20)
grid on;

plot(max_time_limit_2(:,11), max_limit_2(:,11),'LineStyle', '--', 'LineWidth',6.0,'color',[0.2,0.4,0.2]) ;
plot(min_time_limit_2(:,11), min_limit_2(:,11),'LineStyle', '--', 'LineWidth',6.0,'color',[0.2,0.4,0.2]) ;

plot(avg_traj_goal_2(:,1), avg_traj_goal_2(:,11),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0.8,0]) ;
set(gca, 'FontSize', 16)
axis([0.0 110 -0.9 0.5])
hold off;


subplot(4,2,1)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;

% listDemos = [70, 71,72,20,30];
% nbDemos = length(listDemos);

%AUV
% X
plot(max_time_limit_1(:,2), max_limit_1(:,2),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,2), min_limit_1(:,2),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;

plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,2),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;

plot(repro(3).data(:,1), repro(3).data(:,2),'LineWidth', 2.0,'color',[1,0,0]) ;
plot(repro(2).data(:,1), repro(2).data(:,2),'LineWidth', 2.0,'color',[0,0,1]) ;
plot(repro(1).data(:,1), repro(1).data(:,2),'LineWidth', 2.0,'color',[0,1,0]) ;


hold off;

% Y
subplot(4,2,3)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;

plot(max_time_limit_1(:,3), max_limit_1(:,3),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,3), min_limit_1(:,3),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;
plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,3),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;

plot(repro(3).data(:,1), repro(3).data(:,3),'LineWidth', 2.0,'color',[1,0,0]) ;
plot(repro(2).data(:,1), repro(2).data(:,3),'LineWidth', 2.0,'color',[0,0,1]) ;
plot(repro(1).data(:,1), repro(1).data(:,3),'LineWidth', 2.0,'color',[0,1,0]) ;

hold off;

% Z
subplot(4,2,5)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Z (m)', 'FontSize',20)
grid on;

plot(max_time_limit_1(:,4), max_limit_1(:,4),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,4), min_limit_1(:,4),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;
plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,4),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;

plot(repro(3).data(:,1), repro(3).data(:,4),'LineWidth', 2.0,'color',[1,0,0]) ;
plot(repro(2).data(:,1), repro(2).data(:,4),'LineWidth', 2.0,'color',[0,0,1]) ;
plot(repro(1).data(:,1), repro(1).data(:,4),'LineWidth', 2.0,'color',[0,1,0]) ;

hold off;

% Yaw
subplot(4,2,7)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Yaw (rad)', 'FontSize',20)
grid on;

plot(max_time_limit_1(:,5), max_limit_1(:,5),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,5), min_limit_1(:,5),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;
plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,5),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;

plot(repro(3).data(:,1), repro(3).data(:,5),'LineWidth', 2.0,'color',[1,0,0]) ;
plot(repro(2).data(:,1), repro(2).data(:,5),'LineWidth', 2.0,'color',[0,0,1]) ;
plot(repro(1).data(:,1), repro(1).data(:,5),'LineWidth', 2.0,'color',[0,1,0]) ;

hold off;

% End-effector
% X
subplot(4,2,2)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;

plot(max_time_limit_1(:,6), max_limit_1(:,6),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,6), min_limit_1(:,6),'LineStyle', '--', 'LineWidth',5.0,'color',[0.4,0.6,1]) ;
plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,6),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;

plot(repro(3).data(:,1), repro(3).data(:,6),'LineWidth', 2.0,'color',[1,0,0]) ;
plot(repro(2).data(:,1), repro(2).data(:,6),'LineWidth', 2.0,'color',[0,0,1]) ;
plot(repro(1).data(:,1), repro(1).data(:,6),'LineWidth', 2.0,'color',[0,1,0]) ;

hold off;

% Y
subplot(4,2,4)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;

plot(max_time_limit_1(:,7), max_limit_1(:,7),'LineStyle', '--', 'LineWidth',6.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,7), min_limit_1(:,7),'LineStyle', '--', 'LineWidth',6.0,'color',[0.4,0.6,1]) ;
plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,7),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;

plot(repro(3).data(:,1), repro(3).data(:,7),'LineWidth', 2.0,'color',[1,0,0]) ;
plot(repro(2).data(:,1), repro(2).data(:,7),'LineWidth', 2.0,'color',[0,0,1]) ;
plot(repro(1).data(:,1), repro(1).data(:,7),'LineWidth', 2.0,'color',[0,1,0]) ;

hold off;

% Z
subplot(4,2,6)
hold on ;
%xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Z (m)', 'FontSize',20)
grid on;

plot(max_time_limit_1(:,8), max_limit_1(:,8),'LineStyle', '--', 'LineWidth',6.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,8), min_limit_1(:,8),'LineStyle', '--', 'LineWidth',6.0,'color',[0.4,0.6,1]) ;
plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,8),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;

plot(repro(3).data(:,1), repro(3).data(:,8),'LineWidth', 2.0,'color',[1,0,0]) ;
plot(repro(2).data(:,1), repro(2).data(:,8),'LineWidth', 2.0,'color',[0,0,1]) ;
plot(repro(1).data(:,1), repro(1).data(:,8),'LineWidth', 2.0,'color',[0,1,0]) ;

hold off;

% Roll
subplot(4,2,8)
hold on ;
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Roll (rad)', 'FontSize',20)
grid on;

plot(max_time_limit_1(:,11), max_limit_1(:,11),'LineStyle', '--', 'LineWidth',6.0,'color',[0.4,0.6,1]) ;
plot(min_time_limit_1(:,11), min_limit_1(:,11),'LineStyle', '--', 'LineWidth',6.0,'color',[0.4,0.6,1]) ;
plot(avg_traj_goal_1(:,1), avg_traj_goal_1(:,11),'LineStyle', '--', 'LineWidth',2.0,'color',[0,0,1]) ;

plot(repro(3).data(:,1), repro(3).data(:,11),'LineWidth', 2.0,'color',[1,0,0]) ;
plot(repro(2).data(:,1), repro(2).data(:,11),'LineWidth', 2.0,'color',[0,0,1]) ;
plot(repro(1).data(:,1), repro(1).data(:,11),'LineWidth', 2.0,'color',[0,1,0]) ;

hold off;





% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Repro Current %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % group_0 group_1 group_2 group_3
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Repro Group 0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% nbRepro = length(group_0);
% 
% figure();
% subplot(4,2,1)
% hold on
% title('Repro Group 0','FontSize',20);
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'X (m)', 'FontSize',20)
% grid on;
% 
% %AUV
% % X
% plot(max_limit(:,1), max_limit(:,2),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,2),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_0(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,2),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,2),'color',[1,0,0]) ;
% hold off;
% 
% % Y
% subplot(4,2,3)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Y (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,3),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,3),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_0(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,3),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,3),'color',[1,0,0]) ;
% hold off;
% 
% % Z
% subplot(4,2,5)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Z (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,4),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,4),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_0(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,4),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,4),'color',[1,0,0]) ;
% hold off;
% 
% % Yaw
% subplot(4,2,7)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Yaw (rad)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,5),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,5),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_0(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,5),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,5),'color',[1,0,0]) ;
% hold off;
% 
% % End-effector
% % X
% subplot(4,2,2)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'X (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,6),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,6),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_0(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,6),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,6),'color',[1,0,0]) ;
% hold off;
% 
% % Y
% subplot(4,2,4)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Y (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,7),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,7),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_0(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,7),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,7),'color',[1,0,0]) ;
% hold off;
% 
% % Z
% subplot(4,2,6)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Z (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,8),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,8),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_0(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,8),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,8),'color',[1,0,0]) ;
% hold off;
% 
% % Yaw
% subplot(4,2,8)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Roll (rad)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,11),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,11),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_0(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,11),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,11),'color',[1,0,0]) ;
% hold off;
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Repro Group 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% nbRepro = length(group_1);
% 
% figure()
% subplot(4,2,1)
% hold on
% title('Repro Group 1','FontSize',20);
% xlabel( 'Time (s)', 'FontSize',20);
% ylabel( 'X (m)', 'FontSize',20);
% grid on;
% 
% %AUV
% % X
% plot(max_limit(:,1), max_limit(:,2),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,2),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_1(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,2),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,2),'color',[1,0,0]) ;
% hold off;
% 
% % Y
% subplot(4,2,3)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Y (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,3),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,3),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_1(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,3),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,3),'color',[1,0,0]) ;
% hold off;
% 
% % Z
% subplot(4,2,5)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Z (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,4),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,4),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_1(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,4),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,4),'color',[1,0,0]) ;
% hold off;
% 
% % Yaw
% subplot(4,2,7)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Yaw (rad)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,5),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,5),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_1(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,5),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,5),'color',[1,0,0]) ;
% hold off;
% 
% % End-effector
% % X
% subplot(4,2,2)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'X (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,6),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,6),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_1(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,6),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,6),'color',[1,0,0]) ;
% hold off;
% 
% % Y
% subplot(4,2,4)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Y (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,7),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,7),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_1(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,7),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,7),'color',[1,0,0]) ;
% hold off;
% 
% % Z
% subplot(4,2,6)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Z (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,8),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,8),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_1(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,8),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,8),'color',[1,0,0]) ;
% hold off;
% 
% % Yaw
% subplot(4,2,8)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Roll (rad)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,11),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,11),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_1(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,11),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,11),'color',[1,0,0]) ;
% hold off;
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Repro Group 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% nbRepro = length(group_2);
% 
% figure()
% subplot(4,2,1)
% hold on
% title('Repro Group 2','FontSize',20);
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'X (m)', 'FontSize',20)
% grid on;
% 
% %AUV
% % X
% plot(max_limit(:,1), max_limit(:,2),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,2),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_2(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,2),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,2),'color',[1,0,0]) ;
% hold off;
% 
% % Y
% subplot(4,2,3)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Y (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,3),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,3),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_2(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,3),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,3),'color',[1,0,0]) ;
% hold off;
% 
% % Z
% subplot(4,2,5)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Z (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,4),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,4),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_2(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,4),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,4),'color',[1,0,0]) ;
% hold off;
% 
% % Yaw
% subplot(4,2,7)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Yaw (rad)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,5),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,5),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_2(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,5),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,5),'color',[1,0,0]) ;
% hold off;
% 
% % End-effector
% % X
% subplot(4,2,2)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'X (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,6),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,6),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_2(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,6),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,6),'color',[1,0,0]) ;
% hold off;
% 
% % Y
% subplot(4,2,4)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Y (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,7),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,7),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_2(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,7),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,7),'color',[1,0,0]) ;
% hold off;
% 
% % Z
% subplot(4,2,6)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Z (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,8),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,8),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_2(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,8),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,8),'color',[1,0,0]) ;
% hold off;
% 
% % Yaw
% subplot(4,2,8)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Roll (rad)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,11),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,11),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_2(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,11),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,11),'color',[1,0,0]) ;
% hold off;
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Repro Group 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% nbRepro = length(group_3);
% 
% figure()
% subplot(4,2,1)
% hold on
% title('Repro Group 3','FontSize',20);
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'X (m)', 'FontSize',20)
% grid on;
% 
% %AUV
% % X
% plot(max_limit(:,1), max_limit(:,2),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,2),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_3(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,2),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,2),'color',[1,0,0]) ;
% hold off;
% 
% % Y
% subplot(4,2,3)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Y (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,3),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,3),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_3(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,3),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,3),'color',[1,0,0]) ;
% hold off;
% 
% % Z
% subplot(4,2,5)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Z (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,4),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,4),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_3(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,4),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,4),'color',[1,0,0]) ;
% hold off;
% 
% % Yaw
% subplot(4,2,7)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Yaw (rad)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,5),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,5),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_3(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,5),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,5),'color',[1,0,0]) ;
% hold off;
% 
% % End-effector
% % X
% subplot(4,2,2)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'X (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,6),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,6),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_3(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,3),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,6),'color',[1,0,0]) ;
% hold off;
% 
% % Y
% subplot(4,2,4)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Y (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,7),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,7),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_3(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,7),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,7),'color',[1,0,0]) ;
% hold off;
% 
% % Z
% subplot(4,2,6)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Z (m)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,8),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,8),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_3(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,8),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,8),'color',[1,0,0]) ;
% hold off;
% 
% % Yaw
% subplot(4,2,8)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Roll (rad)', 'FontSize',20)
% grid on;
% plot(max_limit(:,1), max_limit(:,11),'color',[0,0,1]) ;
% plot(min_limit(:,1), min_limit(:,11),'color',[0,0,1]) ;
% for n=1:nbRepro
%     ni = group_3(n);
%     plot(repro(ni).data(:,1), repro(ni).data(:,11),'color',[0,0,0]) ;
% end
% plot(avg_traj_goal(:,1), avg_traj_goal(:,11),'color',[1,0,0]) ;
% hold off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Selected Traj %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%nbRepro = length([1]);
% n_0 = 35; %Trajectory group 0
% n_1 = 8; %10; %Trajectory group 1
% n_2 = 14; %14; %Trajectory group 2
% n_3 = 25; %Trajectory group 3
% 
% figure()
% subplot(4,2,1)
% hold on
% title('Selected Traj','FontSize',20);
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'X (m)', 'FontSize',20)
% grid on;
% 
% %AUV
% % X
% plot(max_time_limit(:,2), max_limit(:,2),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(min_time_limit(1:end-10,2), min_limit(1:end-10,2),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(avg_traj_goal(:,1), avg_traj_goal(:,2),'LineStyle', '--','LineWidth', 2,'color',[1,0,0]) ;
% 
% 
% plot(repro(n_0).data(:,1), repro(n_0).data(:,2),'LineWidth', 2,'color', [1.0,0.5,0.0 ]) ;
% plot(repro(n_1).data(:,1), repro(n_1).data(:,2),'LineWidth', 2,'color',[0.13,0.55,0.13]) ;
% plot(repro(n_2).data(:,1), repro(n_2).data(:,2),'LineWidth', 2,'color',[0.8,0.8,0.0]) ;
% plot(repro(n_3).data(:,1), repro(n_3).data(:,2),'LineWidth', 2,'color',[0.58,0,0.83]) ;
% 
% hold off;
% 
% % Y
% subplot(4,2,3)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Y (m)', 'FontSize',20)
% grid on;
% plot(max_time_limit(1:end-5,3), max_limit(1:end-5,3),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(min_time_limit(:,3), min_limit(:,3),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(avg_traj_goal(:,1), avg_traj_goal(:,3),'LineStyle', '--','LineWidth', 2,'color',[1,0,0]) ;
% 
% 
% plot(repro(n_0).data(:,1), repro(n_0).data(:,3),'LineWidth', 2,'color', [1.0,0.5,0.0 ]) ;
% plot(repro(n_1).data(:,1), repro(n_1).data(:,3),'LineWidth', 2,'color',[0.13,0.55,0.13]) ;
% plot(repro(n_2).data(:,1), repro(n_2).data(:,3),'LineWidth', 2,'color',[0.8,0.8,0.0]) ;
% plot(repro(n_3).data(:,1), repro(n_3).data(:,3),'LineWidth', 2,'color',[0.58,0,0.83]) ;
% 
% hold off;
% 
% % Z
% subplot(4,2,5)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Z (m)', 'FontSize',20)
% grid on;
% plot(max_time_limit(:,4), max_limit(:,4),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(min_time_limit(:,4), min_limit(:,4),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(avg_traj_goal(:,1), avg_traj_goal(:,4),'LineStyle', '--','LineWidth', 2,'color',[1,0,0]) ;
% 
% 
% plot(repro(n_0).data(:,1), repro(n_0).data(:,4),'LineWidth', 2,'color', [1.0,0.5,0.0 ]) ;
% plot(repro(n_1).data(:,1), repro(n_1).data(:,4),'LineWidth', 2,'color',[0.13,0.55,0.13]) ;
% plot(repro(n_2).data(:,1), repro(n_2).data(:,4),'LineWidth', 2,'color',[0.8,0.8,0.0]) ;
% plot(repro(n_3).data(:,1), repro(n_3).data(:,4),'LineWidth', 2,'color',[0.58,0,0.83]) ;
% 
% hold off;
% 
% % Yaw
% subplot(4,2,7)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Yaw (rad)', 'FontSize',20)
% grid on;
% plot(max_time_limit(1:end-7,5), max_limit(1:end-7,5),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(min_time_limit(:,5), min_limit(:,5),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(avg_traj_goal(:,1), avg_traj_goal(:,5),'LineStyle', '--','LineWidth', 2,'color',[1,0,0]) ;
% 
% 
% plot(repro(n_0).data(:,1), repro(n_0).data(:,5),'LineWidth', 2,'color', [1.0,0.5,0.0 ]) ;
% plot(repro(n_1).data(:,1), repro(n_1).data(:,5),'LineWidth', 2,'color',[0.13,0.55,0.13]) ;
% plot(repro(n_2).data(:,1), repro(n_2).data(:,5),'LineWidth', 2,'color',[0.8,0.8,0.0]) ;
% plot(repro(n_3).data(:,1), repro(n_3).data(:,5),'LineWidth', 2,'color',[0.58,0,0.83]) ;
% 
% hold off;
% 
% % End-effector
% % X
% subplot(4,2,2)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'X (m)', 'FontSize',20)
% grid on;
% plot(max_time_limit(:,6), max_limit(:,6),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(min_time_limit(:,6), min_limit(:,6),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(avg_traj_goal(:,1), avg_traj_goal(:,6),'LineStyle', '--','LineWidth', 2,'color',[1,0,0]) ;
% 
% 
% plot(repro(n_0).data(:,1), repro(n_0).data(:,6),'LineWidth', 2,'color', [1.0,0.5,0.0 ]) ;
% plot(repro(n_1).data(:,1), repro(n_1).data(:,6),'LineWidth', 2,'color',[0.13,0.55,0.13]) ;
% plot(repro(n_2).data(:,1), repro(n_2).data(:,6),'LineWidth', 2,'color',[0.8,0.8,0.0]) ;
% plot(repro(n_3).data(:,1), repro(n_3).data(:,6),'LineWidth', 2,'color',[0.58,0,0.83]) ;
% 
% hold off;
% 
% % Y
% subplot(4,2,4)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Y (m)', 'FontSize',20)
% grid on;
% plot(max_time_limit(:,7), max_limit(:,7),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(min_time_limit(:,7), min_limit(:,7),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(avg_traj_goal(:,1), avg_traj_goal(:,7),'LineStyle', '--','LineWidth', 2,'color',[1,0,0]) ;
% 
% 
% plot(repro(n_0).data(:,1), repro(n_0).data(:,7),'LineWidth', 2,'color', [1.0,0.5,0.0 ]) ;
% plot(repro(n_1).data(:,1), repro(n_1).data(:,7),'LineWidth', 2,'color',[0.13,0.55,0.13]) ;
% plot(repro(n_2).data(:,1), repro(n_2).data(:,7),'LineWidth', 2,'color',[0.8,0.8,0.0]) ;
% plot(repro(n_3).data(:,1), repro(n_3).data(:,7),'LineWidth', 2,'color',[0.58,0,0.83]) ;
% 
% hold off;
% 
% % Z
% subplot(4,2,6)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Z (m)', 'FontSize',20)
% grid on;
% plot(max_time_limit(:,8), max_limit(:,8),'LineStyle', '--','LineWidth', 2.0,'color',[0,0,1]) ;
% plot(min_time_limit(:,8), min_limit(:,8),'LineStyle', '--','LineWidth', 2.0,'color',[0,0,1]) ;
% plot(avg_traj_goal(:,1), avg_traj_goal(:,8),'LineStyle', '--','LineWidth', 2.0,'color',[1,0,0]) ;
% 
% 
% plot(repro(n_0).data(:,1), repro(n_0).data(:,8),'LineWidth', 2,'color', [1.0,0.5,0.0 ]) ;
% plot(repro(n_1).data(:,1), repro(n_1).data(:,8),'LineWidth', 2,'color',[0.13,0.55,0.13]) ;
% plot(repro(n_2).data(:,1), repro(n_2).data(:,8),'LineWidth', 2,'color',[0.8,0.8,0.0]) ;
% plot(repro(n_3).data(:,1), repro(n_3).data(:,8),'LineWidth', 2,'color',[0.58,0,0.83]) ;
% 
% hold off;
% 
% % Yaw
% subplot(4,2,8)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Roll (rad)', 'FontSize',20)
% grid on;
% plot(max_time_limit(:,11), max_limit(:,11), 'LineStyle', '--', 'LineWidth', 2.0,'color',[0,0,1]) ;
% plot(min_time_limit(:,11), min_limit(:,11),'LineStyle', '--','LineWidth', 2.0,'color',[0,0,1]) ;
% plot(avg_traj_goal(:,1), avg_traj_goal(:,11),'LineStyle', '--','LineWidth', 2.0,'color',[1,0,0]) ;
% 
% plot(repro(n_0).data(:,1), repro(n_0).data(:,11),'LineWidth', 2,'color', [1.0,0.5,0.0 ]) ;
% plot(repro(n_1).data(:,1), repro(n_1).data(:,11),'LineWidth', 2,'color',[0.13,0.55,0.13]) ;
% plot(repro(n_2).data(:,1), repro(n_2).data(:,11),'LineWidth', 2,'color',[0.8,0.8,0.0]) ;
% plot(repro(n_3).data(:,1), repro(n_3).data(:,11),'LineWidth', 2,'color',[0.58,0,0.83]) ;
% hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Plot End-effector Traj %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% data_arm_n0 = importdata('ee_frame_auv_35.csv',',',1);
% data_arm_n1 = importdata('ee_frame_auv_8.csv',',',1);
% data_arm_n2 = importdata('ee_frame_auv_14.csv',',',1);
% data_arm_n3 = importdata('ee_frame_auv_25.csv',',',1);
% 
% n_0 = 35; %Trajectory group 0
% n_1 = 8; %10; %Trajectory group 1
% n_2 = 14; %14; %Trajectory group 2
% n_3 = 25; %Trajectory group 3
% 
% %subplot(2,1,2)
% figure()
% hold on;
% xlabel( 'Time (s)', 'FontSize',20);
% ylabel( 'X (m)', 'FontSize',20);
% grid on;
% 
% plot((data_arm_n0.data(:,1)-data_arm_n0.data(1,1))/(10^9), data_arm_n0.data(:,5),'LineWidth', 2,'color', [1.0,0.5,0.0 ]) ;
% plot((data_arm_n1.data(:,1)-data_arm_n1.data(1,1))/(10^9), data_arm_n1.data(:,5),'LineWidth', 2,'color',[0.13,0.55,0.13]) ;
% plot((data_arm_n2.data(:,1)-data_arm_n2.data(1,1))/(10^9), data_arm_n2.data(:,5),'LineWidth', 2,'color',[0.8,0.8,0.0]) ;
% plot((data_arm_n3.data(:,1)-data_arm_n3.data(1,1))/(10^9), data_arm_n3.data(:,5),'LineWidth', 2,'color',[0.58,0,0.83]) ;
% 
% 
% hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Selected Traj %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%nb_reza

% n_0 = 58;
% %n_1 = 58;
% 
% % End-effector
% % X
% figure()
% subplot(4,1,1)
% hold on ;
% title('Reactive Traj','FontSize',20);
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'X (m)', 'FontSize',20)
% grid on;
% plot(max_time_limit(:,6), max_limit(:,6),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(min_time_limit(:,6), min_limit(:,6),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(avg_traj_goal(:,1), avg_traj_goal(:,6),'LineStyle', '--','LineWidth', 2,'color',[1,0,0]) ;
% 
% plot(repro(n_0).data(:,1), repro(n_0).data(:,6),'LineWidth', 2,'color', [1.0,0.5,0.0 ]) ;
% 
% hold off;
% 
% % Y
% subplot(4,1,2)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Y (m)', 'FontSize',20)
% grid on;
% plot(max_time_limit(:,7), max_limit(:,7),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(min_time_limit(:,7), min_limit(:,7),'LineStyle', '--','LineWidth', 2,'color',[0,0,1]) ;
% plot(avg_traj_goal(:,1), avg_traj_goal(:,7),'LineStyle', '--','LineWidth', 2,'color',[1,0,0]) ;
% 
% plot(repro(n_0).data(:,1), repro(n_0).data(:,7),'LineWidth', 2,'color', [1.0,0.5,0.0 ]) ;
% 
% hold off;
% 
% % Z
% subplot(4,1,3)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Z (m)', 'FontSize',20)
% grid on;
% plot(max_time_limit(:,8), max_limit(:,8),'LineStyle', '--','LineWidth', 2.0,'color',[0,0,1]) ;
% plot(min_time_limit(:,8), min_limit(:,8),'LineStyle', '--','LineWidth', 2.0,'color',[0,0,1]) ;
% plot(avg_traj_goal(:,1), avg_traj_goal(:,8),'LineStyle', '--','LineWidth', 2.0,'color',[1,0,0]) ;
% 
% plot(repro(n_0).data(:,1), repro(n_0).data(:,8),'LineWidth', 2,'color', [1.0,0.5,0.0 ]) ;
% 
% hold off;
% 
% % Yaw
% subplot(4,1,4)
% hold on ;
% xlabel( 'Time (s)', 'FontSize',20)
% ylabel( 'Roll (rad)', 'FontSize',20)
% grid on;
% plot(max_time_limit(:,11), max_limit(:,11), 'LineStyle', '--', 'LineWidth', 2.0,'color',[0,0,1]) ;
% plot(min_time_limit(:,11), min_limit(:,11),'LineStyle', '--','LineWidth', 2.0,'color',[0,0,1]) ;
% plot(avg_traj_goal(:,1), avg_traj_goal(:,11),'LineStyle', '--','LineWidth', 2.0,'color',[1,0,0]) ;
% 
% plot(repro(n_0).data(:,1), repro(n_0).data(:,11),'LineWidth', 2,'color', [1.0,0.5,0.0 ]) ;
% 
% hold off;

end