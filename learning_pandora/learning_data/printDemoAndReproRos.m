function printDemoAndReproRos

clear all;
close all;

listSamples = [0,1,2,3,4,5] ;
nbSamples = length(listSamples);

listSamples2 = [0,1,3,4,5,6] ;
nbSamples2 = length(listSamples2);


%figure()
subplot(2,3,1)
hold on ;
title('Trajectory AUV (Top View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;
% zlabel( 'Z', 'FontSize',20)
% traj_played = importdata('trajectoryPlayed.csv', ' ', 1) ;
% plot3(traj_played.data(:,3),traj_played.data(:,1),traj_played.data(:,2),'-','linewidth',2.0,'color',[1 0 0]) ;
%  
% real_traj = importdata('real_traj.csv',' ',1) ;
% plot3(real_traj.data(:,3),real_traj.data(:,1),real_traj.data(:,2),'-','linewidth',2.0,'color',[0 1 0]) ;
% plot3(real_traj.data(end,3),real_traj.data(end,1),real_traj.data(end,2),'.','markerSize',20,'color',[0 1 0]) ;

% real_traj = importdata('real_traj_bastant_bona.csv',' ', 1);
% plot(real_traj.data(:,3),real_traj.data(:,1),'LineWidth', 3, 'color', [0,0,1]);
% plot(real_traj.data(end,3),real_traj.data(end,1),'.','markerSize',20,'color', [1,0,1]);


%plot(real_traj.data(end,3),real_traj.data(end,1),'.','markerSize',20,'color', [1,0,1]);

% real_traj = importdata('real_traj_perturbation_0.csv',' ', 1);
% plot(real_traj.data(:,3),real_traj.data(:,1),'LineWidth', 3, 'color', [0,0,1]);
% plot(real_traj.data(end,3),real_traj.data(end,1),'.','markerSize',20,'color', [1,0,1]);


% for n=1:nbSamples2
%     ni = listSamples2(n) ;
%     sample = importdata(['real_traj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0.5,0.75,0]) ;
%     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
% end

% Z i X
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
%     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
    %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
    %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
end
real_traj = importdata('real_traj_bastant_bona.csv',' ', 1);
%real_traj = importdata('real_traj_perturbacio.csv',' ', 1);
plot(real_traj.data(:,3),real_traj.data(:,1),'LineWidth', 4, 'color', [1,0,0]);
hold off;


%figure()
subplot(2,3,2)
hold on ;
title('Trajectory AUV (Top Side)', 'FontSize', 40)
xlabel( 'Z (m)', 'FontSize',30)
ylabel( 'Y (m)', 'FontSize',30)
grid on;


% real_traj = importdata('real_traj_bastant_bona.csv',' ', 1);
% plot(real_traj.data(:,3),real_traj.data(:,2),'LineWidth', 3, 'color', [0,0,1]);
% plot(real_traj.data(end,3),real_traj.data(end,2),'.','markerSize',20,'color', [1,0,1]);

% plot(real_traj.data(end,3),real_traj.data(end,2),'.','markerSize',20,'color', [1,0,1]);
% real_traj = importdata('real_traj_perturbation_0.csv',' ', 1);
% plot(real_traj.data(:,3),real_traj.data(:,2),'LineWidth', 3, 'color', [0,0,1]);
% plot(real_traj.data(end,3),real_traj.data(end,2),'.','markerSize',20,'color', [1,0,1]);

% Z i Y
for n=1:nbSamples 
    ni = listSamples(n) ;
    sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,3), sample.data(:,2), 'color',[0,0,0]);
%      plot(sample.data(end,3), sample.data(end,2), '.', 'markerSize',20,'color',[1,0,0]) ;
    %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
    %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
end
real_traj = importdata('real_traj_bastant_bona.csv',' ', 1);
%real_traj = importdata('real_traj_perturbacio.csv',' ', 1);
plot(real_traj.data(:,3),real_traj.data(:,2),'LineWidth', 4, 'color', [1,0,0]);

hold off;

%figure()
subplot(2,3,3)
hold on ;
title('Trajectory AUV (Yaw, orientation)')
ylabel( 'Yaw (rad)', 'FontSize',20)
grid on;

% real_traj = importdata('real_traj_bastant_bona.csv',' ', 1);
% plot(real_traj.data(:,4),'LineWidth', 3, 'color', [0,0,1]);

% real_traj = importdata('real_traj_perturbation_0.csv',' ', 1);
% plot(real_traj.data(:,4),'LineWidth', 3, 'color', [0,0,1]);
%plot(real_traj.data(end,4),'.','markerSize',20,'color', [1,0,1]);
% Z i Y
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    %plot(sample.data(:,4), 'color',[0,0,0]);
    new_vector = zeros(round(length(sample.data(:,4))*2/3),1);
    count = 1;
    count2 = 1;
    for m=1:length(sample.data(:,4))
        if count < 3
            if count2 <= round(length(sample.data(:,4))*2/3)
                new_vector(count2) = sample.data(m,4);
            end
            count = count + 1;
            count2 = count2 +1;
        else
            count = 1;
        end
    end

    plot(new_vector, 'color',[0,0,0]);

    %plot(sample.data(end,4), '.', 'markerSize',20,'color',[1,0,0]) ;
    %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
    %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
end
real_traj = importdata('real_traj_bastant_bona.csv',' ', 1);
%real_traj = importdata('real_traj_perturbacio.csv',' ', 1);
plot(real_traj.data(:,4)-0.05,'LineWidth', 4, 'color', [1,0,0]);
hold off;



%figure()
subplot(2,3,4)
hold on ;
title('Trajectory ARM (Top View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;
% zlabel( 'Z', 'FontSize',20)
% traj_played = importdata('trajectoryPlayed.csv', ' ', 1) ;
% plot3(traj_played.data(:,3),traj_played.data(:,1),traj_played.data(:,2),'-','linewidth',2.0,'color',[1 0 0]) ;
%  
% real_traj = importdata('real_traj.csv',' ',1) ;
% plot3(real_traj.data(:,3),real_traj.data(:,1),real_traj.data(:,2),'-','linewidth',2.0,'color',[0 1 0]) ;
% plot3(real_traj.data(end,3),real_traj.data(end,1),real_traj.data(end,2),'.','markerSize',20,'color',[0 1 0]) ;

% real_traj = importdata('real_traj_bastant_bona.csv',' ', 1);
% plot(real_traj.data(:,7),real_traj.data(:,5), 'LineWidth', 3,'color', [0,0,1]);
% plot(real_traj.data(end,7),real_traj.data(end,5),'.','markerSize',20,'color', [1,0,1]);

% plot(real_traj.data(end,7),real_traj.data(end,5),'.','markerSize',20,'color', [1,0,1]);
% real_traj = importdata('real_traj_perturbation_0.csv',' ', 1);
% plot(real_traj.data(:,7),real_traj.data(:,5), 'LineWidth', 3,'color', [0,1,0]);
% plot(real_traj.data(end,7),real_traj.data(end,5),'.','markerSize',20,'color', [1,0,1]);
% Z i X
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,7), sample.data(:,5), 'color',[0,0,0]) ;
    %plot(sample.data(end,7), sample.data(end,5), '.', 'markerSize',25,'color',[1,0,0]) ;
    %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
    %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
end
real_traj = importdata('real_traj_bastant_bona.csv',' ', 1);
%real_traj = importdata('real_traj_perturbacio.csv',' ', 1);
plot(real_traj.data(:,7),real_traj.data(:,5), 'LineWidth', 4,'color', [1,0,0]);

hold off;


%figure()
subplot(2,3,5)
hold on ;
title('Trajectory ARM (Side View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;
% real_traj = importdata('real_traj_bastant_bona.csv',' ', 1);
% plot(real_traj.data(:,7),real_traj.data(:,6),'LineWidth', 3, 'color', [0,0,1]);
% plot(real_traj.data(end,7),real_traj.data(end,6),'.','markerSize',20,'color', [1,0,1]);

%plot(real_traj.data(end,7),real_traj.data(end,6),'.','markerSize',20,'color', [1,0,1]);
% real_traj = importdata('real_traj_perturbation_0.csv',' ', 1);
% plot(real_traj.data(:,7),real_traj.data(:,6),'LineWidth', 3, 'color', [0,1,0]);
% plot(real_traj.data(end,7),real_traj.data(end,6),'.','markerSize',20,'color', [1,0,1]);
% Z i Y
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,7), sample.data(:,6), 'color',[0,0,0]);
%     plot(sample.data(end,7), sample.data(end,6), '.', 'markerSize',25,'color',[1,0,0]) ;
    %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
    %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
end
real_traj = importdata('real_traj_bastant_bona.csv',' ', 1);
%real_traj = importdata('real_traj_perturbacio.csv',' ', 1);
plot(real_traj.data(:,7),real_traj.data(:,6),'LineWidth', 4, 'color', [1,0,0]);

hold off;

%figure()
subplot(2,3,6)
hold on ;
title('Trajectory ARM (Roll end-effector, aligment)')
ylabel( 'Roll (rad)', 'FontSize',20)
grid on;


% real_traj = importdata('real_traj_bastant_bona.csv',' ', 1);
% plot(real_traj.data(:,10),'LineWidth', 3, 'color', [0,1,0]);
% real_traj = importdata('real_traj_perturbation_0.csv',' ', 1);
% plot(real_traj.data(:,10),'LineWidth', 3, 'color', [0,1,0]);
% Z i Y
% for n=1:nbSamples
%     ni = listSamples(n) ;
%     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,10), 'color',[0,0,0]);
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
% hold off;

for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    %plot(sample.data(:,4), 'color',[0,0,0]);
    new_vector = zeros(round(length(sample.data(:,10))*2/3),1);
    count = 1;
    count2 = 1;
    for m=1:length(sample.data(:,10))
        if count < 3
            if count2 <= round(length(sample.data(:,10))*2/3)
                new_vector(count2) = sample.data(m,10);
            end
            count = count + 1;
            count2 = count2 +1;
        else
            count = 1;
        end
    end

    plot(new_vector, 'color',[0,0,0]);

    %plot(sample.data(end,4), '.', 'markerSize',20,'color',[1,0,0]) ;
    %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
    %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
end
real_traj = importdata('real_traj_bastant_bona.csv',' ', 1);
%real_traj = importdata('real_traj_perturbacio.csv',' ', 1);
plot(real_traj.data(:,10),'LineWidth', 4, 'color', [1,0,0]);
hold off;

%%%% ARM 
% figure()
% hold on ;
% title('Trajectory AUV')
% xlabel( 'X', 'FontSize',20)
% ylabel( 'Y', 'FontSize',20)
% zlabel( 'Z', 'FontSize',20)
% 
% %traj_played = importdata('trajectoryPlayed.csv', ' ', 1) ;
% %plot3(traj_played.data(:,3),traj_played.data(:,1),traj_played.data(:,2),'-','linewidth',2.0,'color',[1 0 0]) ;
% 
% % real_traj = importdata('real_traj.csv',' ',1) ;
% % plot3(real_traj.data(:,5),real_traj.data(:,6),real_traj.data(:,7),'-','linewidth',2.0,'color',[0 1 0]) ;
% % plot3(real_traj.data(end,5),real_traj.data(end,6),real_traj.data(end,7),'.','markerSize',20,'color',[1 0 0]) ;
% 
% for n=1:nbSamples
%     ni = listSamples(n) ;
%     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot3(sample.data(:,5),sample.data(:,6),sample.data(:,7),'-','linewidth',2.0,'color',[0 0 0]) ;
%     plot3(sample.data(end,5),sample.data(end,6),sample.data(end,7), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
% 
% hold off;

end