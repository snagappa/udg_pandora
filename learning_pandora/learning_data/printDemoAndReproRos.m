function printDemoAndReproRos

clear all;
close all;

listSamples = [0,1,2,3] ;
nbSamples = length(listSamples);

%figure()
subplot(2,1,1)
hold on ;
title('Trajectory AUV (Top View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;


% Z i X
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
%     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
    %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
    %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
end
real_traj = importdata('real_traj.csv',' ', 1);
plot(real_traj.data(:,3),real_traj.data(:,1),'LineWidth', 4, 'color', [1,0,0]);
plot(real_traj.data(1,3),real_traj.data(1,1),'.','markerSize', 20, 'color', [0,1,0]);
hold off;


%figure()
subplot(2,1,2)
hold on ;
title('Trajectory AUV (Top Side)', 'FontSize', 40)
xlabel( 'Z (m)', 'FontSize',30)
ylabel( 'Y (m)', 'FontSize',30)
grid on;


% Z i Y
for n=1:nbSamples 
    ni = listSamples(n) ;
    sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,3), sample.data(:,2), 'color',[0,0,0]);
%      plot(sample.data(end,3), sample.data(end,2), '.', 'markerSize',20,'color',[1,0,0]) ;
    %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
    %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
end
real_traj = importdata('real_traj.csv',' ', 1);
plot(real_traj.data(:,3),real_traj.data(:,2),'LineWidth', 4, 'color', [1,0,0]);
plot(real_traj.data(1,3),real_traj.data(1,2),'.','markerSize', 20, 'color', [0,1,0]);
hold off;


figure()
subplot(3,1,1)
hold on;
title('Roll')
xlabel('Time')
ylabel('Degrees(rad)')
grid on;
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4),'color',[0,0,0]) ;
%     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
    %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
    %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
end
real_traj = importdata('real_traj.csv',' ', 1);
plot(real_traj.data(:,4),'LineWidth', 4, 'color', [1,0,0]);

subplot(3,1,2)
hold on;
title('Pitch')
xlabel('Time')
ylabel('Degrees(rad)')
grid on;
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,5),'color',[0,0,0]) ;
%     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
    %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
    %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
end
real_traj = importdata('real_traj.csv',' ', 1);
plot(real_traj.data(:,5),'LineWidth', 4, 'color', [1,0,0]);


subplot(3,1,3)
hold on;
title('Yaw')
xlabel('Time')
ylabel('Degrees(rad)')
grid on;
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,6),'color',[0,0,0]) ;
%     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
    %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
    %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
end
real_traj = importdata('real_traj.csv',' ', 1);
plot(real_traj.data(:,6),'LineWidth', 4, 'color', [1,0,0]);



end