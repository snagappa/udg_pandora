function printDemoAndReproductions

clear all;
close all;

listSamples = [0] ;
nbSamples = length(listSamples);

% display(real_traj.data(end,11)-real_traj.data(1,11))

figure()
subplot(3,1,1)
hold on ;
title('Trajectory AUV Panel (Top View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;

% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_panel_centre_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,2),'color',[0,0,1]) ;
end

sim_traj = importdata(['sim_traj_auv_panel_centre.csv' ], ' ', 1) ;
plot(sim_traj.data(:,4), sim_traj.data(:,2),'color',[1,0,0]) ;

hold off;

subplot(3,1,2)
hold on ;
title('Trajectory AUV Panel (Side View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_panel_centre_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,3),'color',[0,0,1]) ;
end

sim_traj = importdata(['sim_traj_auv_panel_centre.csv' ], ' ', 1) ;
plot(sim_traj.data(:,4), sim_traj.data(:,3),'color',[1,0,0]) ;

hold off


subplot(3,1,3)
hold on ;
title('Ori AUV Panel (Side View)')
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Ori (rad)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_panel_centre_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,5),'color',[0,0,1]) ;
end

sim_traj = importdata(['sim_traj_auv_panel_centre.csv' ], ' ', 1) ;
plot(sim_traj.data(:,4), sim_traj.data(:,2),'color',[1,0,0]) ;

hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure()
subplot(3,1,1)
hold on ;
title('Trajectory AUV Valve 0 (Top View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_valve_0_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,2),'color',[0,0,1]) ;
end

hold off;

subplot(3,1,2)
hold on ;
title('Trajectory AUV Valve 0 (Side View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_valve_0_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,3),'color',[0,0,1]) ;
end

subplot(3,1,3)
hold on ;
title('Ori AUV Valve 0 (Side View)')
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Ori (rad)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_valve_0_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,1), sample.data(:,5),'color',[0,0,1]) ;
end
hold off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure()
subplot(3,1,1)
hold on ;
title('Trajectory AUV Valve 1 (Top View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_valve_1_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,2),'color',[0,0,1]) ;
end

hold off;

subplot(3,1,2)
hold on ;
title('Trajectory AUV Valve 1 (Side View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_valve_1_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,3),'color',[0,0,1]) ;
end


subplot(3,1,3)
hold on ;
title('Ori AUV Valve 1 (Side View)')
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Ori (rad)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_valve_1_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,1), sample.data(:,5),'color',[0,0,1]) ;
end
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure()
subplot(3,1,1)
hold on ;
title('Trajectory AUV Valve 2 (Top View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_valve_2_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,2),'color',[0,0,1]) ;
end

sim_traj = importdata(['sim_traj_auv_valve_2.csv' ], ' ', 1) ;
plot(sim_traj.data(:,4), sim_traj.data(:,2),'color',[1,0,0]) ;

hold off;

subplot(3,1,2)
hold on ;
title('Trajectory AUV Valve 2 (Side View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_valve_2_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,3),'color',[0,0,1]) ;
end

sim_traj = importdata(['sim_traj_auv_valve_2.csv' ], ' ', 1) ;
plot(sim_traj.data(:,4), sim_traj.data(:,3),'color',[1,0,0]) ;


subplot(3,1,3)
hold on ;
title('Ori AUV Valve 2 (Side View)')
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Ori (rad)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_valve_2_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,1), sample.data(:,5),'color',[0,0,1]) ;
end
hold off

sim_traj = importdata(['sim_traj_auv_valve_2.csv' ], ' ', 1) ;
plot(sim_traj.data(:,1), sim_traj.data(:,5),'color',[1,0,0]) ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure()
subplot(3,1,1)
hold on ;
title('Trajectory AUV Valve 3 (Top View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_valve_3_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,2),'color',[0,0,1]) ;
end

hold off;

subplot(3,1,2)
hold on ;
title('Trajectory AUV Valve 3 (Side View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_valve_3_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,3),'color',[0,0,1]) ;
end


subplot(3,1,3)
hold on ;
title('Ori AUV Valve 3 (Side View)')
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Ori (rad)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_valve_3_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,1), sample.data(:,5),'color',[0,0,1]) ;
end
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure()
subplot(3,1,1)
hold on ;
title('Trajectory EE Panel (Top View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_panel_centre_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,2),'color',[0,0,1]) ;
end

hold off;

subplot(3,1,2)
hold on ;
title('Trajectory EE Panel (Side View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_panel_centre_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,3),'color',[0,0,1]) ;
end

subplot(3,1,3)
hold on ;
title('Ori EE Valve panel centre')
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Ori (rad)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_panel_centre_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,1), sample.data(:,5),'color',[0,0,1]) ;
end
hold off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure()
subplot(3,1,1)
hold on ;
title('Trajectory EE Valve 0 (Top View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_valve_0_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,2),'color',[0,0,1]) ;
end

hold off;

subplot(3,1,2)
hold on ;
title('Trajectory EE Valve 0 (Side View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_valve_0_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,3),'color',[0,0,1]) ;
end

subplot(3,1,3)
hold on ;
title('Ori AUV Valve 0')
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Ori (rad)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_valve_0_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,1), sample.data(:,5),'color',[0,0,1]) ;
end
hold off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure()
subplot(3,1,1)
hold on ;
title('Trajectory EE Valve 1 (Top View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_valve_1_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,2),'color',[0,0,1]) ;
end

hold off;

subplot(3,1,2)
hold on ;
title('Trajectory EE Valve 1 (Side View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_valve_1_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,3),'color',[0,0,1]) ;
end


subplot(3,1,3)
hold on ;
title('Ori AUV Valve 2 (Side View)')
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Ori (rad)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_valve_2_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,1), sample.data(:,5),'color',[0,0,1]) ;
end
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure()
subplot(3,1,1)
hold on ;
title('Trajectory EE Valve 2 (Top View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_valve_2_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,2),'color',[0,0,1]) ;
end

hold off;

subplot(3,1,2)
hold on ;
title('Trajectory ee Valve 2 (Side View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_valve_2_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,3),'color',[0,0,1]) ;
end

subplot(3,1,3)
hold on ;
title('Ori AUV Valve 2 (Side View)')
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Ori (rad)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_valve_2_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,1), sample.data(:,5),'color',[0,0,1]) ;
end
hold off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure()
subplot(3,1,1)
hold on ;
title('Trajectory EE Valve 3 (Top View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'X (m)', 'FontSize',20)
grid on;


% Z i X
% for n=1:nbSamples
%     ni = listSamples(n) ;
% %     sample = importdata(['trajectory_demonstration_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     sample = importdata(['smoothtraj_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
%     plot(sample.data(:,3), sample.data(:,1),'color',[0,0,0]) ;
% %     plot(sample.data(end,3), sample.data(end,1), '.', 'markerSize',20,'color',[1,0,0]) ;
%     %plot3(sample.data(:,3),sample.data(:,1),sample.data(:,2),'-','linewidth',2.0,'color',[0 0 0]) ;
%     %plot3(sample.data(end,3),sample.data(end,1),sample.data(end,2), '.', 'markerSize',25,'color',[1 0 0]) ;
% end
for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_valve_3_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,2),'color',[0,0,1]) ;
end

hold off;

subplot(3,1,2)
hold on ;
title('Trajectory EE Valve 3 (Side View)')
xlabel( 'Z (m)', 'FontSize',20)
ylabel( 'Y (m)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_valve_3_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,4), sample.data(:,3),'color',[0,0,1]) ;
end

subplot(3,1,3)
hold on ;
title('Ori AUV Valve 3 (Side View)')
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Ori (rad)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_valve_3_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,1), sample.data(:,5),'color',[0,0,1]) ;
end
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure()
hold on ;
title('AUV vs world')
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Distance (m)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_auv_world_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,2),'color',[1,0,0]) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,3),'color',[0,1,0]) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,4),'color',[0,0,1]) ;
end
sample = importdata('sim_traj_auv_world.csv', ' ', 1) ;
plot(sample.data(:,1)-sample.data(1,1), sample.data(:,2),'.','color',[1,0,0]) ;
plot(sample.data(:,1)-sample.data(1,1), sample.data(:,3),'.','color',[0,1,0]) ;
plot(sample.data(:,1)-sample.data(1,1), sample.data(:,4),'.','color',[0,0,1]) ;
hold off

figure()
hold on ;
title('EE vs world')
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Distance (m)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_world_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,2),'color',[1,0,0]) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,3),'color',[0,1,0]) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,4),'color',[0,0,1]) ;
end
hold off

figure()
hold on ;
title('EE vs AUV')
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Distance (m)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['traj_ee_auv_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,2),'color',[1,0,0]) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,3),'color',[0,1,0]) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,4),'color',[0,0,1]) ;
end

% sample = importdata('sim_traj_ee_auv.csv', ' ', 1) ;
% plot(sample.data(:,1)-sample.data(1,1), sample.data(:,2), '.', 'color',[1,0,0]) ;
% plot(sample.data(:,1)-sample.data(1,1), sample.data(:,3),'.','color',[0,1,0]) ;
% plot(sample.data(:,1)-sample.data(1,1), sample.data(:,4),'.', 'color',[0,0,1]) ;

hold off

figure()
hold on ;
title('Force')
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Force (Newton)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['force_world_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,2),'color',[1,0,0]) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,3),'color',[0,1,0]) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,4),'color',[0,0,1]) ;
end
hold off


figure()
hold on ;
title('Torque')
xlabel( 'Time (s)', 'FontSize',20)
ylabel( 'Torque (Newton)', 'FontSize',20)
grid on;


for n=1:nbSamples
    ni = listSamples(n) ;
    sample = importdata(['force_world_' num2str(ni,'%2d') '.csv' ], ' ', 1) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,5),'color',[1,0,0]) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,6),'color',[0,1,0]) ;
    plot(sample.data(:,1)-sample.data(1,1), sample.data(:,7),'color',[0,0,1]) ;
end
hold off


end

