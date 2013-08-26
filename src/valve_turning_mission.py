#!/usr/bin/env python

# ROS imports
import roslib
roslib.load_manifest('udg_pandora')
import rospy

#use to load the configuration function
import cola2_ros_lib

#include message to move forward or backward the arm
from std_msgs.msg import Float64
from std_msgs.msg import Bool

#To enable or disable the service
from std_srvs.srv import Empty, EmptyResponse
from udg_pandora.srv import WorkAreaError


class valveTurningMission:

    def __init__(self, name):
        self.name = name
        # self.getConfig()
        # self.state = 0
        self.RecoveringWorkArea = False
        rospy.Subscriber('/work_area/evaluation',
                         Float64,
                         self.updateWorkArea)
        rospy.Subscriber('learning/arm_finish',
                         Bool,
                         self.updateArmFinish)
        rospy.Subscriber('learning/auv_finish',
                         Bool,
                         self.updateAuvFinish)
        #AUV Trajectory services
        rospy.wait_for_service(
            '/learning/enable_reproductor_auv_traj_with_s')
        self.enable_auv_traj_s_srv = rospy.ServiceProxy(
            '/learning/enable_reproductor_auv_traj_with_s', WorkAreaError)
        rospy.wait_for_service(
            '/learning/enable_reproductor_auv_traj_with_s')
        self.enable_auv_traj_srv = rospy.ServiceProxy(
            '/learning/enable_reproductor_auv_traj', Empty)
        rospy.wait_for_service(
            '/learning/disable_reproductor_auv_traj')
        self.disable_auv_traj_srv = rospy.ServiceProxy(
            '/learning/disable_reproductor_auv_traj', Empty)

        rospy.loginfo('AUV Services loaded')

        #NTUA Control Activate
        rospy.wait_for_service(
            '/ntua_control_g500/enable_valve_controller')
        self.enable_ntua_srv = rospy.ServiceProxy(
            '/ntua_control_g500/enable_valve_controller', Empty)
        rospy.wait_for_service(
            '/ntua_control_g500/enable_valve_controller')
        self.disable_ntua_srv = rospy.ServiceProxy(
            '/ntua_control_g500/enable_valve_controller', Empty)

        rospy.loginfo('NTUA Services loaded')

        #Enable Arm Control
        rospy.wait_for_service(
            '/learning/enable_reproductor_arm')
        self.enable_arm_srv = rospy.ServiceProxy(
            '/learning/enable_reproductor_arm', Empty)
        rospy.wait_for_service(
            '/learning/disable_reproductor_arm')
        self.disable_arm_srv = rospy.ServiceProxy(
            '/learning/disable_reproductor_arm', Empty)

        rospy.loginfo('Arm Services loaded')

        #Enable Work Area
        rospy.wait_for_service(
            '/learning/enable_work_area')
        self.enable_work_area_srv = rospy.ServiceProxy(
            '/learning/enable_work_area', Empty)
        rospy.wait_for_service(
            '/learning/disable_work_area')
        self.disable_work_area_srv = rospy.ServiceProxy(
            '/learning/disable_work_area', Empty)

        rospy.loginfo('Work Area Services loaded')

        #Sevice Turn 90
        # rospy.wait_for_service(
        #     '/arm/turn_90_degres')
        # self.turn_90_degres_srv = rospy.ServiceProxy(
        #     '/arm/turn_90_degres', Empty)

    # def getConfig(self):
    #     param_dict = {}
    #     cola2_ros_lib.getRosParams(self, param_dict)

    def updateWorkArea(self, evaluation):
        if evaluation < 0 and not self.RecoveringWorkArea:
            rospy.loginfo('Outsite the working area')
            try:
                self.disable_ntua_srv()
                s_value = WorkAreaError()
                s_value.error = -1*evaluation
                self.enable_auv_traj_s_srv(s_value)
                self.disable_ntua_srv()
                rospy.loginfo('The AUV is moving toward The Valve Panel')
            except rospy.ServiceException, e:
                print "Service call failed: %s" %e

    def updateArmFinish(self, data):
        if data:
            self.disable_arm_srv()
            #self.turn_90_degres()

    def updateAuvFinish(self, data):
        if data:
            try:
                self.disable_auv_traj_srv()
                self.enable_ntua_srv()
                rospy.loginfo('Auv Finis, The NTUA controller Enabled')
                #TODO:: Sleep
                rospy.sleep(5.0)
                self.enable_work_area_srv()
                self.enable_arm_srv()
                rospy.loginfo('The Grasping has started')
                if self.RecoveringWorkArea:
                    self.RecoveringWorkArea = False
                rospy.loginfo('The manipulation Begins')
            except rospy.ServiceException, e:
                print "Service call failed: %s" %e

    def startAuv(self):
        rospy.loginfo('Mision start AUV Moving !!!!!')
        self.enable_auv_traj_srv()

if __name__ == '__main__':
    try:
        #Load the configuration file
        # import subprocess
        # config_file_list = roslib.packages.find_resource(
        #     "udg_pandora", "valve_turning_mission.yaml")
        # if len(config_file_list):
        #     config_file = config_file_list[0]
        #     subprocess.call(["rosparam", "load", config_file])
        # else:
        #     rospy.logerr("Could not locate valve_turning_mission.yaml")

        rospy.init_node('valve_turning_mission')
        valve_turning_mission = valveTurningMission(rospy.get_name())
        rospy.loginfo('Mision Initializad')
        valve_turning_mission.startAuv()
#        learning_reproductor.play()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
