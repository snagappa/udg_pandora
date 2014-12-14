#! /usr/bin/env python

import roslib
roslib.load_manifest('learning_pandora')
import rospy

# Brings in the SimpleActionClient
import actionlib

# Brings in the messages used by the fibonacci action, including the
# goal message and the result message.
import learning_pandora.msg

import numpy as np

def learning_client():
    # Creates the SimpleActionClient, passing the type of the action
    # (FibonacciAction) to the constructor.
    client = actionlib.SimpleActionClient('/learning/valve_turning_action',
                                          learning_pandora.msg.ValveTurningAction)


    # Waits until the action server has started up and started
    # listening for goals.
    client.wait_for_server()
    rospy.on_shutdown(client.cancel_goal)
    # Creates a goal to send to the action server.
    # goal = udg_pandora.msg.ValveTurningAction(valve_id=2, long_approach=False)
    goal = learning_pandora.msg.ValveTurningGoal()
    goal.valve_id = 2
    goal.long_approach = False
    goal.desired_increment = np.deg2rad(-90)

    # Sends the goal to the action server.
    client.send_goal(goal)

    # Waits for the server to finish performing the action.
    client.wait_for_result()
    # Prints out the result of executing the action
    return client.get_result()  # A FibonacciResult

if __name__ == '__main__':
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
        rospy.init_node('learning_client_py')

        result = learning_client()
        rospy.loginfo('Result ' + str(result))
    except rospy.ROSInterruptException:
        print "program interrupted before completion"
