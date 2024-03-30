#!/usr/bin/python3
import casadi
import rospy
import numpy as np
import mpc_algo 
import math
import time

from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped,PolygonStamped, Quaternion
from visualization_msgs.msg import Marker

# import tf

class ROSNode():
    def __init__(self):
        self.TOPIC_VEL = "/cmd_vel"
        self.TOPIC_GLOBAL_PLAN = "/move_base/TrajectoryPlannerROS/global_plan"
        self.TOPIC_LOCAL_PLAN = "/move_base/TrajectoryPlannerROS/local_plan"
        self.TOPIC_ODOM = "/odometry/filtered"
        self.TOPIC_MPC_PLAN = "/mpc_plan"
        
        self.pub_vel = rospy.Publisher(self.TOPIC_VEL, Twist, queue_size=1, latch=True)
        self.pub_mpc  = rospy.Publisher(self.TOPIC_MPC_PLAN, Path, queue_size=1)
        
        self.sub_odometry = rospy.Subscriber(self.TOPIC_ODOM, Odometry, self.callback_odom)
        self.sub_global_plan = rospy.Subscriber(self.TOPIC_GLOBAL_PLAN, Path, self.callback_global_plan)
        self.sub_local_plan = rospy.Subscriber(self.TOPIC_LOCAL_PLAN, Path, self.callback_local_plan)

        self.cmd_vel = Twist()
        self.odometry = Odometry()
        self.global_plan = Path()
        self.local_plan = Path()
        
        self.mpc = mpc_algo.NMPC()
        self.v_opt = 0 
        self.w_opt = 0
        
        self.rate = 20
        
    
    def callback_odom(self,data):
        self.odometry = data
        yaw = self.quaternion_to_yaw(data.pose.pose.orientation)
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        v = data.twist.twist.linear.x
        w = data.twist.twist.angular.z
        vr = self.v_opt - (self.w_opt * self.mpc.L)/2
        vl = self.v_opt + (self.w_opt * self.mpc.L)/2
        self.X0 = [x,y,yaw,vr, vl]

    def callback_global_plan(self,data):
        self.global_plan = data
        self.x_ref = [pose.pose.position.x for pose in self.global_plan.poses[5::2]]
        self.y_ref = [pose.pose.position.y for pose in self.global_plan.poses[5::2]]
        # print("Global poses: ",len(data.poses))
        # print("Global")
        
    def callback_local_plan(self, data):
        self.local_plan = data
        # print("Local")

    def quaternion_to_yaw(self, orientation):
    # Convert quaternion orientation data to yaw angle of robot
        q0 = orientation.x
        q1 = orientation.y
        q2 = orientation.z
        q3 = orientation.w
        theta = math.atan2(2.0*(q2*q3 + q0*q1), 1.0 - 2.0*(q1*q1 + q2*q2))
        return theta
    
    def publish_trajectory(self, mpc_x_traj, mpc_y_traj):
        mpc_traj_msg = Path()
        mpc_traj_msg.header.stamp = rospy.Time.now()
        mpc_traj_msg.header.frame_id = "odom"
        for i in range(mpc_x_traj.shape[0]):
            pose = PoseStamped()
            pose.pose.position.x = mpc_x_traj[i]
            pose.pose.position.y = mpc_y_traj[i]
            pose.pose.orientation = Quaternion(0,0,0,1)
            mpc_traj_msg.poses.append(pose)
            
        self.pub_mpc.publish(mpc_traj_msg)

    def publish_velocity(self, v_opt, w_opt):
        vel = Twist()
        vel.linear.x = v_opt
        vel.angular.z = w_opt
        self.pub_vel.publish(vel)

    def run(self):
        try:
            if len(self.x_ref) > 5:
                if len(self.x_ref) > self.mpc.N_ref:
                    self.mpc.N = self.mpc.N_ref
                else: 
                    self.mpc.N = len(self.x_ref)-1
                
                # Setup the MPC
                #TODO: Do this
                self.mpc.setup(self.rate)
                # solve
                self.v_opt, self.w_opt, solve_time = self.mpc.solve(self.x_ref, self.y_ref, self.X0) # Return the optimization variables
                # Control and take only the first step 
                
                rospy.loginfo("Solve time: " + str(solve_time))
                self.publish_velocity(self.v_opt, self.w_opt)
                # self.publish_velocity(0, 0.5)

                
                # Get from the MPC results
                mpc_x_traj = self.mpc.opt_states[0::self.mpc.n]
                mpc_y_traj = self.mpc.opt_states[1::self.mpc.n]
                # print(type(mpc_x_traj), mpc_x_traj.shape)
                self.publish_trajectory(mpc_x_traj, mpc_y_traj)
            else:
                print("Stopped")
                self.publish_velocity(0,0)
        except Exception as e:
            rospy.logerr(e)
    
if __name__ =="__main__":
    rospy.init_node("nmpc")
    rospy.loginfo("Non-Linear MPC Node running")
    node = ROSNode()
    pause = rospy.Rate(node.rate) # 10 Hz
    time.sleep(1)
    while not rospy.is_shutdown():
        node.run()
        pause.sleep()