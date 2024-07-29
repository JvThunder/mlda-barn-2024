#!/usr/bin/python
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, PolygonStamped
from nav_msgs.msg import Path, Odometry
from jackal_helper.msg import ResultData

import numpy as np
from tf.transformations import euler_from_quaternion
from tf_conversions import Quaternion
import csv
import os

INF_CAP = 10000
class Inspection():
    def __init__(self):
        # Topic Defintions
        self.TOPIC_FRONT_SCAN = "/front/scan" # Front Laser Scan (LIDAR)
        self.TOPIC_ODOM = "/odometry/filtered" # Odometry (pose and twist)
        self.TOPIC_CMD_VEL = "/cmd_vel" # Command Velocity (action)

        self.TOPIC_LOCAL_PLAN = "/move_base/TrajectoryPlannerROS/local_plan" # Local plan
        self.TOPIC_LOCAL_FOOTPRINT = "/move_base/local_costmap/footprint" # the bounding box of the robot
        self.TOPIC_GLOBAL_PLAN = "/move_base/TrajectoryPlannerROS/global_plan" # Global plan
        self.TOPIC_MPC = "/mpc_plan" # MPC plan
        self.RESULT_DATA = "/result_data" # Result Data
        
        # Object to store
        self.scan = LaserScan()
        self.cmd_vel = Twist()
        self.global_plan = Path()
        self.local_plan = Path()
        self.odometry = Odometry()
        self.footprint = PolygonStamped() 

        # Subscribe        
        self.sub_front_scan = rospy.Subscriber(self.TOPIC_FRONT_SCAN, LaserScan, self.callback_front_scan)
        self.sub_odometry = rospy.Subscriber(self.TOPIC_ODOM, Odometry, self.callback_odometry)
        self.sub_global_plan = rospy.Subscriber(self.TOPIC_GLOBAL_PLAN, Path, self.callback_global_plan)
        self.sub_local_plan = rospy.Subscriber(self.TOPIC_MPC, Path, self.callback_local_plan)
        self.sub_footprint = rospy.Subscriber(self.TOPIC_LOCAL_FOOTPRINT, PolygonStamped, self.callback_footprint)
        self.sub_cmd_vel = rospy.Subscriber(self.TOPIC_CMD_VEL, Twist, self.callback_cmd_vel)
        self.publish_cmd_vel = rospy.Publisher(self.TOPIC_CMD_VEL, Twist, queue_size=10)
        self.result_pub = rospy.Subscriber(self.RESULT_DATA, ResultData, self.callback_result_data)
        
        # init CSV File
        print("Write to CSV file: data.csv")
        file_path = '/jackal_ws/src/mlda-barn-2024/data.csv'
        if not os.path.exists(file_path):
            new_file = True
        else:
            new_file = False
        self.csv_file = open(file_path, 'a')
        self.writer = csv.writer(self.csv_file)

        self.metadata_rows = ["success", "actual_time", "optimal_time"]
        self.lidar_rows = ["lidar_" + str(i) for i in range(720)]
        self.odometry_rows = ['pos_x', 'pos_y', 'pose_heading', 'twist_linear', 'twist_angular']
        self.action_rows = ['cmd_vel_linear', 'cmd_vel_angular']
        self.data_rows = self.lidar_rows + self.odometry_rows + self.action_rows
        self.all_rows = self.metadata_rows + self.data_rows

        if new_file:
            self.writer.writerow(self.all_rows)
            self.csv_file.flush()
        
        self.data = []
        self.data_dict = {}

    def update_row(self):
        if len(self.data_dict) == len(self.data_rows):
            self.data.append(self.data_dict)
            self.data_dict = {}

    def callback_result_data(self, data):
        for row in self.data:
            row["world_idx"] = data.world_idx
            row["success"] = data.success
            row["actual_time"] = data.actual_time
            row["optimal_time"] = data.optimal_time

        if data.success:
            self.writer.writerows(self.data)
            self.csv_file.flush()
            self.data = []

    def callback_front_scan(self, data):
        self.scan = data
        if 1:
            # print("Scan points: ", len(data.ranges), "From Max: ", data.range_max, "| Min: ", round(data.range_min,2))
            # print("Angle from: ", np.degrees(data.angle_min).round(2), " to: ", np.degrees(data.angle_max).round(2), " increment: ", np.degrees(data.angle_increment).round(3))
           
            # update the data_dict
            assert(len(data.ranges) == 720)
            for i in range(720):
                if data.ranges[i] > data.range_max:
                    self.data_dict["lidar_" + str(i)] = data.range_max
                else:
                    self.data_dict["lidar_" + str(i)] = data.ranges[i]
            self.update_row()
    
    def callback_odometry(self, data):
        if 1:
            self.odometry = data
            # print("==========================")
            # print("----------------------- pose.position")
            # print(data.pose.pose.position)
            # print("----------------------- pose.orientation")
            # print(data.pose.pose.orientation)
            q = Quaternion()
            q.x = data.pose.pose.orientation.x
            q.y = data.pose.pose.orientation.y
            q.z = data.pose.pose.orientation.z
            q.w = data.pose.pose.orientation.w
            # print("----------------------- pose.heading")
            heading_rad = np.array(euler_from_quaternion([q.x, q.y, q.z,q.w])[2])
            heading_deg = np.degrees(heading_rad)
            # print("Rad: " + str(heading_rad.round(3)))
            # print("Degree: " + str( heading_deg.round(3)))

            # print("----------------------- twist.linear")
            # print(data.twist.twist.linear)
            # print("----------------------- twist.angular")
            # print(data.twist.twist.angular)

            # update the data_dict
            self.data_dict["pos_x"] = data.pose.pose.position.x
            self.data_dict["pos_y"] = data.pose.pose.position.y
            self.data_dict["pose_heading"] = heading_rad
            self.data_dict["twist_linear"] = data.twist.twist.linear.x
            self.data_dict["twist_angular"] = data.twist.twist.angular.z
            self.update_row()
    
    def callback_cmd_vel(self, data):
        if 1:
            print("Linear: ", round(data.linear.x,3), "; Angular: ", round(data.angular.z,3))
            # linear.y and angular.x are always 0
            
            # update the data_dict
            self.data_dict["cmd_vel_linear"] = data.linear.x
            self.data_dict["cmd_vel_angular"] = data.angular.z
            self.update_row()

    def callback_footprint(self, data):
        self.footprint = data
        if 0: 
            points_array = []
            for point in data.polygon.points:
                points_array.append([point.x, point.y,point.z])
            np_array = np.array(points_array)
            print("Number of points on the Polygon: ", len(data.polygon.points))
            print("Points: ", np.round(np_array,3))

    def callback_global_plan(self, data):
        self.global_plan = data
        if 0: 
            print("Global Path points ", len(data.poses))
            print(data.poses[3])
            print("Local Path points ", len(self.local_plan.poses))
    
    def callback_local_plan(self, data):
        self.local_plan = data
        if 0:
            print("Local Path points ", len(data.poses))
            print(data.poses[3])
            print("Global Path points ", len(self.global_plan.poses))



if __name__ == "__main__":
    rospy.init_node('inspection_node')
    rospy.loginfo("Inspection Node Started")
    inspect = Inspection()
    rospy.spin()
    