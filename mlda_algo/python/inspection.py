#!/usr/bin/python
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, PolygonStamped
from nav_msgs.msg import Path, Odometry
from jackal_helper.msg import ResultData

import numpy as np
from tf_conversions import Quaternion
import csv
import os
import copy
import scipy
import math
from scipy.signal import savgol_filter

class Inspection():
    def __init__(self):
        # Topic Defintions
        self.TOPIC_FRONT_SCAN = "/front/scan" # Front Laser Scan (LIDAR)
        self.TOPIC_ODOM = "/odometry/filtered" # Odometry (pose and twist)
        self.TOPIC_CMD_VEL = "/cmd_vel" # Command Velocity (action)

        self.TOPIC_LOCAL_PLAN = "/move_base/TrajectoryPlannerROS/local_plan" # Local plan
        self.TOPIC_GLOBAL_PLAN = "/move_base/TrajectoryPlannerROS/global_plan" # Global plan
        self.RESULT_DATA = "/result_data" # Result Data
    
        # Object to store
        self.scan = LaserScan()
        self.cmd_vel = Twist()
        self.global_plan = Path()
        self.odometry = Odometry()
        
        # init CSV File
        print("Write to CSV file")
        file_path = "/jackal_ws/src/mlda-barn-2024/test_data.csv"
        self.metadata_rows = ["success", "actual_time", "optimal_time", "world_idx", "timestep", "goal_x", "goal_y"]
        self.lidar_rows = ["lidar_" + str(i) for i in range(360)]
        self.odometry_rows = ['pos_x', 'pos_y', 'pose_heading', 'twist_linear', 'twist_angular']
        self.action_rows = ['cmd_vel_linear', 'cmd_vel_angular']
        self.goal_rows = ['local_goal_x', 'local_goal_y']
        self.data_rows = self.lidar_rows + self.odometry_rows + self.action_rows + self.goal_rows
        self.fieldnames = self.metadata_rows + self.data_rows

        # Subscribe        
        self.sub_front_scan = rospy.Subscriber(self.TOPIC_FRONT_SCAN, LaserScan, self.callback_front_scan)
        self.sub_odometry = rospy.Subscriber(self.TOPIC_ODOM, Odometry, self.callback_odometry)
        self.sub_global_plan = rospy.Subscriber(self.TOPIC_GLOBAL_PLAN, Path, self.callback_global_plan)
        self.sub_cmd_vel = rospy.Subscriber(self.TOPIC_CMD_VEL, Twist, self.callback_cmd_vel)
        self.sub_result = rospy.Subscriber(self.RESULT_DATA, ResultData, self.callback_result_data)
        
        file_exist = False
        if os.path.exists(file_path):
            file_exist = True
        
        self.csv_file = open(file_path, 'a')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        if not file_exist:
            self.writer.writeheader()

        self.stop = False
        rospy.set_param('/inspection_done', False)

        self.data = []
        self.data_dict = {}
        self.look_ahead = 1.0

    def transform_path(self, path):
        pos_x = self.data_dict["pos_x"]
        pos_y = self.data_dict["pos_y"]
        heading = self.data_dict["pose_heading"]
        for i, points in enumerate(path):
            new_x = np.cos(heading) * (points[0] - pos_x) + np.sin(heading) * (points[1] - pos_y)
            new_y = -np.sin(heading) * (points[0] - pos_x) + np.cos(heading) * (points[1] - pos_y)
            path[i] = [new_x, new_y]
        return path

    def compute_local_goal(self):
        global_plan = self.global_plan
        x = [e.pose.position.x for e in global_plan.poses]
        y = [e.pose.position.y for e in global_plan.poses]

        sm_x = savgol_filter(x, 19, 3)
        sm_y = savgol_filter(y, 19, 3)
        gp = np.stack([sm_x, sm_y], axis=-1)
        gp = self.transform_path(gp)

        if len(gp) > 0:
            for wp in gp:
                if np.linalg.norm(wp) > self.look_ahead:
                    break
            return wp[0], wp[1]
        else:
            return 0, 0

    def update_row(self):
        # check if global_plan has attribute poses
        if len(self.data_dict) >= len(self.data_rows) - len(self.goal_rows) and len(self.global_plan.poses) > 0 and not self.stop:            
            # check if data_dict is NaN
            data_dict = copy.deepcopy(self.data_dict)
            for key in data_dict.keys():
                if np.isnan(data_dict[key]):
                    return
                
            local_goal_x, local_goal_y = self.compute_local_goal()
            data_dict["local_goal_x"] = local_goal_x
            data_dict["local_goal_y"] = local_goal_y
            self.data.append(data_dict)
            
    def callback_result_data(self, metadata):
        print("---- Processing Result Data ----")
        print("Length of data: ", len(self.data))
        print("Metadata: ", metadata)
        self.stop = True

        for i in range(len(self.data)):
            self.data[i]["world_idx"] = metadata.world_idx
            self.data[i]["success"] = metadata.success
            self.data[i]["actual_time"] = metadata.actual_time
            self.data[i]["optimal_time"] = metadata.optimal_time
            self.data[i]["goal_x"] = metadata.goal_x
            self.data[i]["goal_y"] = metadata.goal_y
            self.data[i]["timestep"] = i
            self.writer.writerow(self.data[i])
        print("---- Writing to CSV Done ----")
        rospy.set_param('/inspection_done', True)  # Add this line to set a completion flag


    def callback_front_scan(self, data):
        assert(len(data.ranges) == 720)
        cnt = 0
        for i in range(0, 720, 2):
            self.data_dict["lidar_" + str(cnt)] = min(5, data.ranges[i])
            cnt += 1

    def euler_from_quaternion(self, x, y, z, w):
        # Roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        # Pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        # Yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw
    
    def callback_odometry(self, data):
        if 1:
            self.odometry = data
            q = Quaternion()
            q.x = data.pose.pose.orientation.x
            q.y = data.pose.pose.orientation.y
            q.z = data.pose.pose.orientation.z
            q.w = data.pose.pose.orientation.w
            heading_rad = self.euler_from_quaternion(q.x, q.y, q.z, q.w)[2]

            self.data_dict["pos_x"] = data.pose.pose.position.x
            self.data_dict["pos_y"] = data.pose.pose.position.y
            self.data_dict["pose_heading"] = heading_rad
            self.data_dict["twist_linear"] = data.twist.twist.linear.x
            self.data_dict["twist_angular"] = data.twist.twist.angular.z
    
    def callback_cmd_vel(self, data):
        if not self.stop:
            print("Linear: ", round(data.linear.x,3), "; Angular: ", round(data.angular.z,3))            
            # update the data_dict
            self.data_dict["cmd_vel_linear"] = data.linear.x
            self.data_dict["cmd_vel_angular"] = data.angular.z
            self.update_row()

    def callback_global_plan(self, data):
        self.global_plan = data


if __name__ == "__main__":
    rospy.init_node('inspection_node')
    rospy.loginfo("Inspection Node Started")
    inspect = Inspection()
    rospy.spin()
    