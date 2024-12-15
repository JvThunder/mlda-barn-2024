#!/usr/bin/python3
import rospy
import numpy as np
import math
import time

from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped, PolygonStamped, Quaternion
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
import laser_geometry.laser_geometry as lg

from cnn_model import CNNModel
from transformer_model import Transformer
from diffusion_policy_model import DiffusionModel

from sklearn.preprocessing import MinMaxScaler
import torch
import pickle
import pandas as pd
import json
import threading
import time
from collections import namedtuple
import easydict
import random

from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

# model_arch = "transformer" 
model_arch = "cnn"
# model_arch = "diffusion"

class ROSNode:
    def __init__(self):
        self.TOPIC_VEL = "/cmd_vel"
        self.TOPIC_GLOBAL_PLAN = "/move_base/TrajectoryPlannerROS/global_plan"
        self.TOPIC_FRONT_SCAN = "/front/scan"
        self.TOPIC_ODOM = "/odometry/filtered"

        self.scan = LaserScan()
        self.cmd_vel = Twist()
        self.odometry = Odometry()
        self.global_plan = Path()

        self.x_ref = []
        self.y_ref = []
        self.obs_x = []
        self.obs_y = []
        self.og_x_ref = []
        self.og_y_ref = []
        self.theta_ref = []
        self.lidar_data = []
        self.device = torch.device('cpu')
        self.look_ahead = 1.0
        base_path = "/jackal_ws/src/mlda-barn-2024/train_imitation/models"

        if model_arch == "transformer":
            config_dict = easydict.EasyDict({
                "input_dim": 32,
                "num_patch": 36,
                "model_dim": 32,
                "ffn_dim": 256,
                "attention_heads": 4,
                "attention_dropout": 0.0,
                "dropout": 0.5,
                "encoder_layers": 2,
                "decoder_layers": 2,
                "lidar_dim": 360,
                "non_lidar_dim": 4,
                "device": "cpu",
            })
            self.model = Transformer(config_dict)
            filepath = base_path + "/transformer_model.pth"
        elif model_arch == "cnn":
            self.model = CNNModel(360, 4, 2)
            filepath = base_path + "/cnn_model.pth"
        elif model_arch == "diffusion":
            self.model = DiffusionModel()
            filepath = base_path + "/diffusion_model.pth"

        self.model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
        self.model = self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.pos_x = 0
        self.pos_y = 0
        self.local_goal_x = 0
        self.local_goal_y = 0
        self.goal_x = 0
        self.goal_y = 0
        self.pose_heading = 0
        self.v = 0
        self.w = 0
        self.start_time = 0

        # with open('/jackal_ws/src/mlda-barn-2024/train_imitation/model/scaler_params.json', 'r') as f:
        #     self.scaler_params = json.load(f)

        self.sub_front_scan = rospy.Subscriber(self.TOPIC_FRONT_SCAN, LaserScan, self.callback_front_scan)
        self.pub_vel = rospy.Publisher(self.TOPIC_VEL, Twist, queue_size=10, latch=True)

        self.sub_odometry = rospy.Subscriber(
            self.TOPIC_ODOM, Odometry, self.callback_odometry
        )
        self.sub_global_plan = rospy.Subscriber(
            self.TOPIC_GLOBAL_PLAN, Path, self.callback_global_plan
        )   

        if model_arch == "transformer":
            self.tensor_lidar = torch.rand(1, 360, 1)
            self.tensor_non_lidar = torch.rand(1, 4, 1)
        elif model_arch == "cnn":
            self.tensor_lidar = torch.rand(1, 1, 360)
            self.tensor_non_lidar = torch.rand(1, 1, 4)
        elif model_arch == "diffusion":
            self.tensor_lidar = torch.rand(1, 4, 360)
            self.tensor_non_lidar = torch.rand(1, 4, 4)
    
    def callback_front_scan(self, data):
        with torch.no_grad():
            self.lidar_data = list(np.clip(data.ranges, 0, 5))[::2]
            self.non_lidar_data = [self.local_goal_x, self.local_goal_y, self.goal_x, self.goal_y]

            if model_arch == "transformer":
                self.tensor_lidar = torch.tensor(self.lidar_data, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(self.device)            
                self.tensor_non_lidar = torch.tensor(self.non_lidar_data, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(self.device)
            elif model_arch == "cnn":
                self.tensor_lidar = torch.tensor(self.lidar_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)            
                self.tensor_non_lidar = torch.tensor(self.non_lidar_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            elif model_arch == "diffusion":
                curr_tensor_lidar = torch.tensor(self.lidar_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)            
                curr_tensor_non_lidar = torch.tensor(self.non_lidar_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                # stack the lidar data with the previous lidar data
                self.tensor_lidar = torch.cat((self.tensor_lidar, curr_tensor_lidar), dim=1)
                self.tensor_non_lidar = torch.cat((self.tensor_non_lidar, curr_tensor_non_lidar), dim=1)
                # keep only the last 4 frames
                self.tensor_lidar = self.tensor_lidar[:, -4:, :]
                self.tensor_non_lidar = self.tensor_non_lidar[:, -4:, :]
            
            self.start_time = time.time()
            if model_arch == "transformer":
                actions, _, _ = self.model(self.tensor_lidar, self.tensor_non_lidar)
            elif model_arch == "cnn":
                actions = self.model(self.tensor_lidar, self.tensor_non_lidar)
            elif model_arch == "diffusion":
                actions = self.model(self.tensor_lidar, self.tensor_non_lidar)
            print("Time taken for inference: {0}".format(time.time() - self.start_time))
            
            self.v, self.w = actions[0][0].item(), actions[0][1].item()

            vel = Twist()
            vel.linear.x = self.v
            vel.angular.z = self.w
            self.pub_vel.publish(vel)

    def get_normalized_goal(self, x, y, goal_x, goal_y, theta):
        local_x = np.cos(theta) * (goal_x - x) + np.sin(theta) * (goal_y - y)
        local_y = -np.sin(theta) * (goal_x - x) + np.cos(theta) * (goal_y - y)
        return local_x, local_y

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
        x = data.pose.pose.orientation.x
        y = data.pose.pose.orientation.y
        z = data.pose.pose.orientation.z
        w = data.pose.pose.orientation.w
        heading_rad = self.euler_from_quaternion(x, y, z, w)[2]

        pos_x = data.pose.pose.position.x
        pos_y = data.pose.pose.position.y
        goal_x, goal_y = (0, 10)
        goal_x, goal_y = self.get_normalized_goal(x=pos_x, y=pos_y, goal_x=goal_x, goal_y=goal_y, theta=heading_rad)

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.pose_heading = heading_rad

    def transform_path(self, path):
        pos_x = self.pos_x
        pos_y = self.pos_y
        heading = self.pose_heading
        for i, points in enumerate(path):
            new_x = np.cos(heading) * (points[0] - pos_x) + np.sin(heading) * (points[1] - pos_y)
            new_y = -np.sin(heading) * (points[0] - pos_x) + np.cos(heading) * (points[1] - pos_y)
            path[i] = [new_x, new_y]
        return path

    def compute_local_goal(self):
        global_plan = self.global_plan
        if len(global_plan.poses) == 0:
            return 0, 0
        
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
    
    def callback_global_plan(self, data):
        self.global_plan = data
        self.local_goal_x, self.local_goal_y = self.compute_local_goal()

    def run(self):
        output = self.model(self.tensor_lidar, self.tensor_non_lidar)

if __name__ == "__main__":
    rospy.init_node("imit_node")
    rospy.loginfo("Behaviour Cloning Node running")
    node = ROSNode()
    while not rospy.is_shutdown():
        node.run()
