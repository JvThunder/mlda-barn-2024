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

from sklearn.preprocessing import MinMaxScaler
import torch
import pickle
import pandas as pd
import json
import threading
import time
from collections import namedtuple
import easydict

INF = 10000

class ROSNode:
    def __init__(self):
        self.TOPIC_VEL = "/cmd_vel"
        self.TOPIC_GLOBAL_PLAN = "/move_base/TrajectoryPlannerROS/global_plan"
        # self.TOPIC_LOCAL_PLAN = "/move_base/TrajectoryPlannerROS/local_plan"
        self.TOPIC_FRONT_SCAN = "/front/scan" # Front Laser Scan (LIDAR)
        self.TOPIC_ODOM = "/odometry/filtered"
        # self.TOPIC_MPC_PLAN = "/mpc_plan"
        # self.TOPIC_CLOUD = "/front/odom/cloud"
        # self.TOPIC_MAP_CLOUD = "/map/cloud"
        # self.TOPIC_MARKER = "/mode"

        self.scan = LaserScan()
        self.projector = lg.LaserProjection()
        self.cmd_vel = Twist()
        self.odometry = Odometry()
        self.global_plan = Path()
        self.local_plan = Path()
        self.N = 10

        self.L = 0.37558
        self.v_opt = 0
        self.w_opt = 0

        self.x_ref = []
        self.y_ref = []
        self.obs_x = []
        self.obs_y = []
        self.og_x_ref = []
        self.og_y_ref = []
        self.theta_ref = []
        self.count = 0
        self.mode = "safe"
        self.display = ""

        self.rate = rospy.Rate(50)
        self.multiplier = 0.3
        self.look_ahead = 0.325
        self.data_dict = {}
        self.lidar_data = []

        self.lidar_cols = ["lidar_" + str(i) for i in range(0, 720, 1)]
        self.non_lidar_cols = ['local_x', 'local_y']
        self.action_cols = ['cmd_vel_linear', 'cmd_vel_angular']

        lidar_cols = len(self.lidar_cols)
        non_lidar_cols = len(self.non_lidar_cols)
        no_actions = len(self.action_cols)

        # self.model = CNNModel(lidar_cols, non_lidar_cols, no_actions)
        # self.model.load_state_dict(torch.load('/jackal_ws/src/mlda-barn-2024/train_imitation/model/cnn_model.pth', map_location=torch.device('cpu')))
        
        config_dict = easydict.EasyDict({
            "input_dim": 32,
            "num_patch": 36,
            "model_dim": 32,
            "ffn_dim": 256,
            "attention_heads": 4,
            "attention_dropout": 0.0,
            "dropout": 0.2,
            "encoder_layers": 3,
            "decoder_layers": 3,
            "device": 'cpu',
        })

        self.model = Transformer(config_dict)
        self.model.load_state_dict(torch.load('/jackal_ws/src/mlda-barn-2024/train_imitation/model/transformer_model.pth', map_location=torch.device('cpu')))
        self.model.eval()

        self.v = 0
        self.w = 0

        with open('/jackal_ws/src/mlda-barn-2024/train_imitation/model/scaler_params.json', 'r') as f:
            self.scaler_params = json.load(f)
        # print(self.scaler_params)

        self.sub_front_scan = rospy.Subscriber(self.TOPIC_FRONT_SCAN, LaserScan, self.callback_front_scan)
        self.pub_vel = rospy.Publisher(self.TOPIC_VEL, Twist, queue_size=10, latch=True)
        # self.pub_mpc = rospy.Publisher(self.TOPIC_MPC_PLAN, Path, queue_size=1)
        # self.pub_marker = rospy.Publisher(self.TOPIC_MARKER, Marker, queue_size=1)

        self.sub_odometry = rospy.Subscriber(
            self.TOPIC_ODOM, Odometry, self.callback_odom
        )
        self.sub_global_plan = rospy.Subscriber(
            self.TOPIC_GLOBAL_PLAN, Path, self.callback_global_plan
        )
        # self.sub_local_plan = rospy.Subscriber(
        #     self.TOPIC_LOCAL_PLAN, Path, self.callback_local_plan
        # )
        # self.sub_cloud = rospy.Subscriber(
        #     self.TOPIC_CLOUD, PointCloud2, self.callback_cloud
        # )
        # self.sub_map_cloud = rospy.Subscriber(
        #     self.TOPIC_MAP_CLOUD, PointCloud2, self.callback_map_cloud
        # )

         # Create a lock for updating the velocity
        # self.lock = threading.Lock()

        # Start the computation thread
        # self.thread = threading.Thread(target=self.compute_velocity)
        # self.thread.start()
    
    def callback_front_scan(self, data):
        self.lidar_data = data.ranges

    def callback_cloud(self, data):
        point_generator = pc2.read_points(data)
        self.obs_x = []
        self.obs_y = []
        for point in point_generator:
            self.obs_x.append(point[0])
            self.obs_y.append(point[1])

    def callback_map_cloud(self, data):
        point_generator = pc2.read_points(data)
        self.map_x = []
        self.map_y = []
        for point in point_generator:
            self.map_x.append(point[0])
            self.map_y.append(point[1])

    def get_local_goal(self, x, y, goal_x, goal_y, theta):
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        distance = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)

        local_x /= distance
        local_y /= distance

        # print(f"x:{x}, y:{y}, goal_x:{goal_x}, goal_y:{goal_y}, theta:{theta}")
        # print(f"local_x:{local_x}, local_y:{local_y}")

        return local_x, local_y, distance
    
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

    def callback_odom(self, data):
        x = data.pose.pose.orientation.x
        y = data.pose.pose.orientation.y
        z = data.pose.pose.orientation.z
        w = data.pose.pose.orientation.w
        v = data.twist.twist.linear.x
        heading_rad = self.euler_from_quaternion(x, y, z, w)[2]

        # text = (
        #     "V: " + str(round(v, 3)) + " W: " + str(round(w, 3)) + "\n" + self.display
        # )

        pos_x = data.pose.pose.position.x
        pos_y = data.pose.pose.position.y

        # print("====================================")
        # print(f"x:{round(pos_x, 3)}, y:{round(pos_y, 3)}")
        # print(f"v:{round(self.v,3)}, w:{round(self.w,3)}")

        # self.pub_marker.publish(marker)

        goal_x, goal_y = self.compute_local_goal(pos_x, pos_y)

        print("Pos x: ", round(pos_x, 3), "; Pos y: ", round(pos_y, 3))
        print("Goal x: ", round(goal_x, 3), "; Goal y: ", round(goal_y, 3))
        local_x, local_y, distance = self.get_local_goal(x=pos_x, y=pos_y, goal_x=goal_x, goal_y=goal_y, theta=heading_rad)
        # print(f"local_x:{round(local_x, 3)}, local_y:{round(local_y, 3)}")
        self.data_dict["local_x"] = local_x
        self.data_dict["local_y"] = local_y
        # self.data_dict["distance"] = distance
        # self.data_dict["twist_linear"] = data.twist.twist.linear.x
        # self.data_dict["twist_angular"] = data.twist.twist.angular.z
        # print("====================================")
        # print("Local x: ", round(local_x,3), "; Local y: ", round(local_y,3))
        # print("Twist Linear: ", round(data.twist.twist.linear.x,3), "; Twist Angular: ", round(data.twist.twist.angular.z,3))

    def callback_global_plan(self, data):
        self.global_plan = data

    # def callback_local_plan(self, data):
    #     self.local_plan = data

    def heading_preprocess_radian(self, center, target):
        min = center - np.pi
        max = center + np.pi
        if target < min:
            while target < min:
                # print('Processed')
                target += 2 * np.pi
        elif target > max:
            while target > max:
                # print('Processed')
                target -= 2 * np.pi
        return target

    def quaternion_to_yaw(self, orientation):
        # Convert quaternion orientation data to yaw angle of robot
        q0 = orientation.x
        q1 = orientation.y
        q2 = orientation.z
        q3 = orientation.w
        theta = math.atan2(2.0 * (q2 * q3 + q0 * q1), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
        return theta

    # def publish_trajectory(self, mpc_x_traj, mpc_y_traj):
    #     mpc_traj_msg = Path()
    #     mpc_traj_msg.header.stamp = rospy.Time.now()
    #     mpc_traj_msg.header.frame_id = "odom"
    #     for i in range(mpc_x_traj.shape[0]):
    #         pose = PoseStamped()
    #         pose.pose.position.x = mpc_x_traj[i]
    #         pose.pose.position.y = mpc_y_traj[i]
    #         pose.pose.orientation = Quaternion(0, 0, 0, 1)
    #         mpc_traj_msg.poses.append(pose)

    #     self.pub_mpc.publish(mpc_traj_msg)

    def publish_velocity(self, v_opt, w_opt):
        vel = Twist()
        vel.linear.x = np.clip(v_opt * self.multiplier, -0.3, 0.8)
        vel.angular.z = np.clip(w_opt, -1.5, 1.5)
        self.pub_vel.publish(vel)
        self.rate.sleep()

    def compute_local_goal(self, pos_x, pos_y):
        # check which is shortest distance to pos
        min_dist = INF
        local_goal_x = 0
        local_goal_y = 0
        global_plan = self.global_plan

        if len(global_plan.poses) == 0:
            print("------------- No global plan ----------------")
            self.multiplier = 0.3
        else:
            self.multiplier = 0.9

        for i in range(len(self.global_plan.poses)):
            global_x = global_plan.poses[i].pose.position.x
            global_y = global_plan.poses[i].pose.position.y
            dist = np.sqrt((pos_x - global_x)**2 + (pos_y - global_y)**2)
            if dist < min_dist and dist > self.look_ahead:
                min_dist = dist
                local_goal_x = global_x
                local_goal_y = global_y
        return local_goal_x, local_goal_y

    
    def compute_velocity(self):
        # print("Lidar Data: ", len(self.lidar_data))
        # print("Data Dict: ", len(self.data_dict.keys()))
        if len(self.data_dict.keys()) == len(self.non_lidar_cols) and len(self.lidar_data) == 720:
            start = time.time()

            lidar_data = np.clip(self.lidar_data, 0, 5)
            tensor_lidar = torch.tensor(lidar_data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)            
            tensor_non_lidar = torch.tensor([self.data_dict["local_x"], self.data_dict["local_y"]], 
                                            dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

            actions, _, _ = self.model(tensor_lidar, tensor_non_lidar)
            v, w = actions[0][0].item(), actions[0][1].item()
            self.v, self.w = v, w
            
            end = time.time()
            print("Time taken: {}".format(end - start))
            

    def run(self):
        self.compute_velocity()
        self.publish_velocity(self.v, self.w)

if __name__ == "__main__":
    rospy.init_node("imit_node")
    rospy.loginfo("Behaviour Cloning Node running")
    node = ROSNode()
    while not rospy.is_shutdown():
        node.run()
