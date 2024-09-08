#!/usr/bin/python3
import rospy
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped, PolygonStamped, Quaternion
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
import laser_geometry.laser_geometry as lg
import math
import time


class ROSNode:
    def __init__(self):
        self.TOPIC_VEL = "/cmd_vel"
        self.TOPIC_GLOBAL_PLAN = "/move_base/TrajectoryPlannerROS/global_plan"
        self.TOPIC_LOCAL_PLAN = "/move_base/TrajectoryPlannerROS/local_plan"
        self.TOPIC_ODOM = "/odometry/filtered"
        self.TOPIC_MPC_PLAN = "/mpc_plan"
        self.TOPIC_CLOUD = "/front/odom/cloud"
        self.TOPIC_MAP_CLOUD = "/map/cloud"
        self.TOPIC_MARKER = "/mode"

        self.pub_vel = rospy.Publisher(self.TOPIC_VEL, Twist, queue_size=10, latch=True)
        self.pub_mpc = rospy.Publisher(self.TOPIC_MPC_PLAN, Path, queue_size=1)
        self.pub_marker = rospy.Publisher(self.TOPIC_MARKER, Marker, queue_size=1)

        self.sub_odometry = rospy.Subscriber(
            self.TOPIC_ODOM, Odometry, self.callback_odom
        )
        self.sub_global_plan = rospy.Subscriber(
            self.TOPIC_GLOBAL_PLAN, Path, self.callback_global_plan
        )
        self.sub_local_plan = rospy.Subscriber(
            self.TOPIC_LOCAL_PLAN, Path, self.callback_local_plan
        )
        self.sub_cloud = rospy.Subscriber(
            self.TOPIC_CLOUD, PointCloud2, self.callback_cloud
        )
        self.sub_map_cloud = rospy.Subscriber(
            self.TOPIC_MAP_CLOUD, PointCloud2, self.callback_map_cloud
        )

        self.projector = lg.LaserProjection()
        self.cmd_vel = Twist()
        self.odometry = Odometry()
        self.global_plan = Path()
        self.local_plan = Path()
        self.rate = 10
        self.N = 10

        self.mpc = mpc_algo.NMPC(freq=self.rate, N=self.N)
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

    def callback_odom(self, data):
        self.odometry = data
        yaw = self.quaternion_to_yaw(data.pose.pose.orientation)
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        v = data.twist.twist.linear.x
        w = data.twist.twist.angular.z
        vr = v + w * self.mpc.L / 2
        vl = v - w * self.mpc.L / 2
        self.X0 = [x, y, yaw, vr, vl]
        marker = Marker()
        marker.type = 2
        marker.color.a = 1
        if self.mode == "safe":
            marker.color.r = 0
            marker.color.g = 1
            marker.color.b = 0
        elif self.mode == "obs":
            marker.color.r = 1
            marker.color.g = 1
            marker.color.b = 0
        else:
            marker.color.r = 1
            marker.color.g = 0
            marker.color.b = 0
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.pose.position.x = x + 0.5
        marker.pose.position.y = y
        marker.pose.position.z = 0
        # self.pub_marker.publish(marker)
        marker.type = 9
        marker.text = (
            "V: " + str(round(v, 3)) + " W: " + str(round(w, 3)) + "\n" + self.display
        )

        self.pub_marker.publish(marker)

    def callback_global_plan(self, data):
        self.global_plan = data
        if 1:
            if len(data.poses) <= 2 * (self.mpc.N + 5):
                self.og_x_ref = [
                    pose.pose.position.x for pose in self.global_plan.poses[::1]
                ]
                self.og_y_ref = [
                    pose.pose.position.y for pose in self.global_plan.poses[::1]
                ]
            else:
                self.og_x_ref = [
                    pose.pose.position.x for pose in self.global_plan.poses[::2]
                ]
                self.og_y_ref = [
                    pose.pose.position.y for pose in self.global_plan.poses[::2]
                ]
            self.theta_ref = []
            center_heading = self.X0[2]
            for i in range(len(self.og_x_ref) - 1):
                theta = math.atan2(
                    (self.og_y_ref[i + 1] - self.og_y_ref[i]),
                    (self.og_x_ref[i + 1] - self.og_x_ref[i]),
                )
                theta_preprocessed = self.heading_preprocess_radian(
                    center_heading, theta
                )
                self.theta_ref.append(theta_preprocessed)
                if i == 0:
                    self.theta_ref.append(theta_preprocessed)
                center_heading = theta_preprocessed

    def callback_local_plan(self, data):
        self.local_plan = data
        # print("Local")

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

    def publish_trajectory(self, mpc_x_traj, mpc_y_traj):
        mpc_traj_msg = Path()
        mpc_traj_msg.header.stamp = rospy.Time.now()
        mpc_traj_msg.header.frame_id = "odom"
        for i in range(mpc_x_traj.shape[0]):
            pose = PoseStamped()
            pose.pose.position.x = mpc_x_traj[i]
            pose.pose.position.y = mpc_y_traj[i]
            pose.pose.orientation = Quaternion(0, 0, 0, 1)
            mpc_traj_msg.poses.append(pose)

        self.pub_mpc.publish(mpc_traj_msg)

    def publish_velocity(self, v_opt, w_opt):
        vel = Twist()
        vel.linear.x = v_opt
        vel.angular.z = w_opt
        self.pub_vel.publish(vel)

    def run(self):
        try:
            self.publish_velocity(1, 1)
        except Exception as e:
            rospy.logerr(e)


if __name__ == "__main__":
    rospy.init_node("nmpc")
    rospy.loginfo("CNN Node running")
    node = ROSNode()
    pause = rospy.Rate(10)  # Match the calculation ? Else idle?
    time.sleep(1)
    while not rospy.is_shutdown():
        node.run()
        pause.sleep()
