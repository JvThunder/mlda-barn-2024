from sklearn.preprocessing import MinMaxScaler
import pickle 

# get scaler values from scaler.pkl
with open('diffusion_policy/model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print(scaler)

# torch Dataset
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class KULBarnDataset(Dataset):
    def get_local_goal(self):
        x = self.data['pos_x']
        y = self.data['pos_y']
        theta = self.data['pose_heading']
        goal_x = self.data['goal_x']
        goal_y = self.data['goal_y']
        self.data['local_x'] = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        self.data['local_y'] = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
    
    def __init__(self, df, scaler=None):
        super().__init__()

        self.data = df
        self.get_local_goal()   
        self.data = self.data.drop(columns=[
            'world_idx', 'timestep', 'actual_time', 'optimal_time', 
            'pos_x', 'pos_y', 'pose_heading', 'goal_x', 'goal_y', 'success'
        ])

        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.data)
        else:
            self.scaler = scaler
        self.data = pd.DataFrame(self.scaler.transform(self.data), columns=self.data.columns)
        
        # get all the column values that contain the word lidar
        self.lidar_cols = [col for col in self.data.columns if 'lidar' in col]
        # get actions columns
        self.actions_cols = [col for col in self.data.columns if 'cmd' in col]
        # get other columns
        self.non_lidar_cols = [col for col in self.data.columns if col not in self.lidar_cols and col not in self.actions_cols]
        print("Lidar Columns:", self.lidar_cols)
        print("Non Lidar Columns:", self.non_lidar_cols)
        print("Action Columns:", self.actions_cols)

        self.lidar_data = self.data[self.lidar_cols].values
        self.non_lidar_data = self.data[self.non_lidar_cols].values
        self.actions_data = self.data[self.actions_cols].values

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        lidar = self.lidar_data[idx]
        non_lidar = self.non_lidar_data[idx]
        actions = self.actions_data[idx]
        return lidar, non_lidar, actions

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
df = pd.read_csv('diffusion_policy/data_sorted.csv')
train_dataset = KULBarnDataset(df, scaler=scaler)
for batch in train_dataset:
    print(batch)
    break

import torch
from diffusion_policy.cnn_model import CNNModel

# load CNN model from model/cnn_model.pth
lidar_cols = 720
non_lidar_cols = 4
no_actions = 2
model = CNNModel(lidar_cols, non_lidar_cols, no_actions)
model.load_state_dict(torch.load('model/cnn_model.pth'))
print(model)