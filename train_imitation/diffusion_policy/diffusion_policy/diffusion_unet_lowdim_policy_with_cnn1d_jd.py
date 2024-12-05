import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
from typing import Union, Dict

from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.common.dict_of_tensor_mixin import DictOfTensorMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer

class CNNModel(nn.Module):
    def __init__(self, num_lidar_features, num_non_lidar_features, output_dim=32, nframes=1):
        super(CNNModel, self).__init__()
        self.output_dim = output_dim
        self.act_fea_cv1 = nn.Conv1d(
            in_channels=nframes, out_channels=32, kernel_size=5, stride=2, padding=2, padding_mode='circular'
        )
        self.act_fea_cv2 = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, padding_mode='circular'
        )

        # conv_output_size = (num_lidar_features - 5 + 2*6) // 2 + 1  # Output size after self.act_fea_cv1
        # conv_output_size = (conv_output_size - 3 + 2*1) // 2 + 1  # Output size after self.act_fea_cv2
        # conv_output_size *= 32  # Multiply by the number of output channels
        with torch.no_grad():
            sample_input = torch.randn(1, nframes, num_lidar_features)
            sample_output = self.act_fea_cv1(sample_input)
            sample_output = self.act_fea_cv2(sample_output)
            conv_output_size = sample_output.view(1, -1).shape[1]

        # Calculate the output size of the CNN
        self.fc1 = nn.Linear(conv_output_size + num_non_lidar_features*nframes, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.norm = nn.LayerNorm(output_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, lidar, non_lidar):
        lidar_batch_size = lidar.shape[:-2]
        non_lidar_batch_size = non_lidar.shape[:-2]
        if len(lidar.shape) > 3:
            if len(lidar.shape) == 4:
                lidar = einops.rearrange(lidar, 'b n c l -> (b n) c l')
        # lidar = lidar.unsqueeze(1)  # Add channel dimension
        feat = F.relu(self.act_fea_cv1(lidar))
        feat = F.relu(self.act_fea_cv2(feat))
        feat = feat.view(feat.shape[0], -1)
        # print("feat shape: ", feat.shape)
        # print("non_lidar shape: ", non_lidar.shape)
        # print("non_lidar shape: ",  non_lidar.view(-1, non_lidar.shape[-1]*non_lidar.shape[-2]).shape)
        feat = torch.cat((feat, non_lidar.view(-1, non_lidar.shape[-1]*non_lidar.shape[-2])), dim=-1)
        feat = F.relu(self.fc1(feat))
        feat = self.fc2(feat)
        feat = self.norm(feat)
        feat = einops.rearrange(feat, '(b n) d -> b n d', b=lidar_batch_size[0])
        return feat



class DiffusionUnetLowdimPolicyWithCNN1D(DiffusionUnetLowdimPolicy):
    def __init__(self, cnn_model,*args, **kwargs):
        super(DiffusionUnetLowdimPolicyWithCNN1D, self).__init__(*args, **kwargs)
        self.cnn_model = cnn_model
        self.normalizer = LinearNormalizer()
        self.obs_normalizer_params = nn.ParameterDict({
            'scale': torch.tensor([1.0]),
            'offset': torch.tensor([0.0])
        })
        self.normalizer.params_dict['obs'] = self.obs_normalizer_params

    # def predict_action(self, lidar, non_lidar) -> Dict[str, torch.Tensor]:
    #     # lidar = self.normalizer['lidar_data'].normalize(lidar)
    #     # non_lidar = self.normalizer['non_lidar_data'].normalize(non_lidar)
    #     batch = {'lidar_data': lidar, 'non_lidar_data': non_lidar}
    def predict_action(self, batch) -> Dict[str, torch.Tensor]:
        nbatch = self.normalizer.normalize(batch)
        obs = self.cnn_model(nbatch['lidar_data'], nbatch['non_lidar_data'])
        batch['obs'] = obs
        action = super(DiffusionUnetLowdimPolicyWithCNN1D, self).predict_action(batch)
        return action

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        print(self.normalizer.params_dict)
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.normalizer.params_dict['obs'] = self.obs_normalizer_params
        print(self.normalizer.params_dict)

    def compute_loss(self, batch):
        nbatch = self.normalizer.normalize(batch)
        obs = self.cnn_model(nbatch['lidar_data'], nbatch['non_lidar_data'])
        batch['obs'] = obs
        return super(DiffusionUnetLowdimPolicyWithCNN1D, self).compute_loss(batch)
    