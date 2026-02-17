import torch
import json
import numpy as np
import torchvision.transforms as T
from PIL import Image

from eva.controllers.replayer import Replayer
import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(precision=3, suppress=True)
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import concurrent.futures

import time
import random
from pathlib import Path
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'
import os
import h5py
import cv2
import eva.utils.aawr_utils as h
from eva.utils.parameters import hand_camera_id

from transformers import AutoImageProcessor, AutoModel
from collections import namedtuple

def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def obs_encoder(state_input_dim, enc_dim, latent_dim, privileged=False):
    """Observation is a dictionary with keys:
    'robot_state': (B, 12)
    'avg_patch_tokens': (B, 384)
    'segmentation_mask': (B, 1, 224, 224)
    We will concatenate the robot_state and avg_patch_tokens and call that the state. The state is processed by an MLP into latent dim. 
    The segmentation mask is processed by a CNN into a latent dim.

    The privileged information is the segmentation mask.
    """
    encoders = {}
    state_enc_layers = [
        nn.Linear(state_input_dim, enc_dim), nn.ELU(),
        nn.Linear(enc_dim, latent_dim), nn.LayerNorm(latent_dim), nn.Sigmoid()
    ]
    encoders['state'] = nn.Sequential(*state_enc_layers)
    if privileged:
        C = 1
        num_channels = 32
        img_size = 84    
        segmentation_enc_layers = [
                h.NormalizeImg(),
                nn.Conv2d(C, num_channels, 7, stride=2), nn.ReLU(),
                nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(),
                nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(),
                nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU()
        ]
        out_shape = h._get_out_shape((C, img_size, img_size), segmentation_enc_layers)
        segmentation_enc_layers.extend([
            h.Flatten(), nn.Linear(np.prod(out_shape), latent_dim),
            nn.LayerNorm(latent_dim), nn.Sigmoid()
        ])
        encoders['segmentation'] = h.ConvExt(nn.Sequential(*segmentation_enc_layers))
    return h.Multiplexer(nn.ModuleDict(encoders))


def q_head(input_dim, mlp_dim):
    """Returns a Q-function that uses Layer Normalization."""
    return nn.Sequential(nn.Linear(input_dim, mlp_dim), nn.LayerNorm(mlp_dim), nn.Tanh(),
                        nn.Linear(mlp_dim, mlp_dim), nn.ELU(),
                        nn.Linear(mlp_dim, 1))

def v_head(input_dim, mlp_dim):
    """Returns a state value function that uses Layer Normalization."""
    return nn.Sequential(nn.Linear(input_dim, mlp_dim), nn.LayerNorm(mlp_dim), nn.Tanh(),
                        nn.Linear(mlp_dim, mlp_dim), nn.ELU(),
                        nn.Linear(mlp_dim, 1))

def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.Mish()):
    """Returns an MLP."""
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    return nn.Sequential(
        nn.Linear(in_dim, mlp_dim[0]),
        nn.LayerNorm(mlp_dim[0]),
        act_fn,
        nn.Linear(mlp_dim[0], mlp_dim[1]),
        nn.LayerNorm(mlp_dim[1]),
        act_fn,
        nn.Linear(mlp_dim[1], out_dim),
    )

class AAWRNetwork(nn.Module):
    """Encoder and policy head for AAWR."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        state_input_dim = cfg.obs_shape['robot_state'][0] + cfg.obs_shape['avg_patch_tokens'][0]
        self._Q_encoder = obs_encoder(state_input_dim, cfg.enc_dim, cfg.latent_dim, privileged=True)
        self._Qs = nn.ModuleList([q_head(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim) for _ in range(cfg.num_q)])
        self._V_encoder = obs_encoder(state_input_dim, cfg.enc_dim, cfg.latent_dim, privileged=True)
        self._V = v_head(cfg.latent_dim, cfg.mlp_dim)
        self._pi_encoder = obs_encoder(state_input_dim, cfg.enc_dim, cfg.latent_dim, privileged=False)
        self._pi = mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
        self.apply(h.orthogonal_init)     
        for m in [*self._Qs, self._V]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in self._Qs:
            h.set_requires_grad(m, enable)

    def track_v_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        if hasattr(self, '_V'):
            h.set_requires_grad(self._V, enable)

    def encode_pi(self, obs):
        out = self._pi_encoder(obs)
        if isinstance(obs, dict):
            # fusion
            out = torch.stack([v for k, v in out.items()]).mean(dim=0)
        return out
    
    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return h.TruncatedNormal(mu, std).sample(clip=0.3)
        return mu
    
    def encode_V(self, obs):
        out = self._V_encoder(obs)
        if isinstance(obs, dict):
            # fusion
            out = torch.stack([v for k, v in out.items()]).mean(dim=0)
        return out
    
    def V(self, z):
        """Predict state value (V)."""
        return self._V(z)

    def encode_Q(self, obs):
        out = self._Q_encoder(obs)
        if isinstance(obs, dict):
            # fusion
            out = torch.stack([v for k, v in out.items()]).mean(dim=0)
        return out
    
    def Q(self, z, a, return_type):
        """Predict state-action value (Q)."""
        assert return_type in {'min', 'avg', 'all'}
        x = torch.cat([z, a], dim=-1)

        if return_type == 'all':
            return torch.stack(list(q(x) for q in self._Qs), dim=0)

        idxs = np.random.choice(self.cfg.num_q, 2, replace=False)
        Q1, Q2 = self._Qs[idxs[0]](x), self._Qs[idxs[1]](x)
        return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2

class AAWRAvgPoolPolicy:
    """
    AAWR policy for the active perception project.
    It takes in a 224x224x3 image and encodes it to DINO features.
    It takes in the cartesian position, gripper position, and joint positions to make a 14 dimensional vector.
    """

    def __init__(self, policy_path, action_space="cartesian_velocity", gripper_action_space="velocity"):
        action_space, gripper_action_space = "cartesian_velocity", "velocity"

        self.action_space = action_space
        self.gripper_action_space = gripper_action_space
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_path = policy_path

        cfg = namedtuple('cfg', ['obs_shape', 'enc_dim', 'latent_dim', 'mlp_dim', 'num_q', 'action_dim'])
        cfg.obs_shape = {
            'robot_state': (14,),
            'avg_patch_tokens': (384,),
            'segmentation_mask': (84, 84),
        }
        cfg.enc_dim = 256
        cfg.latent_dim = 50
        cfg.mlp_dim = 512
        cfg.num_q = 5
        cfg.action_dim = 7
        self.aawr = AAWRNetwork(cfg).to(self.device)

        print(f"======================Loading policy from {self.policy_path}")
        d = torch.load(self.policy_path)
            
        self.aawr.load_state_dict(d['model'])
        self.aawr.eval()

        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small', use_fast=False)
        self.image_encoder = AutoModel.from_pretrained('facebook/dinov2-small').to(self.device)
        self.image_encoder.eval()
        self.counter = 0


    def get_name(self):
        return "aawr_policy"
    
    def get_policy_name(self):
        return self.policy_path

    
    def forward(self, obs):
        robot_state_dict = obs["robot_state"]
        cartesian_position = robot_state_dict["cartesian_position"]
        gripper_position = robot_state_dict["gripper_position"]
        joint_positions = robot_state_dict["joint_positions"]
        robot_state = np.concatenate([cartesian_position, [gripper_position], joint_positions], axis=-1)
        # use the wrist camera.
        img_array = obs["image"][f"{hand_camera_id}_left"]
                
        # Convert BGR to RGB before creating PIL Image
        # OpenCV uses BGR by default, PIL expects RGB
        img_array_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Create PIL image from the RGB array
        img = Image.fromarray(img_array_rgb)

        inputs = self.processor(images=[img], return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
        with torch.no_grad():
            outputs = self.image_encoder(**inputs) # 14 * 14 Patch sizse 
            last_hidden_states = outputs.last_hidden_state 
            # get patch tokens and average them, assuming CLS is the first token in the sequence.
            avg_patch_tokens = last_hidden_states[:,1:].mean(dim=1).cpu().numpy()  # (batch, 384)
            # 'state': torch.cat([batch['robot_state'], batch['avg_patch_tokens']], dim=-1).to(self.device).float(),
            policy_input = {
                'state': torch.from_numpy(np.concatenate([robot_state[None,:], avg_patch_tokens], axis=-1) ).to(self.device).float(),
            }
            z = self.aawr.encode_pi(policy_input)
            action = self.aawr.pi(z).squeeze(0).cpu().numpy()
        # for now use zeros
        time.sleep(0.1)
        action = np.clip(action, -1, 1)
        print("POLICY ACTION:", action)
        self.counter += 1
        if self.counter % 10 == 0:
            print(f"Counter: {self.counter}")

        return action, {}


    
    def reset_state(self):
        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
        }
        self.counter = 0
    
    def get_info(self):
        return self._state
    
    def register_key(self, key):
        if key == ord(" "):
            self._state["movement_enabled"] = not self._state["movement_enabled"]
            print("Movement enabled:", self._state["movement_enabled"])
        elif key == ord("r"):
            self.reset_state()
            print("State reset")
        elif key == ord("y"):
            self._state["success"] = True
            print("Success")
        elif key == ord("n"):
            self._state["failure"] = True
            print("Failure")

# Backward-compatibility alias
Policy = AAWRAvgPoolPolicy

if __name__ == "__main__":
    policy = AAWRAvgPoolPolicy()
