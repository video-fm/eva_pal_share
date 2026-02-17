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
import pickle
from collections import deque

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
from eva.utils.misc_utils import now_hms, blue_print
from eva.utils.parameters import hand_camera_id

from transformers import AutoImageProcessor, AutoModel
from collections import namedtuple

from sklearn.decomposition import PCA


def obs_encoder(state_input_dim, enc_dim, latent_dim, privileged=False):
    """Observation is a dictionary with keys:
    'robot_state': (B, 12) --> Now (B, 6)
    'robot_state_history': (B, 5*6)
    'patch_tokens': (B, 256 * 16)
    'segmentation_mask': (B, 1, 84, 84)
    'occupancy_grid': (B, 1, 84, 84)
    We will concatenate the robot_state and patch_tokens and call that the state. The state is processed by an MLP into latent dim. 
    The segmentation mask is processed by a CNN into a latent dim.

    The privileged information is the segmentation mask.
    """
    encoders = {}
    # NOTE: currently the input dim of the first layer is 6 + 5*6 + 4096 = 4132. Do we want a separate encoder for the history?
    state_enc_layers = [
        nn.Linear(state_input_dim, enc_dim), nn.ELU(),
        nn.Linear(enc_dim, enc_dim), nn.LayerNorm(enc_dim), nn.ELU(),
        nn.Linear(enc_dim, latent_dim), nn.LayerNorm(latent_dim), nn.Sigmoid()
    ]
    encoders['state'] = nn.Sequential(*state_enc_layers)
    C = 1
    num_channels = 32
    img_size = 84
    occupancy_enc_layers = [
            h.NormalizeImg(),
            nn.Conv2d(C, num_channels, 7, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU()
    ]
    out_shape = h._get_out_shape((C, img_size, img_size), occupancy_enc_layers)
    occupancy_enc_layers.extend([
        h.Flatten(), nn.Linear(np.prod(out_shape), latent_dim),
        nn.LayerNorm(latent_dim), nn.Sigmoid()
    ])
    encoders['occupancy_grid'] = h.ConvExt(nn.Sequential(*occupancy_enc_layers))
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
    """Encoder and policy head for BC."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        #####
        # state_input_dim = cfg.obs_shape['robot_state'][0] + cfg.obs_shape['patch_tokens'][0]
        state_input_dim = cfg.obs_shape['robot_state_history'][0] + cfg.obs_shape['robot_state'][0] + cfg.obs_shape['patch_tokens'][0]
        ######
        self._Q_encoder = obs_encoder(state_input_dim, cfg.enc_dim, cfg.latent_dim, privileged=True)
        self._Qs = nn.ModuleList([q_head(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim) for _ in range(cfg.num_q)])
        self._V_encoder = obs_encoder(state_input_dim, cfg.enc_dim, cfg.latent_dim, privileged=True)
        self._V = v_head(cfg.latent_dim, cfg.mlp_dim)
        self._pi_encoder = obs_encoder(state_input_dim, cfg.enc_dim, cfg.latent_dim, privileged=False)
        self._pi = mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim * cfg.action_chunk_size)
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
            out =  h.TruncatedNormal(mu, std).sample(clip=0.3)
            return out.reshape(-1, self.cfg.action_chunk_size, self.cfg.action_dim)
        return mu.reshape(-1, self.cfg.action_chunk_size, self.cfg.action_dim)
    
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

# Fiona's occupancy generation logic
def rpy_to_direction(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R @ np.array([0, 0, 1])

def intersect_plane(origin, direction, plane_normal, plane_point):
    denom = np.dot(plane_normal, direction)
    if np.abs(denom) < 1e-6:
        return None, None
    t = np.dot(plane_point - origin, plane_normal) / denom
    if t < 0:
        return None, None
    return origin + t * direction, t

def paint_circle(grid, px, py, res, radius):
    height, width = grid.shape
    cx = int(px / res)
    cy = int(py / res)
    rad_px = int(radius / res)
    Y, X = np.ogrid[:height, :width]
    mask = (X - cx)**2 + (Y - cy)**2 <= rad_px**2
    grid[mask] = 1.0

def generate_occupancy_grids_with_circle(
    positions, angles, res=0.01, spray_radius=0.15, most_recent=False
):
    # Bounds
    x_h_min, x_h_max = -0.2, 1.0
    y_min, y_max     = -0.9, 0.7
    z_min, z_max     = 0.0, 1.0

    h_w = int((x_h_max - x_h_min) / res)
    h_h = int((y_max - y_min) / res)
    v_w = int((z_max - z_min) / res)
    v_h = h_h

    images = []
    horizontal_grid = np.zeros((h_h, h_w))
    vertical_grid   = np.zeros((v_h, v_w))

    for (x, y, z), (roll, pitch, yaw) in zip(positions, angles):
        origin = np.array([x, y, z])
        direction = rpy_to_direction(roll, pitch, yaw)

        # XY plane (z = 0)
        point_xy, _ = intersect_plane(origin, direction, np.array([0, 0, 1]), np.array([0, 0, 0]))
        if point_xy is not None:
            px, py = point_xy[0], point_xy[1]
            if (x_h_min <= px <= x_h_max) and (y_min <= py <= y_max):
                paint_circle(horizontal_grid, px - x_h_min, py - y_min, res, spray_radius)

        # YZ plane (x = 1)
        point_yz, _ = intersect_plane(origin, direction, np.array([1, 0, 0]), np.array([1, 0, 0]))
        if point_yz is not None:
            py, pz = point_yz[1], point_yz[2]
            if (y_min <= py <= y_max) and (z_min <= pz <= z_max):
                paint_circle(vertical_grid, pz - z_min, py - y_min, res, spray_radius)

        # Combine
        combined = np.concatenate([horizontal_grid, vertical_grid], axis=1)
        combined = np.rot90(combined)
        # combined = np.flipud(combined)
        combined = np.fliplr(combined)
        # assert combined.shape == (220, 140), f"Got {combined.shape}, expected (220, 140)"
        images.append(combined.copy())
    # convert images to numpy array
    images = np.array(images, dtype=bool)

    # SAVE image - convert boolean to uint8 and scale to 0-255
    save_image = (images[-1].astype(np.uint8) * 255)
    # Create directory if it doesn't exist
    os.makedirs("aawr_pca_history/vis", exist_ok=True)
    cv2.imwrite(f"aawr_pca_history/vis/occupancy_grid_{now_hms()}.png", save_image)

    if most_recent:
        return images[-1]
    return images

class AAWRPolicy:
    """
    AAWR policy for the active perception project.
    It takes in a 224x224x3 image and encodes it to DINO features.
    PCA(16) of the patch tokens.
    It takes in the cartesian position, gripper position, and joint positions to make a 14 dimensional vector.
    """

    def __init__(self, policy_path, pca_path="models/dinov2/pca_model_pca_16.pkl", action_space="cartesian_velocity", gripper_action_space="velocity"):
        action_space, gripper_action_space = "cartesian_velocity", "velocity"

        self.action_space = action_space
        self.gripper_action_space = gripper_action_space
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_path = policy_path
        self.pca_path = pca_path

        cfg = namedtuple('cfg', ['obs_shape', 'enc_dim', 'latent_dim', 'mlp_dim', 'num_q', 'action_dim', 'pca_components', 'history_length', 'action_chunk_size'])
        cfg.pca_components = 16
        cfg.obs_shape = {
            #####
            'robot_state': (6,),
            'robot_state_history': (5 * 6,),
            ######
            'patch_tokens': (256 * cfg.pca_components,),
            'segmentation_mask': (84, 84),
            'occupancy_grid': (84, 84),
        }
        cfg.enc_dim = 512
        cfg.latent_dim = 50
        cfg.mlp_dim = 1024
        cfg.num_q = 5
        cfg.action_dim = 6
        cfg.action_chunk_size = 10
        cfg.history_length = 5
        self.aawr = AAWRNetwork(cfg).to(self.device)
        self.history_length = cfg.history_length
        self.cfg = cfg

        self.hand_camera_id = hand_camera_id

        blue_print(f"======================Loading policy from {self.policy_path}")
        d = torch.load(self.policy_path)
            
        self.aawr.load_state_dict(d['model'])
        self.aawr.eval()

        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small', use_fast=False)
        self.image_encoder = AutoModel.from_pretrained('facebook/dinov2-small').to(self.device)
        self.image_encoder.eval()
        self.counter = 0

        # load PCA model
        blue_print("Loading PCA model from: ", self.pca_path)
        with open(self.pca_path, 'rb') as f:
            self.pca = pickle.load(f)
        # history of cartesian positions and angles for occupancy grid
        self.prev_positions = []
        self.prev_angles = []

        # history of cartesian positions, max size is 6, since its: [t-5, t-4, t-3, t-2, t-1, t]. 
        self.cartesian_pos_history = deque(maxlen=self.cfg.history_length+1)

        # keeps the action queue, since policy outputs a chunk of actions. 
        self.action_execution_horizon = 5
        self.action_queue = deque(maxlen=self.action_execution_horizon)

    def get_name(self):
        return "pca_history_aawr_policy"
    
    def get_policy_name(self):
        return self.policy_path

    def forward(self, obs):
        robot_state_dict = obs["robot_state"]
        cartesian_position = robot_state_dict["cartesian_position"]
        gripper_position = robot_state_dict["gripper_position"]
        joint_positions = robot_state_dict["joint_positions"]
        robot_state = np.concatenate([cartesian_position, [gripper_position], joint_positions], axis=-1)
        self.prev_positions.append(cartesian_position[:3])
        self.prev_angles.append(cartesian_position[3:])
        # if robbot state history is empty, first fill it up to max size.   
        if len(self.cartesian_pos_history) == 0:
            for _ in range(self.cfg.history_length+1):
                self.cartesian_pos_history.append(cartesian_position)
        else:
            self.cartesian_pos_history.append(cartesian_position)
        
        if len(self.action_queue) > 0:
            action = self.action_queue.popleft()
        else:
            # use the wrist camera.
            img_array = obs["image"][f"{self.hand_camera_id}_left"]
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
                # METHOD 2: PCA(3)
                patch_tokens = last_hidden_states[:, 1:]   # (B, 256, 384)
                patch_tokens = patch_tokens.reshape(-1, 384) # (B * 256, 384)
                # Move tensor to CPU before PCA transformation
                patch_tokens = patch_tokens.cpu().numpy()
                pca_patches = self.pca.transform(patch_tokens) # (B * 256, 16)
                pca_patches = pca_patches.reshape(1, 256 * 16) # (B, 256 * 16)
                # generate occupancy grid
                raw_occupancy_image = generate_occupancy_grids_with_circle(
                    self.prev_positions, self.prev_angles, res=0.01, spray_radius=0.05, most_recent=True
                )
                occupancy_grid = cv2.resize(raw_occupancy_image.astype(np.uint8), (84, 84), interpolation=cv2.INTER_NEAREST)
                # Add batch and channel dimensions to occupancy grid
                occupancy_grid = torch.from_numpy(occupancy_grid).to(self.device).float()
                occupancy_grid = occupancy_grid.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions [1, 1, 84, 84]
                
                # give it [t-5, t-4, t-3, t-2, t-1, t] cartesian positions. 
                cartesian_pos_history = torch.from_numpy(np.array(self.cartesian_pos_history)).to(self.device).float()
                cartesian_pos_history = cartesian_pos_history.flatten()[None]
                pca_patches = torch.from_numpy(pca_patches).to(self.device).float()
                policy_input = {
                    'state': torch.concatenate([cartesian_pos_history, pca_patches], dim=-1),
                    'occupancy_grid': occupancy_grid,
                }
                z = self.aawr.encode_pi(policy_input)
                action_chunk = self.aawr.pi(z).squeeze(0).cpu().numpy() # (10, 6)
                for idx in range(self.action_execution_horizon):
                    action = action_chunk[idx]
                    # add a dummy gripper action.
                    action = np.concatenate([action, [0.0]], axis=-1)
                    self.action_queue.append(action)

        time.sleep(0.5)
        action = np.clip(action, -1, 1)
        blue_print("POLICY ACTION:", action)
        self.counter += 1
        if self.counter % 100 == 0:
            blue_print(f"Counter: {self.counter}")

        return action, {}

    def save_grid(self, save_filepath):
        """
        Save the most recent occupancy grid to EVA recording dir filepath.
        eg: '/home/franka/eva_tony/eva/utils/../../data/failure/2025-05-11/2025-05-11_18-13-29/trajectory.h5'
        """
        if not self.prev_positions or not self.prev_angles:
            blue_print("No occupancy grid data available to save")
            return
            
        raw_occupancy_image = generate_occupancy_grids_with_circle(
            self.prev_positions, self.prev_angles, res=0.01, spray_radius=0.05, most_recent=True
        )
        
            # Convert boolean to uint8 and scale to 0-255
        save_image = (raw_occupancy_image.astype(np.uint8) * 255)
        
        save_dir = save_filepath.replace("trajectory.h5", f"grid")
        os.makedirs(save_dir, exist_ok=True)
        save_filepath = os.path.join(save_dir, f"{now_hms()}.png")
        
        success = cv2.imwrite(save_filepath, save_image)
        if success:
            # blue_print(f"Occupancy grid saved to {save_filepath}")
            pass
        else:
            blue_print(f"Failed to save occupancy grid to {save_filepath}")
    
    def reset_state(self):
        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
        }
        self.counter = 0
        self.history = []
        self.prev_positions = []
        self.prev_angles = []
        self.cartesian_pos_history.clear()
        self.action_queue.clear()

    def get_info(self):
        return self._state
    
    def register_key(self, key):
        if key == ord(" "):
            self._state["movement_enabled"] = not self._state["movement_enabled"]
            blue_print("Movement enabled:", self._state["movement_enabled"])
        elif key == ord("r"):
            self.reset_state()
            blue_print("State reset")
        elif key == ord("y"):
            self._state["success"] = True
            blue_print("Success")
        elif key == ord("n"):
            self._state["failure"] = True
            blue_print("Failure")

# Backward-compatibility alias
AAWRController = AAWRPolicy

