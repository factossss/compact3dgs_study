#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

from einops import reduce
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import math

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, model):
        self.active_sh_degree = 0
        self.max_sh_degree = model.sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        
        self.vq_scale = ResidualVQ(dim = 3, codebook_size = model.rvq_size, num_quantizers = model.rvq_num, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, learnable_codebook=True, in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
        self.vq_rot = ResidualVQ(dim = 4, codebook_size = model.rvq_size, num_quantizers = model.rvq_num, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, learnable_codebook=True, in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
        # vq of features_rest
        self.vq_sh_1 = ResidualVQ(dim = 3 * 3, codebook_size = model.rvq_size, num_quantizers = model.rvq_num, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, learnable_codebook=True, in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
        self.vq_sh_2 = ResidualVQ(dim = 3 * 5, codebook_size = model.rvq_size, num_quantizers = model.rvq_num, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, learnable_codebook=True, in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
        self.vq_sh_3 = ResidualVQ(dim = 3 * 7, codebook_size = model.rvq_size, num_quantizers = model.rvq_num, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, learnable_codebook=True, in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
        self.rvq_bit = math.log2(model.rvq_size) # 用于后续的内存计算
        self.rvq_num = model.rvq_num

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_npz(self, path):
        mkdir_p(os.path.dirname(path))

        save_dict = dict()
        
        save_dict["xyz"] = self._xyz.detach().cpu().half().numpy()
        save_dict["opacity"] = self._opacity.detach().cpu().half().numpy()
        save_dict["scale"] = np.packbits(np.unpackbits(self.sca_idx.unsqueeze(-1).cpu().numpy().astype(np.uint8), axis=-1, count=int(self.rvq_bit), bitorder='little').flatten(), axis=None)
        save_dict["rotation"] = np.packbits(np.unpackbits(self.rot_idx.unsqueeze(-1).cpu().numpy().astype(np.uint8), axis=-1, count=int(self.rvq_bit), bitorder='little').flatten(), axis=None)
        save_dict["sh1"] = np.packbits(np.unpackbits(self.sh1_idx.unsqueeze(-1).cpu().numpy().astype(np.uint8), axis=-1, count=int(self.rvq_bit), bitorder='little').flatten(), axis=None)
        save_dict["sh2"] = np.packbits(np.unpackbits(self.sh2_idx.unsqueeze(-1).cpu().numpy().astype(np.uint8), axis=-1, count=int(self.rvq_bit), bitorder='little').flatten(), axis=None)
        save_dict["sh3"] = np.packbits(np.unpackbits(self.sh3_idx.unsqueeze(-1).cpu().numpy().astype(np.uint8), axis=-1, count=int(self.rvq_bit), bitorder='little').flatten(), axis=None)
        # save_dict["hash"] = self.recolor.params.cpu().half().numpy()
        # save_dict["mlp"] = self.mlp_head.params.cpu().half().numpy()
        save_dict["features_dc"] = self._features_dc.detach().cpu().half().numpy()
        # save_dict["features_rest"] = self._features_rest.detach().cpu().half().numpy()
        save_dict["codebook_scale"] = self.vq_scale.cpu().state_dict()
        save_dict["codebook_rotation"] = self.vq_rot.cpu().state_dict()
        save_dict["codebook_sh_1"] = self.vq_sh_1.cpu().state_dict()
        save_dict["codebook_sh_2"] = self.vq_sh_2.cpu().state_dict()
        save_dict["codebook_sh_3"] = self.vq_sh_3.cpu().state_dict()
        save_dict["rvq_info"] = np.array([int(self.rvq_num), int(self.rvq_bit)])
        
        np.savez(path, **save_dict)
        
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path+".ply")

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)) 

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def load_model(self, path):
        if os.path.isfile(path + '.npz'):
            path = path + '.npz'
            print("Loading ", path)
            load_dict = np.load(path, allow_pickle=True)

            scale = np.packbits(np.unpackbits(load_dict["scale"], axis=None)[:load_dict["xyz"].shape[0]*load_dict["rvq_info"][0]*load_dict["rvq_info"][1]].reshape(-1, load_dict["rvq_info"][1]), axis=-1, bitorder='little')
            rotation = np.packbits(np.unpackbits(load_dict["rotation"], axis=None)[:load_dict["xyz"].shape[0]*load_dict["rvq_info"][0]*load_dict["rvq_info"][1]].reshape(-1, load_dict["rvq_info"][1]), axis=-1, bitorder='little')
            sh1 = np.packbits(np.unpackbits(load_dict["sh1"], axis=None)[:load_dict["xyz"].shape[0]*load_dict["rvq_info"][0]*load_dict["rvq_info"][1]].reshape(-1, load_dict["rvq_info"][1]), axis=-1, bitorder='little')
            sh2 = np.packbits(np.unpackbits(load_dict["sh2"], axis=None)[:load_dict["xyz"].shape[0]*load_dict["rvq_info"][0]*load_dict["rvq_info"][1]].reshape(-1, load_dict["rvq_info"][1]), axis=-1, bitorder='little')
            sh3 = np.packbits(np.unpackbits(load_dict["sh3"], axis=None)[:load_dict["xyz"].shape[0]*load_dict["rvq_info"][0]*load_dict["rvq_info"][1]].reshape(-1, load_dict["rvq_info"][1]), axis=-1, bitorder='little')

            self.vq_scale.load_state_dict(load_dict["codebook_scale"].item())
            self.vq_rot.load_state_dict(load_dict["codebook_rotation"].item())
            self.vq_sh_1.load_state_dict(load_dict["codebook_sh_1"].item())
            self.vq_sh_2.load_state_dict(load_dict["codebook_sh_2"].item())
            self.vq_sh_3.load_state_dict(load_dict["codebook_sh_3"].item())
            scale_codes = self.vq_scale.get_codes_from_indices(torch.from_numpy(scale).cuda().reshape(-1,1,load_dict["rvq_info"][0]).long())
            scale = self.vq_scale.project_out(reduce(scale_codes, 'q ... -> ...', 'sum'))
            rotation_codes = self.vq_rot.get_codes_from_indices(torch.from_numpy(rotation).cuda().reshape(-1,1,load_dict["rvq_info"][0]).long())
            rotation = self.vq_rot.project_out(reduce(rotation_codes, 'q ... -> ...', 'sum'))
            
            sh1_codes = self.vq_sh_1.get_codes_from_indices(torch.from_numpy(sh1).cuda().reshape(-1,1,load_dict["rvq_info"][0]).long())
            sh1 = self.vq_sh_1.project_out(reduce(sh1_codes, 'q ... -> ...', 'sum'))
            sh2_codes = self.vq_sh_2.get_codes_from_indices(torch.from_numpy(sh2).cuda().reshape(-1,1,load_dict["rvq_info"][0]).long())
            sh2 = self.vq_sh_2.project_out(reduce(sh2_codes, 'q ... -> ...', 'sum'))
            sh3_codes = self.vq_sh_3.get_codes_from_indices(torch.from_numpy(sh3).cuda().reshape(-1,1,load_dict["rvq_info"][0]).long())
            sh3 = self.vq_sh_3.project_out(reduce(sh3_codes, 'q ... -> ...', 'sum'))

            self._xyz = nn.Parameter(torch.from_numpy(load_dict["xyz"]).cuda().float().requires_grad_(True))
            self._opacity = nn.Parameter(torch.from_numpy(load_dict["opacity"]).reshape(-1,1).cuda().float().requires_grad_(True))
            self._scaling = nn.Parameter(scale.squeeze(1).requires_grad_(True))
            self._rotation = nn.Parameter(rotation.squeeze(1).requires_grad_(True))
            # self.recolor.params = nn.Parameter(torch.from_numpy(load_dict["hash"]).cuda().half().requires_grad_(True))
            # self.mlp_head.params = nn.Parameter(torch.from_numpy(load_dict["mlp"]).cuda().half().requires_grad_(True))
            self._features_dc = nn.Parameter(torch.from_numpy(load_dict["features_dc"]).contiguous().cuda().half().requires_grad_(True))
            # self._features_rest = nn.Parameter(torch.from_numpy(load_dict["features_rest"]).contiguous().cuda().half().requires_grad_(True)
            sh1 = sh1.squeeze(1).reshape(self._xyz.shape[0], 3, 3)
            sh2 = sh2.squeeze(1).reshape(self._xyz.shape[0], 5, 3)
            sh3 = sh3.squeeze(1).reshape(self._xyz.shape[0], 7, 3)
            self._features_rest = nn.Parameter(torch.cat((sh1, sh2, sh3), dim=1).requires_grad_(True))
            self.active_sh_degree = self.max_sh_degree
        else:
            self.load_ply(path)
            
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        
    def densify_and_prune_split(self, max_grad, min_opacity, extent, max_screen_size, mask):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split_mask(grads, max_grad, extent, mask)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def densify_and_split_mask(self, grads, grad_threshold, scene_extent, mask, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        padded_mask = torch.zeros((n_init_points), dtype=torch.bool, device='cuda')
        padded_mask[:grads.shape[0]] = mask # mask_blur
        selected_pts_mask = torch.logical_or(selected_pts_mask, padded_mask)
        

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
    def final_prune(self): # 导出模型时使用
        
        for m in self.vq_scale.layers:
            m.training = False
        for m in self.vq_rot.layers: 
            m.training = False
        for m in self.vq_sh_1.layers:
            m.training = False
        for m in self.vq_sh_2.layers:
            m.training = False
        for m in self.vq_sh_3.layers:
            m.training = False

        self._scaling, self.sca_idx, _ = self.vq_scale(self.get_scaling.unsqueeze(1))
        self._rotation, self.rot_idx, _ = self.vq_rot(self.get_rotation.unsqueeze(1))
        sh1, self.sh1_idx, _ = self.vq_sh_1(self.get_features[:, 1:4, :].reshape(self._xyz.shape[0], -1).unsqueeze(1))
        sh2, self.sh2_idx, _ = self.vq_sh_2(self.get_features[:, 4:9, :].reshape(self._xyz.shape[0], -1).unsqueeze(1))
        sh3, self.sh3_idx, _ = self.vq_sh_3(self.get_features[:, 9:16, :].reshape(self._xyz.shape[0], -1).unsqueeze(1))
        self._scaling = self._scaling.squeeze()
        self._rotation = self._rotation.squeeze()
        sh1 = sh1.squeeze().reshape(self._xyz.shape[0], 3, 3)
        sh2 = sh2.squeeze().reshape(self._xyz.shape[0], 5, 3)
        sh3 = sh3.squeeze().reshape(self._xyz.shape[0], 7, 3)
        # features_rest = features_rest.squeeze().reshape(pc._xyz.shape[0], 3, (pc.max_sh_degree + 1) ** 2 - 1)
        # shs = torch.cat((pc._features_dc, features_rest), dim=1)
        self._features_rest = torch.cat((sh1, sh2, sh3), dim=1)
        self._opacity = self.get_opacity

        torch.cuda.empty_cache()