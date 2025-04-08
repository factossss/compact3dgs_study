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
import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, itr=-1, rvq_iter=False):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # Get scaling and rotation parameters. If iteration is -1 (training finished), use the ones from the model. Otherwise, use the ones
    # if itr == -1:
    #     scales = pc._scaling
    #     rotations = pc._rotation
    #     opacity = pc._opacity
    if itr == -1:
        scales = pc._scaling
        rotations = pc._rotation
        opacity = pc._opacity
        shs = pc.get_features
        
    else:
        if rvq_iter:
            scales = pc.vq_scale(pc.get_scaling.unsqueeze(0))[0] # 通过VQ-VAE获得的尺度
            rotations = pc.vq_rot(pc.get_rotation.unsqueeze(0))[0]
            features_rest = pc.vq_shs(pc._features_rest.flatten().unsqueeze(0))[0]
            scales = scales.squeeze()
            rotations = rotations.squeeze()
            features_rest = features_rest.squeeze().reshape(pc._xyz.shape[0], 3, (pc.max_sh_degree + 1) ** 2 - 1)
            shs = torch.cat((pc._features_dc, features_rest), dim=1)
            opacity = pc.get_opacity

        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation
            opacity = pc.get_opacity 
            shs = pc.get_features
            
    # shs = pc.get_features
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D.float(),
        means2D = means2D,
        shs = shs.float(),
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_img(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, itr=-1, rvq_iter=False):
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # Get scaling and rotation parameters. If iteration is -1 (training finished), use the ones from the model. Otherwise, use the ones
    if itr == -1:
        scales = pc._scaling
        rotations = pc._rotation
        opacity = pc._opacity
        shs = pc.get_features
        
    else:
        if rvq_iter:
            scales = pc.vq_scale(pc.get_scaling.unsqueeze(0))[0] # 通过VQ-VAE获得的尺度
            rotations = pc.vq_rot(pc.get_rotation.unsqueeze(0))[0]
            features_rest = pc.vq_shs(pc._features_rest.flatten().unsqueeze(0))[0]
            scales = scales.squeeze()
            rotations = rotations.squeeze()
            features_rest = features_rest.squeeze().reshape(pc._xyz.shape[0], 3, (pc.max_sh_degree + 1) ** 2 - 1)
            shs = torch.cat((pc._features_dc, features_rest), dim=1)
            opacity = pc.get_opacity

        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation
            opacity = pc.get_opacity 
            shs = pc.get_features
            
    # shs = pc.get_features
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, \
    accum_weights_ptr, accum_weights_count, accum_max_count  = rasterizer(  #这里有修改
        means3D = means3D.float(),
        means2D = means2D,
        shs = shs.float(),
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "accum_weights": accum_weights_ptr,
            "area_proj": accum_weights_count,
            "area_max": accum_max_count,
            }