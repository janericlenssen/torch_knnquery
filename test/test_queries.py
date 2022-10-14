from typing import Tuple

import torch
from torch import Tensor

from torch_knnquery import VoxelGrid
import pytest
from numpy.testing import assert_almost_equal


batch_size = 1
num_points = 1000
num_rays = 3
num_samples_per_ray = 100
k = 3
r = 4
max_shading_pts = 3 #24
torch.manual_seed(1234)
dtypes = [torch.float, torch.double]

def query_keypoints(x: Tensor, kp_pos: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    parameters:
    x: [B, num_rays, num_shading_pts, 3]
    kp_pos: [B, num_kp, 3]

    returns:
    neighbor_mask: [B, num_rays, max_shading_pts, num_kp]
    sample_loc: [B, num_rays, max_shading_pts, 3]
    mask: [B, num_rays, max_shading_pts, 1]
    """
    B, num_rays, num_shading_pts = x.shape[:3]
    num_kp = kp_pos.size(1)
    device = x.device
    dist = torch.cdist(x.view(B, -1, 3), kp_pos).view(B, num_rays, num_shading_pts, -1)             # [B, num_rays, num_shading_pts, num_kp]

    topk = torch.topk(dist, k, dim=-1, largest=False, sorted=False)

    mask = torch.zeros((B, num_rays, num_shading_pts, num_kp), dtype=torch.bool, device=device)
    mask.scatter_(-1, topk.indices, topk.values < r)

    valid_pts_mask = mask.max(dim=-1).values                                                        # [B, num_rays, num_shading_pts]
    pts_cumsum = torch.cumsum(valid_pts_mask, dim=-1)
    valid_pts_mask = torch.logical_and(valid_pts_mask, pts_cumsum <= max_shading_pts)           
    shading_loc = torch.masked_select(x, valid_pts_mask.unsqueeze(-1).expand(-1, -1, -1, 3))
    indices = torch.masked_select(topk.indices, valid_pts_mask.unsqueeze(-1).expand(-1, -1, -1, k))
    values = torch.masked_select(topk.values, valid_pts_mask.unsqueeze(-1).expand(-1, -1, -1, k))
    shading_loc = shading_loc.view(B, num_rays, max_shading_pts, 3)
    indices = indices.view(B, num_rays, max_shading_pts, k)
    values = values.view(B, num_rays, max_shading_pts, k)
    indices[values >= r] = -1

    return indices, shading_loc, values


def query_keypoints_voxel(voxel_grid, x: Tensor, kp_pos: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    parameters:
    x: [B, num_rays, num_shading_pts, 3]
    kp_pos: [B, num_kp, 3]

    returns:
    neighbor_mask: [B, num_rays, max_shading_pts, num_kp]
    sample_loc: [B, num_rays, max_shading_pts, 3]
    mask: [B, num_rays, max_shading_pts, 1]
    """
    B, num_rays = x.shape[:2]
    sample_idx, sample_loc, ray_mask = voxel_grid.query(x, k, r, max_shading_pts)
    sample_idx = torch.clamp(sample_idx, min=0).view(-1).long()
    sampled_neighb = torch.index_select(kp_pos, 1, sample_idx).view(B, num_rays, max_shading_pts, k, 3)
    distances =  torch.norm(sampled_neighb - sample_loc[..., None, :], dim=-1)
    sample_idx = sample_idx.view(batch_size, num_rays, max_shading_pts, k)
    
    return sample_idx, sample_loc, distances


@pytest.mark.parametrize('dtype', dtypes)
def test_knnquery(dtype):
    
    voxel_grid = VoxelGrid(
        voxel_size=(1., 1., 1.),                    # Voxel size for each dimension
        voxel_scale=(4., 4., 4.),                      # Voxel scale for each dimension
        kernel_size=(3, 3, 3),                      # Range of voxels searched for neighbors 
                                                    # (default: [3, 3, 3])
        max_points_per_voxel=num_points,                    # Maximum number of points stored in a voxel 
                                                    # (default: 26)
        max_occ_voxels_per_example=610000,          # Maximum number of occupied voxels per point cloud 
                                                    # (default: 600000)
        ranges=(-20.0,-20.0,-20.0,20.0,20.0,20.0)   # Maximum ranges the VoxelGrid spans 
                                                    # (default: inferred from data)
    )
    

    points_tensor = 15 * (torch.rand(batch_size, num_points, 3).cuda() -0.5)
    num_tensor = num_points * torch.ones(batch_size, dtype=torch.int).cuda()

    

    

    rays_o = 3*(torch.rand(batch_size, num_rays, num_samples_per_ray, 3).cuda()-0.5)
    rays_d = torch.rand(batch_size, num_rays, num_samples_per_ray, 3)-0.5
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1).cuda()

    depth = torch.linspace(-5, 5, num_samples_per_ray)[None, None, :, None].cuda()

    raypos_tensor = rays_o + depth * rays_d
    
    neighbor_idx, sample_loc, distances = query_keypoints(raypos_tensor, points_tensor)
    distances = distances.sort(dim=-1)[0]
    sample_loc = sample_loc.sort(dim=-1)[0]
    neighbor_idx = neighbor_idx.sort(dim=-1)[0]
    
    raypos_tensor = raypos_tensor.to(dtype)
    points_tensor = points_tensor.to(dtype)

    voxel_grid.set_pointset(
        points=points_tensor,                       # Tensor of size [B, max_num_points, 3] containing 
                                                    # B point clouds.
        actual_num_points_per_example=num_tensor   # Tensor of size [B] containing the actual
                                                    # num_points<=max_num_points for each point cloud.
    )

    neighbor_idx_2, sample_loc_2, distances_2 = query_keypoints_voxel(voxel_grid, raypos_tensor, points_tensor)
    distances_2 = distances_2.sort(dim=-1)[0]
    sample_loc_2 = sample_loc_2.sort(dim=-1)[0]
    neighbor_idx_2 = neighbor_idx_2.sort(dim=-1)[0]

    assert_almost_equal(distances.cpu().numpy(), distances_2.cpu().numpy(), decimal=5)
    assert_almost_equal(sample_loc.cpu().numpy(), sample_loc_2.cpu().numpy(), decimal=5)
    assert torch.all(neighbor_idx == neighbor_idx_2)
    
