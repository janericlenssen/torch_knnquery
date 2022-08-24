import torch
from typing import Tuple
import time

if torch.cuda.is_available():
    import knnquery_cuda
else:
    raise NotImplementedError('torch_knnquery only supported with CUDA')


class VoxelGrid(object):
    r""" Defines a Voxel Grid object in which points can be inserted and then queried.
    Args:
        voxel_grid_size (Tuple[int]): Tuple of size 3, specifying x, y, z voxel grid dimensions.
        num_samples_per_ray (int): Number of samples per ray.
        max_o (int): Maximum of non-empty voxels stored at each frustrum
        P (int): Maximum of points stored at each block

    """
    def __init__(self,
                 voxel_grid_size: Tuple[int],
                 num_samples_per_ray: int,
                 vscale: Tuple[float],
                 vsize: Tuple[float],
                 kernel_size: Tuple[int],
                 query_size: Tuple[int],
                 radius_limit_scale: float,
                 depth_limit_scale: float,
                 P: int,
                 max_o: float,
                 ranges: Tuple[int] = None,
                 ):
        self.voxel_grid_size = voxel_grid_size
        self.num_samples_per_ray = num_samples_per_ray
        self.vscale = vscale
        self.vsize = vsize
        self.scaled_vsize = vscale * vscale
        self.scaled_vsize_tensor = torch.Tensor(self.scaled_vsize).cuda()
        self.kernel_size = kernel_size
        self.kernel_size_tensor = torch.Tensor(kernel_size).cuda()
        self.query_size = query_size
        self.query_size_tensor = torch.Tensor(query_size).cuda()
        self.radius_limit_scale = radius_limit_scale
        self.depth_limit_scale = depth_limit_scale
        self.P = P
        self.max_o = max_o
        self.ranges = ranges
        self.kMaxThreadsPerBlock = 1024

        # Get CUDA kernel functions
        self.claim_occ = getattr(knnquery_cuda, 'claim_occ')
        self.map_coor2occ = getattr(knnquery_cuda, 'map_coor2occ')
        self.fill_occ2pnts = getattr(knnquery_cuda, 'fill_occ2pnts')
        self.mask_raypos = getattr(knnquery_cuda, 'mask_raypos')
        self.get_shadingloc = getattr(knnquery_cuda, 'get_shadingloc')
        self.query_along_ray = getattr(knnquery_cuda, 'query_along_ray_layered')
        

    def set_pointset(self, points, actual_num_points_per_batch):
        assert points.is_cuda
        self.points = points
        min_xyz, max_xyz = torch.min(points, dim=-2)[0][0], torch.max(points, dim=-2)[0][0]
        self.B, self.N = points.shape[0], points.shape[1]
        if self.ranges is not None:
            # print("min_xyz", min_xyz.shape)
            # print("max_xyz", max_xyz.shape)
            # print("ranges", ranges)
            min_xyz = torch.max(torch.stack([min_xyz, self.ranges[:3]], dim=0), dim=0)[0], 
            max_xyz = torch.min(torch.stack([max_xyz, self.ranges[3:]], dim=0), dim=0)[0]
        min_xyz = min_xyz - self.scaled_vsize * self.kernel_size / 2
        max_xyz = max_xyz + self.scaled_vsize * self.kernel_size / 2

        self.ranges = torch.cat([min_xyz, max_xyz], dim=-1)
        # print("ranges_np",ranges_np)
        vdim_np = (max_xyz - min_xyz) / self.vsize

        self.scaled_vdim_tensor = torch.ceil(vdim_np / self.vscale).type(torch.int32)
        self.scaled_vdim = self.scaled_vdim_tensor.numpy()

        self.radius_limit = self.radius_limit_scale * max(self.vsize[0], self.vsize[1]), 
        self.depth_limit = self.depth_limit_scale * self.vsize[2]
        
        self.pixel_size = self.scaled_vdim[0] * self.scaled_vdim[1]
        self.grid_size_vol = self.pixel_size * self.scaled_vdim[2]
        self.d_coord_shift = self.ranges[:3]
        self.d_coord_shift_tensor = torch.Tensor(self.d_coord_shift).cuda()


        device = points.device
        self.coor_occ_tensor = torch.zeros([self.B, self.scaled_vdim[0], self.scaled_vdim[1], self.scaled_vdim[2]], dtype=torch.int32, device=device)
        self.occ_2_pnts_tensor = torch.full([self.B, self.max_o, self.P], -1, dtype=torch.int32, device=device)
        self.occ_2_coor_tensor = torch.full([self.B, self.max_o, 3], -1, dtype=torch.int32, device=device)
        self.occ_numpnts_tensor = torch.zeros([self.B, self.max_o], dtype=torch.int32, device=device)
        self.coor_2_occ_tensor = torch.full([self.B, self.scaled_vdim[0], self.scaled_vdim[1], self.scaled_vdim[2]], -1, dtype=torch.int32, device=device)
        occ_idx_tensor = torch.zeros([self.B], dtype=torch.int32, device=device)
        seconds = time.time()

        self.claim_occ(
            points,
            actual_num_points_per_batch,
            self.B,
            self.N,
            self.d_coord_shift_tensor,
            self.scaled_vsize_tensor,
            self.scaled_vdim_tensor,
            self.grid_size_vol,
            self.max_o,
            occ_idx_tensor,
            self.coor_2_occ_tensor,
            self.occ_2_coor_tensor,
            seconds
            )
        # torch.cuda.synchronize()
        self.coor_2_occ_tensor = torch.full([self.B, self.scaled_vdim[0], self.scaled_vdim[1], self.scaled_vdim[2]], -1,
                                       dtype=torch.int32, device=device)

        self.map_coor2occ(
            self.B,
            self.scaled_vdim_tensor,
            self.kernel_size_tensor,
            self.grid_size_vol,
            self.max_o,
            occ_idx_tensor,
            self.coor_occ_tensor,
            self.coor_2_occ_tensor,
            self.occ_2_coor_tensor
            )
        # torch.cuda.synchronize()
        seconds = time.time()

        self.fill_occ2pnts(
            points,
            actual_num_points_per_batch,
            self.B,
            self.N,
            self.P,
            self.d_coord_shift_tensor,
            self.scaled_vsize_tensor,
            self.scaled_vdim_tensor,
            self.grid_size_vol,
            self.max_o,
            self.coor_2_occ_tensor,
            self.occ_2_pnts_tensor,
            self.occ_numpnts_tensor,
            seconds
            )


    def query(self, raypos_tensor, num_neighbors, samples_per_ray): 
        device = raypos_tensor.device
        R, D = raypos_tensor.size(1), raypos_tensor.size(2)

        
        # torch.cuda.synchronize()
        # print("coor_occ_tensor", torch.min(coor_occ_tensor), torch.max(coor_occ_tensor), torch.min(occ_2_coor_tensor), torch.max(occ_2_coor_tensor), torch.min(coor_2_occ_tensor), torch.max(coor_2_occ_tensor), torch.min(occ_idx_tensor), torch.max(occ_idx_tensor), torch.min(occ_numpnts_tensor), torch.max(occ_numpnts_tensor), torch.min(occ_2_pnts_tensor), torch.max(occ_2_pnts_tensor), occ_2_pnts_tensor.shape)
        # print("occ_numpnts_tensor", torch.sum(occ_numpnts_tensor > 0), ranges_np)
        # vis_vox(ranges_np, scaled_vsize_np, coor_2_occ_tensor)

        raypos_mask_tensor = torch.zeros([self.B, R, D], dtype=torch.int32, device=device)

        self.mask_raypos(
            raypos_tensor,  # [1, 2048, 400, 3]
            self.coor_occ_tensor,  # [1, 2048, 400, 3]
            self.B,
            R,
            D,
            self.grid_size_vol,
            self.d_coord_shift_tensor,
            self.scaled_vdim,
            self.scaled_vsize,
            raypos_mask_tensor
        )

        # torch.cuda.synchronize()
        # print("raypos_mask_tensor", raypos_mask_tensor.shape, torch.sum(coor_occ_tensor), torch.sum(raypos_mask_tensor))
        # save_points(raypos_tensor.reshape(-1, 3), "./", "rawraypos_pnts")
        # raypos_masked = torch.masked_select(raypos_tensor, raypos_mask_tensor[..., None] > 0)
        # save_points(raypos_masked.reshape(-1, 3), "./", "raypos_pnts")

        ray_mask_tensor = torch.max(raypos_mask_tensor, dim=-1)[0] > 0 # B, R
        R = torch.max(torch.sum(ray_mask_tensor.to(torch.int32))).cpu().numpy()
        sample_loc_tensor = torch.zeros([self.B, R, samples_per_ray, 3], dtype=torch.float32, device=device)
        sample_pidx_tensor = torch.full([self.B, R, samples_per_ray, K], -1, dtype=torch.int32, device=device)
        if R > 0:
            raypos_tensor = torch.masked_select(raypos_tensor, ray_mask_tensor[..., None, None].expand(-1, -1, D, 3)).reshape(self.B, R, D, 3)
            raypos_mask_tensor = torch.masked_select(raypos_mask_tensor, ray_mask_tensor[..., None].expand(-1, -1, D)).reshape(self.B, R, D)
            # print("R", R, raypos_tensor.shape, raypos_mask_tensor.shape)

            raypos_maskcum = torch.cumsum(raypos_mask_tensor, dim=-1).to(torch.int32)
            raypos_mask_tensor = (raypos_mask_tensor * raypos_maskcum * (raypos_maskcum <= SR)) - 1
            sample_loc_mask_tensor = torch.zeros([self.B, R, samples_per_ray], dtype=torch.int32, device=device)
            self.get_shadingloc(
                raypos_tensor,  # [1, 2048, 400, 3]
                raypos_mask_tensor,
                self.B,
                R,
                D,
                samples_per_ray,
                sample_loc_tensor,
                sample_loc_mask_tensor,
                block=(self.kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1)
            )

            # torch.cuda.synchronize()
            # print("shadingloc_mask_tensor", torch.sum(sample_loc_mask_tensor, dim=-1), torch.sum(torch.sum(sample_loc_mask_tensor, dim=-1) > 0), torch.sum(sample_loc_mask_tensor > 0))
            # shadingloc_masked = torch.masked_select(sample_loc_tensor, sample_loc_mask_tensor[..., None] > 0)
            # save_points(shadingloc_masked.reshape(-1, 3), "./", "shading_pnts{}".format(self.count))

            seconds = time.time()
            gridSize = int((self.B * R * samples_per_ray + self.kMaxThreadsPerBlock - 1) / self.kMaxThreadsPerBlock)
            self.query_along_ray(
                self.points,
                self.B,
                samples_per_ray,
                R,
                self.max_o,
                self.P,
                num_neighbors,
                self.grid_size_vol,
                self.radius_limit ** 2,
                self.d_coord_shift_tensor,
                self.scaled_vdim_tensor,
                self.scaled_vsize_tensor,
                self.kernel_size_tensor,
                self.occ_numpnts_tensor,
                self.occ_2_pnts_tensor,
                self.coor_2_occ_tensor,
                sample_loc_tensor,
                sample_loc_mask_tensor,
                sample_pidx_tensor,
                block=(self.kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1))
            # torch.cuda.synchronize()
            # print("point_xyz_w_tensor",point_xyz_w_tensor.shape)
            # queried_masked = point_xyz_w_tensor[0][sample_pidx_tensor.reshape(-1).to(torch.int64), :]
            # save_points(queried_masked.reshape(-1, 3), "./", "queried_pnts{}".format(self.count))
            # print("valid ray",  torch.sum(torch.sum(sample_loc_mask_tensor, dim=-1) > 0))
            #
            masked_valid_ray = torch.sum(sample_pidx_tensor.view(self.B, R, -1) >= 0, dim=-1) > 0
            R = torch.max(torch.sum(masked_valid_ray.to(torch.int32), dim=-1)).cpu().numpy()
            ray_mask_tensor.masked_scatter_(ray_mask_tensor, masked_valid_ray)
            sample_pidx_tensor = torch.masked_select(sample_pidx_tensor, masked_valid_ray[..., None, None].expand(-1, -1, samples_per_ray, num_neighbors))
            sample_pidx_tensor = sample_pidx_tensor.reshape(self.B, R, samples_per_ray, num_neighbors)
            sample_loc_tensor = torch.masked_select(sample_loc_tensor, masked_valid_ray[..., None, None].expand(-1, -1, samples_per_ray, 3)).reshape(self.B, R, samples_per_ray, 3)
        # self.count+=1
        return sample_pidx_tensor, sample_loc_tensor, ray_mask_tensor.to(torch.int8)
