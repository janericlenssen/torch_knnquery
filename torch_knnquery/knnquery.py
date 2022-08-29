import torch
from typing import Tuple, Optional
import time

if torch.cuda.is_available():
    import knnquery_cuda
else:
    raise NotImplementedError('torch_knnquery only supported with CUDA')


class VoxelGrid(object):
    r""" Defines a Voxel Grid object in which points can be inserted and then queried.
    Args:
        vsize (Tuple[float]): Tuple of size 3, specifying x, y, z voxel grid dimensions.
        vscale (Tuple[float]): Tuple of size 3, specifying x, y, z voxel scales.
        kernel_size (Tuple[int]): Size of voxel kernel to consider for nearest neighbor queries
        max_points_per_voxel (int): Maximum number of points in each voxel
        max_occ_voxels_per_example (float): Maximum of occupied voxels for each example
        ranges (Tuple[int], optional): 

    """
    def __init__(
        self,
        voxel_size: Tuple[float],
        voxel_scale: Tuple[float],
        kernel_size: Tuple[int],
        max_points_per_voxel: int,
        max_occ_voxels_per_example: float,
        ranges: Optional[Tuple[int]] = None,
        ):

        self.vscale = torch.Tensor(voxel_scale).to(torch.float32).cuda()
        self.vsize = torch.Tensor(voxel_size).to(torch.float32).cuda()

        self.scaled_vsize = (self.vsize * self.vscale).cuda()
        self.kernel_size = torch.Tensor(kernel_size).cuda()
        self.P = max_points_per_voxel
        self.max_o = max_occ_voxels_per_example
        self.ranges = torch.Tensor(ranges).to(torch.float32).cuda()
        self.kMaxThreadsPerBlock = 1024

        # Get C++ functions
        self.find_occupied_voxels = getattr(knnquery_cuda, 'find_occupied_voxels')
        self.create_coor_occ_maps = getattr(knnquery_cuda, 'create_coor_occ_maps')
        self.assign_points_to_occ_voxels = getattr(knnquery_cuda, 'assign_points_to_occ_voxels')
        self.create_raypos_mask = getattr(knnquery_cuda, 'create_raypos_mask')
        self.get_shadingloc = getattr(knnquery_cuda, 'get_shadingloc')
        self.query_along_ray = getattr(knnquery_cuda, 'query_along_ray')
        

    def set_pointset(
        self, 
        points, 
        actual_num_points_per_example
        ):
        r""" Inserts a set of point clouds into the VoxelGrid.
        Args:
            points (torch.Tensor): Tensor of size [B, max_num_points, 3] containing B point clouds.
            actual_num_points_per_example (torch.Tensor): Tensor of size [B] containing the 
                actual num_points<=max_num_points for each point cloud.

        """
        assert points.is_cuda
        self.points = points
        min_xyz, max_xyz = torch.min(points, dim=-2)[0][0], torch.max(points, dim=-2)[0][0]
        self.B, self.N = points.shape[0], points.shape[1]
        if self.ranges is not None:
            # print("min_xyz", min_xyz.shape)
            # print("max_xyz", max_xyz.shape)
            # print("ranges", ranges)
            min_xyz = torch.max(torch.stack([min_xyz, self.ranges[:3]], dim=0), dim=0)[0]
            max_xyz = torch.min(torch.stack([max_xyz, self.ranges[3:]], dim=0), dim=0)[0]

        min_xyz = min_xyz - self.scaled_vsize * self.kernel_size / 2
        max_xyz = max_xyz + self.scaled_vsize * self.kernel_size / 2

        self.ranges = torch.cat([min_xyz, max_xyz], dim=-1)
        # print("ranges_np",ranges_np)
        vdim_np = (max_xyz - min_xyz) / self.vsize

        self.scaled_vdim = torch.ceil(vdim_np / self.vscale).type(torch.int32)

        
        self.pixel_size = self.scaled_vdim[0] * self.scaled_vdim[1]
        self.grid_size_vol = self.pixel_size * self.scaled_vdim[2]
        self.d_coord_shift = self.ranges[:3]


        device = points.device
        self.coor_occ_tensor = torch.zeros([self.B, self.scaled_vdim[0], self.scaled_vdim[1], self.scaled_vdim[2]], dtype=torch.int32, device=device)
        self.occ_2_pnts_tensor = torch.full([self.B, self.max_o, self.P], -1, dtype=torch.int32, device=device)
        self.occ_2_coor_tensor = torch.full([self.B, self.max_o, 3], -1, dtype=torch.int32, device=device)
        self.occ_numpnts_tensor = torch.zeros([self.B, self.max_o], dtype=torch.int32, device=device)
        self.coor_2_occ_tensor = torch.full([self.B, self.scaled_vdim[0], self.scaled_vdim[1], self.scaled_vdim[2]], -1, dtype=torch.int32, device=device)
        occ_idx_tensor = torch.zeros([self.B], dtype=torch.int32, device=device)
        seconds = time.time()

        # Find the set of voxels that get occupied by the given point set.
        # Outputs:
        # occ_idx_tensor: for each example in the batch holds the number of occupied voxels in occ_2_coor_tensor
        # coor_2_occ_tensor: for each voxel in 3D grid, 0 of voxel is occupied, -1 otherwise
        # occ_2_corr_tensor: one entry for each occupied voxel, contains 3D coordinates of voxel in voxel grid
        self.find_occupied_voxels(
            points,
            actual_num_points_per_example,
            self.B,
            self.N,
            self.d_coord_shift,
            self.scaled_vsize,
            self.scaled_vdim,
            self.grid_size_vol,
            self.max_o,
            occ_idx_tensor,
            self.coor_2_occ_tensor,
            self.occ_2_coor_tensor,
            seconds
            )

        self.coor_2_occ_tensor = torch.full([self.B, self.scaled_vdim[0], self.scaled_vdim[1], self.scaled_vdim[2]], -1,
                                       dtype=torch.int32, device=device)

        # For each occupied voxel in the 3D grid, occupies voxels within kernel_size around it
        # For each occupied voxel stores the id of voxel in occ_2_corr_tensor
        # Outputs:
        # coor_occ_tensor: or each voxel in 3D grid, 1 of voxel is occupied, 0 otherwise
        # coor_2_occ_tensor: has the index of voxel in occ_2_corr_tensor
        self.create_coor_occ_maps(
            self.B,
            self.scaled_vdim,
            self.kernel_size,
            self.grid_size_vol,
            self.max_o,
            occ_idx_tensor,
            self.coor_occ_tensor,
            self.coor_2_occ_tensor,
            self.occ_2_coor_tensor
            )
        # torch.cuda.synchronize()
        seconds = time.time()

        # Assigns each point to an occupied voxel
        # Outputs:
        # occ_2_pnts_tensor: For each occupied voxel, stores the indices of points assigned to this voxel
        # occ_numpnts_tensor: For each occupied voxel, stores the number of points assigned to this voxel
        self.assign_points_to_occ_voxels(
            points,
            actual_num_points_per_example,
            self.B,
            self.N,
            self.P,
            self.d_coord_shift,
            self.scaled_vsize,
            self.scaled_vdim,
            self.grid_size_vol,
            self.max_o,
            self.coor_2_occ_tensor,
            self.occ_2_pnts_tensor,
            self.occ_numpnts_tensor,
            seconds
            )


    def query(self, 
        raypos: torch.Tensor,
        k: int, 
        radius_limit_scale: float,
        max_shading_points_per_ray: Optional[int] = 24
        ): 
        r""" Find the k-nearest neighbors of ray samples from each point cloud
        Args:
            raypos (torch.Tensor): Tensor of size [num_rays, num_samples_per_ray, 3] containing query positions.
            k (int): number of nearest neighbors to sample.
            radius_limit_scale (float): radius limit in which to search for nearest neighbors.
            max_shading_points_per_ray (int, optional): The maximum number of points per ray for which neighbors are sampled.
                The first max_shading_points_per_ray samples of each ray that hit occupied voxels return neighbors.
        """
        device = raypos.device
        R, D = raypos.size(1), raypos.size(2)
        assert k <= 20, "k cannot be greater than 20"

        raypos_mask_tensor = torch.zeros([self.B, R, D], dtype=torch.int32, device=device)

        # Check which query positions actually hit occupied voxels.
        # Output: 
        # # raypos_mask_tensor contains binary indicators for each query position
        self.create_raypos_mask(
            raypos,  # [1, 2048, 400, 3]
            self.coor_occ_tensor,  # [1, 2048, 400, 3]
            self.B,
            R,
            D,
            self.grid_size_vol,
            self.d_coord_shift,
            self.scaled_vdim,
            self.scaled_vsize,
            raypos_mask_tensor
        )


        ray_mask_tensor = torch.max(raypos_mask_tensor, dim=-1)[0] > 0 # B, R
        R = torch.max(torch.sum(ray_mask_tensor.to(torch.int32))).cpu().numpy()
        sample_loc_tensor = torch.zeros([self.B, R, max_shading_points_per_ray, 3], dtype=torch.float32, device=device)
        sample_pidx_tensor = torch.full([self.B, R, max_shading_points_per_ray, k], -1, dtype=torch.int32, device=device)
        if R > 0:
            raypos_tensor = torch.masked_select(raypos_tensor, ray_mask_tensor[..., None, None].expand(-1, -1, D, 3)).reshape(self.B, R, D, 3)
            raypos_mask_tensor = torch.masked_select(raypos_mask_tensor, ray_mask_tensor[..., None].expand(-1, -1, D)).reshape(self.B, R, D)
            # print("R", R, raypos_tensor.shape, raypos_mask_tensor.shape)

            raypos_maskcum = torch.cumsum(raypos_mask_tensor, dim=-1).to(torch.int32)
            raypos_mask_tensor = (raypos_mask_tensor * raypos_maskcum * (raypos_maskcum <= max_shading_points_per_ray)) - 1
            sample_loc_mask_tensor = torch.zeros([self.B, R, max_shading_points_per_ray], dtype=torch.int32, device=device)

            # For each ray query that hits occupied voxels, 
            # determine the maximally max_shading_points_per_ray many shading points
            # Output: 
            # sample_loc_tensor contains the actual ray queries for which neighbors should be found
            # sample_loc_mask_tensor contains 1 if the same index in sample_loc_tensor contains a valid sample point
            self.get_shadingloc(
                raypos_tensor,  # [1, 2048, 400, 3]
                raypos_mask_tensor,
                self.B,
                R,
                D,
                max_shading_points_per_ray,
                sample_loc_tensor,
                sample_loc_mask_tensor
            )

            # Performs the actual knn queries for all points in sample_loc_tensor
            # Output: 
            # sample_pidx_tensor: for each entry in sample_loc_tensor, contains the indices of found neighbors
            radius_limit = radius_limit_scale * max(self.vsize[0], self.vsize[1]), 
            self.query_along_ray(
                self.points,
                self.B,
                max_shading_points_per_ray,
                R,
                self.max_o,
                self.P,
                k,
                self.grid_size_vol,
                radius_limit ** 2,
                self.d_coord_shift,
                self.scaled_vdim,
                self.scaled_vsize,
                self.kernel_size,
                self.occ_numpnts_tensor,
                self.occ_2_pnts_tensor,
                self.coor_2_occ_tensor,
                sample_loc_tensor,
                sample_loc_mask_tensor,
                sample_pidx_tensor
                )
            
            masked_valid_ray = torch.sum(sample_pidx_tensor.view(self.B, R, -1) >= 0, dim=-1) > 0
            R = torch.max(torch.sum(masked_valid_ray.to(torch.int32), dim=-1)).cpu().numpy()
            ray_mask_tensor.masked_scatter_(ray_mask_tensor, masked_valid_ray)
            sample_pidx_tensor = torch.masked_select(sample_pidx_tensor, masked_valid_ray[..., None, None].expand(-1, -1, max_shading_points_per_ray, k))
            sample_pidx_tensor = sample_pidx_tensor.reshape(self.B, R, max_shading_points_per_ray, k)
            sample_loc_tensor = torch.masked_select(sample_loc_tensor, masked_valid_ray[..., None, None].expand(-1, -1, max_shading_points_per_ray, 3)).reshape(self.B, R, max_shading_points_per_ray, 3)
        # self.count+=1
        return sample_pidx_tensor, sample_loc_tensor, ray_mask_tensor.to(torch.int8)
