import torch
from typing import Tuple

if torch.cuda.is_available():
    import knnquery_cuda
else:
    raise NotImplementedError('torch_knnquery only supported with CUDA')


class VoxelGrid(object):
    r""" Computes eigenvectors and eigenvalues of symmetric 3x3 matrices in a batch.
    Args:
        voxel_grid_size (Tuple[int]): Tuple of size 3, specifying x, y, z voxel grid dimensions.
        num_samples_per_ray (int): Number of samples per ray.

    """
    def __init__(self,
                 voxel_grid_size: Tuple[int],
                 num_samples_per_ray: int,
                 ):
        self.voxel_grid_size = voxel_grid_size
        self.num_samples_per_ray = num_samples_per_ray
        self.claim_occ = getattr(knnquery_cuda, 'claim_occ')
        self.map_coor2occ = getattr(knnquery_cuda, 'map_coor2occ')
        self.fill_occ2pnts = getattr(knnquery_cuda, 'fill_occ2pnts')
        self.mask_raypos = getattr(knnquery_cuda, 'mask_raypos')
        self.get_shadingloc = getattr(knnquery_cuda, 'get_shadingloc')
        self.query_along_ray = getattr(knnquery_cuda, 'query_along_ray_layered')
        
    def insert_points(self, points):
        device = points.device
        B = points.size(0)
        coor_occ_tensor = torch.zeros([B, scaled_vdim_np[0], scaled_vdim_np[1], scaled_vdim_np[2]], dtype=torch.int32, device=device)
        occ_2_pnts_tensor = torch.full([B, max_o, P], -1, dtype=torch.int32, device=device)
        occ_2_coor_tensor = torch.full([B, max_o, 3], -1, dtype=torch.int32, device=device)
        occ_numpnts_tensor = torch.zeros([B, max_o], dtype=torch.int32, device=device)
        coor_2_occ_tensor = torch.full([B, scaled_vdim_np[0], scaled_vdim_np[1], scaled_vdim_np[2]], -1, dtype=torch.int32, device=device)
        occ_idx_tensor = torch.zeros([B], dtype=torch.int32, device=device)
        seconds = time.time()

        self.claim_occ(
            Holder(point_xyz_w_tensor),
            Holder(actual_numpoints_tensor),
            np.int32(B),
            np.int32(N),
            d_coord_shift,
            scaled_vsize_gpu,
            scaled_vdim_gpu,
            np.int32(grid_size_vol),
            np.int32(max_o),
            Holder(occ_idx_tensor),
            Holder(coor_2_occ_tensor),
            Holder(occ_2_coor_tensor),
            np.uint64(seconds),
            block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1))
        # torch.cuda.synchronize()
        coor_2_occ_tensor = torch.full([B, scaled_vdim_np[0], scaled_vdim_np[1], scaled_vdim_np[2]], -1,
                                       dtype=torch.int32, device=device)
        gridSize = int((B * max_o + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        self.map_coor2occ(
            np.int32(B),
            scaled_vdim_gpu,
            kernel_size_gpu,
            np.int32(grid_size_vol),
            np.int32(max_o),
            Holder(occ_idx_tensor),
            Holder(coor_occ_tensor),
            Holder(coor_2_occ_tensor),
            Holder(occ_2_coor_tensor),
            block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1))
        # torch.cuda.synchronize()
        seconds = time.time()

        gridSize = int((B * N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        self.fill_occ2pnts(
            Holder(point_xyz_w_tensor),
            Holder(actual_numpoints_tensor),
            np.int32(B),
            np.int32(N),
            np.int32(P),
            d_coord_shift,
            scaled_vsize_gpu,
            scaled_vdim_gpu,
            np.int32(grid_size_vol),
            np.int32(max_o),
            Holder(coor_2_occ_tensor),
            Holder(occ_2_pnts_tensor),
            Holder(occ_numpnts_tensor),
            np.uint64(seconds),
            block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1))

        indices = getattr(knnquery_cuda, 'insert_points')(points)
        self.current_point_indices = indices



    def query(self, points): 
        device = point_xyz_w_tensor.device
        B, N = point_xyz_w_tensor.shape[0], point_xyz_w_tensor.shape[1]
        pixel_size = scaled_vdim_np[0] * scaled_vdim_np[1]
        grid_size_vol = pixel_size * scaled_vdim_np[2]
        d_coord_shift = ranges_gpu[:3]
        R, D = raypos_tensor.shape[1], raypos_tensor.shape[2]
        R = pixel_idx_tensor.reshape(B, -1, 2).shape[1]
        gridSize = int((B * N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)

        coor_occ_tensor, occ_2_coor_tensor, coor_2_occ_tensor, occ_idx_tensor, occ_numpnts_tensor, occ_2_pnts_tensor = self.build_occ_vox(point_xyz_w_tensor, actual_numpoints_tensor, B, N, P, max_o, scaled_vdim_np, kMaxThreadsPerBlock, gridSize, scaled_vsize_gpu, scaled_vdim_gpu, query_size_gpu, grid_size_vol, d_coord_shift)

        # torch.cuda.synchronize()
        # print("coor_occ_tensor", torch.min(coor_occ_tensor), torch.max(coor_occ_tensor), torch.min(occ_2_coor_tensor), torch.max(occ_2_coor_tensor), torch.min(coor_2_occ_tensor), torch.max(coor_2_occ_tensor), torch.min(occ_idx_tensor), torch.max(occ_idx_tensor), torch.min(occ_numpnts_tensor), torch.max(occ_numpnts_tensor), torch.min(occ_2_pnts_tensor), torch.max(occ_2_pnts_tensor), occ_2_pnts_tensor.shape)
        # print("occ_numpnts_tensor", torch.sum(occ_numpnts_tensor > 0), ranges_np)
        # vis_vox(ranges_np, scaled_vsize_np, coor_2_occ_tensor)

        raypos_mask_tensor = torch.zeros([B, R, D], dtype=torch.int32, device=device)
        gridSize = int((B * R * D + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        self.mask_raypos(
            Holder(raypos_tensor),  # [1, 2048, 400, 3]
            Holder(coor_occ_tensor),  # [1, 2048, 400, 3]
            np.int32(B),
            np.int32(R),
            np.int32(D),
            np.int32(grid_size_vol),
            d_coord_shift,
            scaled_vdim_gpu,
            scaled_vsize_gpu,
            Holder(raypos_mask_tensor),
            block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1)
        )

        # torch.cuda.synchronize()
        # print("raypos_mask_tensor", raypos_mask_tensor.shape, torch.sum(coor_occ_tensor), torch.sum(raypos_mask_tensor))
        # save_points(raypos_tensor.reshape(-1, 3), "./", "rawraypos_pnts")
        # raypos_masked = torch.masked_select(raypos_tensor, raypos_mask_tensor[..., None] > 0)
        # save_points(raypos_masked.reshape(-1, 3), "./", "raypos_pnts")

        ray_mask_tensor = torch.max(raypos_mask_tensor, dim=-1)[0] > 0 # B, R
        R = torch.max(torch.sum(ray_mask_tensor.to(torch.int32))).cpu().numpy()
        sample_loc_tensor = torch.zeros([B, R, SR, 3], dtype=torch.float32, device=device)
        sample_pidx_tensor = torch.full([B, R, SR, K], -1, dtype=torch.int32, device=device)
        if R > 0:
            raypos_tensor = torch.masked_select(raypos_tensor, ray_mask_tensor[..., None, None].expand(-1, -1, D, 3)).reshape(B, R, D, 3)
            raypos_mask_tensor = torch.masked_select(raypos_mask_tensor, ray_mask_tensor[..., None].expand(-1, -1, D)).reshape(B, R, D)
            # print("R", R, raypos_tensor.shape, raypos_mask_tensor.shape)

            raypos_maskcum = torch.cumsum(raypos_mask_tensor, dim=-1).to(torch.int32)
            raypos_mask_tensor = (raypos_mask_tensor * raypos_maskcum * (raypos_maskcum <= SR)) - 1
            sample_loc_mask_tensor = torch.zeros([B, R, SR], dtype=torch.int32, device=device)
            self.get_shadingloc(
                Holder(raypos_tensor),  # [1, 2048, 400, 3]
                Holder(raypos_mask_tensor),
                np.int32(B),
                np.int32(R),
                np.int32(D),
                np.int32(SR),
                Holder(sample_loc_tensor),
                Holder(sample_loc_mask_tensor),
                block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1)
            )

            # torch.cuda.synchronize()
            # print("shadingloc_mask_tensor", torch.sum(sample_loc_mask_tensor, dim=-1), torch.sum(torch.sum(sample_loc_mask_tensor, dim=-1) > 0), torch.sum(sample_loc_mask_tensor > 0))
            # shadingloc_masked = torch.masked_select(sample_loc_tensor, sample_loc_mask_tensor[..., None] > 0)
            # save_points(shadingloc_masked.reshape(-1, 3), "./", "shading_pnts{}".format(self.count))

            seconds = time.time()
            gridSize = int((B * R * SR + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
            self.query_along_ray(
                Holder(point_xyz_w_tensor),
                np.int32(B),
                np.int32(SR),
                np.int32(R),
                np.int32(max_o),
                np.int32(P),
                np.int32(K),
                np.int32(grid_size_vol),
                np.float32(radius_limit_np ** 2),
                d_coord_shift,
                scaled_vdim_gpu,
                scaled_vsize_gpu,
                kernel_size_gpu,
                Holder(occ_numpnts_tensor),
                Holder(occ_2_pnts_tensor),
                Holder(coor_2_occ_tensor),
                Holder(sample_loc_tensor),
                Holder(sample_loc_mask_tensor),
                Holder(sample_pidx_tensor),
                np.uint64(seconds),
                np.int32(self.opt.NN),
                block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1))
            # torch.cuda.synchronize()
            # print("point_xyz_w_tensor",point_xyz_w_tensor.shape)
            # queried_masked = point_xyz_w_tensor[0][sample_pidx_tensor.reshape(-1).to(torch.int64), :]
            # save_points(queried_masked.reshape(-1, 3), "./", "queried_pnts{}".format(self.count))
            # print("valid ray",  torch.sum(torch.sum(sample_loc_mask_tensor, dim=-1) > 0))
            #
            masked_valid_ray = torch.sum(sample_pidx_tensor.view(B, R, -1) >= 0, dim=-1) > 0
            R = torch.max(torch.sum(masked_valid_ray.to(torch.int32), dim=-1)).cpu().numpy()
            ray_mask_tensor.masked_scatter_(ray_mask_tensor, masked_valid_ray)
            sample_pidx_tensor = torch.masked_select(sample_pidx_tensor, masked_valid_ray[..., None, None].expand(-1, -1, SR, K)).reshape(B, R, SR, K)
            sample_loc_tensor = torch.masked_select(sample_loc_tensor, masked_valid_ray[..., None, None].expand(-1, -1, SR, 3)).reshape(B, R, SR, 3)
        # self.count+=1
        return sample_pidx_tensor, sample_loc_tensor, ray_mask_tensor.to(torch.int8)
