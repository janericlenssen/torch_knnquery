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
        
    def insert_points(self, points):
        indices = getattr(knnquery_cuda, 'insert_points')(points)
        self.current_point_indices = indices



