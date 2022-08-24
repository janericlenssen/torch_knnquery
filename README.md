# Voxel grid k-nearest-neighbor queries in PyTorch

This is an inoffical implementation of the voxel grid datastructure for k-nearest-neighbor queries presented in [Xu et al. Point-NeRF: Point-based Neural Radiance Fields](https://arxiv.org/abs/2201.08845).

In comparison to the original implementation, it is:
- standalone
- implemented as a PyTorch module instead of using pycuda


## Usage

Create an empty `VoxelGrid`:

```python
import torch_knnquery

voxel_grid = VoxelGrid(
    voxel_size=(0.8, 0.8, 0.8),                 # Voxel size for each dimension
    voxel_scale=(2, 2, 2),                      # Voxel scale for each dimension
    kernel_size=(3, 3, 3),                      # Range of voxels searched for neighbors (default: [3, 3, 3])
    max_points_per_voxel=26,                    # Maximum number of points stored in a voxel (default: 26)
    max_occ_voxels_per_pointcloud=610000,       # Maximum number of occupied voxels per point cloud (default: 600000)
    ranges=(-10.0,-10.0,-10.0,10.0,10.0,10.0)   # Maximum ranges the VoxelGrid spans (default: inferred from data)
)

```

Insert a set of point clouds into the `VoxelGrid`:
```python
voxel_grid.set_pointset(
        points=points_tensor,                       # Tensor of size [B, max_num_points, 3] containing B point clouds.
        actual_num_points_per_example=num_tensor    # Tensor of size [B] containing the 
                                                    # actual num_points<=max_num_points for each point cloud.
        )

```

Query k-nearest neighbors for ray samples through the `VoxelGrid`:
```python
voxel_grid.query(self, 
        raypos=raypos_tensor,           # Tensor of size [num_rays, num_samples_per_ray, 3] containing query positions.
        k=8,                            # Number of neighbors to sample for each ray sample 
        radius_limit_scale=4,           # Maximum radius to search for neighbors in
        max_shading_points_per_ray=24   # The maximum number of points per ray for which neighbors are sampled.
                                        # The first max_shading_points_per_ray samples of each ray that hit 
                                        # occupied voxels return neighbors.
        )
```
