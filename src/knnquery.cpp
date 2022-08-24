#include <torch/extension.h>

#define IS_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor");
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");
#define CHECK_INPUT(x) IS_CUDA(x) IS_CONTIGUOUS(x)

void claim_occ(
  at::Tensor points,
  at::Tensor actual_num_points_per_batch,
  size_t B,
  size_t N,
  at::Tensor d_coord_shift,
  at::Tensor scaled_vsize,
  scalar_t grid_size_vol,
  scalar_t max_o,
  at::Tensor coor_2_occ_tensor,
  at::Tensor occ_2_coor_tensor,
  unsigned long seconds
);

void claim_occ(
  at::Tensor points,
  at::Tensor actual_num_points_per_batch,
  size_t B,
  size_t N,
  at::Tensor d_coord_shift,
  at::Tensor scaled_vsize,
  scalar_t grid_size_vol,
  scalar_t max_o,
  at::Tensor coor_2_occ_tensor,
  at::Tensor occ_2_coor_tensor,
  unsigned long seconds
) {
  CHECK_INPUT(points);
  CHECK_INPUT(actual_num_points_per_batch);
  CHECK_INPUT(d_coord_shift);
  CHECK_INPUT(scaled_vsize);
  CHECK_INPUT(coor_2_occ_tensor);
  CHECK_INPUT(occ_2_coor_tensor);
  return claim_occ_kernel(
    points,
    actual_num_points_per_batch,
    B,
    N,
    d_coord_shift,
    scaled_vsize,
    grid_size_vol,
    max_o,
    coor_2_occ_tensor,
    occ_2_coor_tensor,
    seconds);
}


void map_coor2occ(
  size_t B,
  at::Tensor scaled_vdim,
  at::Tensor kernel_size,
  scalar_t grid_size_vol,
  scalar_t max_o,
  at::Tensor occ_idx_tensor,
  at::Tensor coor_occ_tensor,
  at::Tensor coor_2_occ_tensor,
  at::Tensor occ_2_coor_tensor
);

void map_coor2occ(
  size_t B,
  at::Tensor scaled_vdim,
  at::Tensor kernel_size,
  scalar_t grid_size_vol,
  scalar_t max_o,
  at::Tensor occ_idx_tensor,
  at::Tensor coor_occ_tensor,
  at::Tensor coor_2_occ_tensor,
  at::Tensor occ_2_coor_tensor
) {
  CHECK_INPUT(scaled_vdim);
  CHECK_INPUT(kernel_size);
  CHECK_INPUT(occ_idx_tensor);
  CHECK_INPUT(coor_occ_tensor);
  CHECK_INPUT(coor_2_occ_tensor);
  CHECK_INPUT(occ_2_coor_tensor);
  return map_coor2occ_kernel(
    B,
    scaled_vdim,
    kernel_size,
    grid_size_vol,
    max_o,
    occ_idx_tensor,
    coor_occ_tensor,
    coor_2_occ_tensor,
    occ_2_coor_tensor);
}

void fill_occ2pnts(
  at::Tensor points,
  at::Tensor actual_num_points_per_batch,
  size_t B,
  size_t N,
  size_t P,
  at::Tensor d_coord_shift,
  at::Tensor scaled_vsize,
  at::Tensor scaled_vdim,
  scalar_t grid_size_vol,
  scalar_t max_o,
  at::Tensor coor_2_occ_tensor,
  at::Tensor occ_2_pnts_tensor,
  at::Tensor occ_numpnts_tensor,
  unsigned long seconds
);

void fill_occ2pnts(
  at::Tensor points,
  at::Tensor actual_num_points_per_batch,
  size_t B,
  size_t N,
  size_t P,
  at::Tensor d_coord_shift,
  at::Tensor scaled_vsize,
  at::Tensor scaled_vdim,
  scalar_t grid_size_vol,
  scalar_t max_o,
  at::Tensor coor_2_occ_tensor,
  at::Tensor occ_2_pnts_tensor,
  at::Tensor occ_numpnts_tensor,
  unsigned long seconds
) {
  CHECK_INPUT(points);
  CHECK_INPUT(actual_num_points_per_batch);
  CHECK_INPUT(d_coord_shift);
  CHECK_INPUT(scaled_vsize);
  CHECK_INPUT(scaled_vdim);
  CHECK_INPUT(coor_2_occ_tensor);
  CHECK_INPUT(occ_2_pnts_tensor);
  CHECK_INPUT(occ_numpnts_tensor);
  return fill_occ2pnts_kernel(
    points,
    actual_num_points_per_batch,
    B,
    N,
    P,
    d_coord_shift,
    scaled_vsize,
    scaled_vdim,
    grid_size_vol,
    max_o,
    coor_2_occ_tensor,
    occ_2_pnts_tensor,
    occ_numpnts_tensor,
    seconds
    );
}

void mask_raypos(
  at::Tensor raypos_tensor,
  at::Tensor coor_occ_tensor,
  size_t B,
  size_t R,
  size_t D,
  scalar_t grid_size_vol,
  at::Tensor d_coord_shift,
  at::Tensor scaled_vdim,
  at::Tensor scaled_vsize,
  scalar_t max_o,
  at::Tensor raypos_mask_tensor
);

void mask_raypos(
  at::Tensor raypos_tensor,
  at::Tensor coor_occ_tensor,
  size_t B,
  size_t R,
  size_t D,
  scalar_t grid_size_vol,
  at::Tensor d_coord_shift,
  at::Tensor scaled_vdim,
  at::Tensor scaled_vsize,
  scalar_t max_o,
  at::Tensor raypos_mask_tensor
) {
  CHECK_INPUT(raypos_tensor);
  CHECK_INPUT(coor_occ_tensor);
  CHECK_INPUT(d_coord_shift);
  CHECK_INPUT(scaled_vsize);
  CHECK_INPUT(scaled_vdim);
  CHECK_INPUT(raypos_mask_tensor);
  return mask_raypos_kernel(
    raypos_tensor,
    coor_occ_tensor,
    B,
    R,
    D,
    grid_size_vol,
    d_coord_shift,
    scaled_vdim,
    scaled_vsize,
    max_o,
    raypos_mask_tensor
    );
}

void get_shadingloc(
  at::Tensor raypos_tensor,
  at::Tensor raypos_mask_tensor,
  size_t B,
  size_t R,
  size_t D,
  size_t samples_per_ray,
  at::Tensor sample_loc_tensor,
  at::Tensor sample_loc_mask_tensor
);

void get_shadingloc(
  at::Tensor raypos_tensor,
  at::Tensor raypos_mask_tensor,
  size_t B,
  size_t R,
  size_t D,
  size_t samples_per_ray,
  at::Tensor sample_loc_tensor,
  at::Tensor sample_loc_mask_tensor
) {
  CHECK_INPUT(raypos_tensor);
  CHECK_INPUT(coor_occ_tensor);
  CHECK_INPUT(d_coord_shift);
  CHECK_INPUT(scaled_vsize);
  CHECK_INPUT(scaled_vdim);
  CHECK_INPUT(raypos_mask_tensor);
  return get_shadingloc_kernel(
    raypos_tensor,
    raypos_mask_tensor,
    B,
    R,
    D,
    samples_per_ray,
    sample_loc_tensor,
    sample_loc_mask_tensor
    );
}

void query_along_ray(
  at::Tensor points,
  size_t B,
  size_t samples_per_ray,
  size_t R,
  size_t max_o,
  size_t P,
  size_t num_neighbors,
  scalar_t grid_size_vol,
  scalar_t radius_limit,
  at::Tensor d_coord_shift,
  at::Tensor scaled_vdim,
  at::Tensor scaled_vsize,
  at::Tensor kernel_size,
  at::Tensor occ_numpnts_tensor,
  at::Tensor occ_2_pnts_tensor,
  at::Tensor coor_2_occ_tensor,
  at::Tensor sample_loc_tensor,
  at::Tensor sample_loc_mask_tensor,
  at::Tensor sample_pidx_tensor
);

void query_along_ray(
  at::Tensor points,
  size_t B,
  size_t samples_per_ray,
  size_t R,
  size_t max_o,
  size_t P,
  size_t num_neighbors,
  scalar_t grid_size_vol,
  scalar_t radius_limit,
  at::Tensor d_coord_shift,
  at::Tensor scaled_vdim,
  at::Tensor scaled_vsize,
  at::Tensor kernel_size,
  at::Tensor occ_numpnts_tensor,
  at::Tensor occ_2_pnts_tensor,
  at::Tensor coor_2_occ_tensor,
  at::Tensor sample_loc_tensor,
  at::Tensor sample_loc_mask_tensor,
  at::Tensor sample_pidx_tensor
) {
  CHECK_INPUT(raypos_tensor);
  CHECK_INPUT(coor_occ_tensor);
  CHECK_INPUT(d_coord_shift);
  CHECK_INPUT(scaled_vsize);
  CHECK_INPUT(scaled_vdim);
  CHECK_INPUT(raypos_mask_tensor);
  return query_along_ray_kernel(
    points,
    B,
    samples_per_ray,
    R,
    max_o,
    P,
    num_neighbors,
    grid_size_vol,
    radius_limit,
    d_coord_shift,
    scaled_vdim,
    scaled_vsize,
    kernel_size,
    occ_numpnts_tensor,
    occ_2_pnts_tensor,
    coor_2_occ_tensor,
    sample_loc_tensor,
    sample_loc_mask_tensor,
    sample_pidx_tensor
    );
}
