#include <torch/extension.h>

#define IS_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor");
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");
#define CHECK_INPUT(x) IS_CUDA(x) IS_CONTIGUOUS(x)

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

// --- Headers --- //

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

// --- Implementation --- //

void claim_occ(
  at::Tensor points,
  at::Tensor actual_num_points_per_batch,
  size_t B,
  size_t N,
  at::Tensor d_coord_shift,
  at::Tensor scaled_vsize,
  float grid_size_vol,
  size_t max_o,
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

  int units = B*N;

  AT_DISPATCH_FLOATING_TYPES(points.type(), "claim_occ_kernel", [&] {
    claim_occ_kernel<scalar_t><<<BLOCKS(units), THREADS>>>(
      points.data<scalar_t>(),
      actual_num_points_per_batch.data<scalar_t>(),
      B,
      N,
      d_coord_shift.data<scalar_t>(),
      scaled_vsize.data<scalar_t>(),
      grid_size_vol,
      max_o,
      coor_2_occ_tensor.data<scalar_t>(),
      occ_2_coor_tensor.data<scalar_t>(),
      seconds
    );
  });
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
) {
  CHECK_INPUT(scaled_vdim);
  CHECK_INPUT(kernel_size);
  CHECK_INPUT(occ_idx_tensor);
  CHECK_INPUT(coor_occ_tensor);
  CHECK_INPUT(coor_2_occ_tensor);
  CHECK_INPUT(occ_2_coor_tensor);

  int units = B * max_o;

  AT_DISPATCH_FLOATING_TYPES(coor_occ_tensor.type(), "map_coor2occ_kernel", [&] {
    map_coor2occ_kernel<scalar_t><<<BLOCKS(units), THREADS>>>(
      B,
      scaled_vdim.data<scalar_t>(),
      kernel_size.data<scalar_t>(),
      grid_size_vol,
      max_o,
      occ_idx_tensor.data<scalar_t>(),
      coor_occ_tensor.data<scalar_t>(),
      coor_2_occ_tensor.data<scalar_t>(),
      occ_2_coor_tensor.data<scalar_t>()
    );
  });
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
) {
  CHECK_INPUT(points);
  CHECK_INPUT(actual_num_points_per_batch);
  CHECK_INPUT(d_coord_shift);
  CHECK_INPUT(scaled_vsize);
  CHECK_INPUT(scaled_vdim);
  CHECK_INPUT(coor_2_occ_tensor);
  CHECK_INPUT(occ_2_pnts_tensor);
  CHECK_INPUT(occ_numpnts_tensor);

  int units = B*N;

  AT_DISPATCH_FLOATING_TYPES(points.type(), "fill_occ2pnts_kernel", [&] {
    fill_occ2pnts_kernel<scalar_t><<<BLOCKS(units), THREADS>>>(
      points.data<scalar_t>(),
      actual_num_points_per_batch.data<scalar_t>(),
      B,
      N,
      P,
      d_coord_shift.data<scalar_t>(),
      scaled_vsize.data<scalar_t>(),
      scaled_vdim.data<scalar_t>(),
      grid_size_vol,
      max_o,
      coor_2_occ_tensor.data<scalar_t>(),
      occ_2_pnts_tensor.data<scalar_t>(),
      occ_numpnts_tensor.data<scalar_t>(),
      seconds
    );
  });
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
) {
  CHECK_INPUT(raypos_tensor);
  CHECK_INPUT(coor_occ_tensor);
  CHECK_INPUT(d_coord_shift);
  CHECK_INPUT(scaled_vsize);
  CHECK_INPUT(scaled_vdim);
  CHECK_INPUT(raypos_mask_tensor);

  int units = B * R * D;

  AT_DISPATCH_FLOATING_TYPES(raypos_tensor.type(), "mask_raypos_kernel", [&] {
    mask_raypos_kernel<scalar_t><<<BLOCKS(units), THREADS>>>(
      raypos_tensor.data<scalar_t>(),
      coor_occ_tensor.data<scalar_t>(),
      B,
      R,
      D,
      grid_size_vol,
      d_coord_shift.data<scalar_t>(),
      scaled_vdim.data<scalar_t>(),
      scaled_vsize.data<scalar_t>(),
      max_o,
      raypos_mask_tensor.data<scalar_t>()
    );
  });
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
) {
  CHECK_INPUT(raypos_tensor);
  CHECK_INPUT(coor_occ_tensor);
  CHECK_INPUT(d_coord_shift);
  CHECK_INPUT(scaled_vsize);
  CHECK_INPUT(scaled_vdim);
  CHECK_INPUT(raypos_mask_tensor);

  int units = B * R * D;

  AT_DISPATCH_FLOATING_TYPES(raypos_tensor.type(), "get_shadingloc_kernel", [&] {
    get_shadingloc_kernel<scalar_t><<<BLOCKS(units), THREADS>>>(
      raypos_tensor.data<scalar_t>(),
      raypos_mask_tensor.data<scalar_t>(),
      B,
      R,
      D,
      samples_per_ray,
      sample_loc_tensor.data<scalar_t>(),
      sample_loc_mask_tensor.data<scalar_t>()
    );
  });
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
) {
  CHECK_INPUT(raypos_tensor);
  CHECK_INPUT(coor_occ_tensor);
  CHECK_INPUT(d_coord_shift);
  CHECK_INPUT(scaled_vsize);
  CHECK_INPUT(scaled_vdim);
  CHECK_INPUT(raypos_mask_tensor);

  int units = B * R * samples_per_ray;

  AT_DISPATCH_FLOATING_TYPES(points.type(), "query_along_ray_kernel", [&] {
    query_along_ray_kernel<scalar_t><<<BLOCKS(units), THREADS>>>(
      points.data<scalar_t>(),
      B,
      samples_per_ray,
      R,
      max_o,
      P,
      num_neighbors,
      grid_size_vol,
      radius_limit,
      d_coord_shift.data<scalar_t>(),
      scaled_vdim.data<scalar_t>(),
      scaled_vsize.data<scalar_t>(),
      kernel_size.data<scalar_t>(),
      occ_numpnts_tensor.data<scalar_t>(),
      occ_2_pnts_tensor.data<scalar_t>(),
      coor_2_occ_tensor.data<scalar_t>(),
      sample_loc_tensor.data<scalar_t>(),
      sample_loc_mask_tensor.data<scalar_t>(),
      sample_pidx_tensor.data<scalar_t>()
    );
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("claim_occ", &claim_occ, "Claim occupied voxels");
  m.def("map_coor2occ", &map_coor2occ, "Map coordinates to occupied voxels");
  m.def("fill_occ2pnts", &fill_occ2pnts, "Fill occupied voxels with points");
  m.def("mask_raypos", &mask_raypos, "Find mask of ray positions that hit occupied voxels");
  m.def("get_shadingloc", &get_shadingloc, "Get shading locations");
  m.def("query_along_ray", &query_along_ray, "Query KNN point indices");
}