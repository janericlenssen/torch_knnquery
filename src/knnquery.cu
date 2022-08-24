// From https://github.com/Xharlie/pointnerf/blob/master/models/neural_points/query_point_indices_worldcoords.py
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <curand_kernel.h> 


template <typename scalar_t>
__global__ void claim_occ_kernel(
    const float* in_data,   // B * N * 3
    const int* in_actual_numpoints, // B 
    const int B,
    const int N,
    const scalar_t *d_coord_shift,     // 3
    const scalar_t *d_voxel_size,      // 3
    const int *d_grid_size,       // 3
    const int grid_size_vol,
    const int max_o,
    int* occ_idx, // B, all 0
    int *coor_2_occ,  // B * 400 * 400 * 400, all -1
    int *occ_2_coor,  // B * max_o * 3, all -1
    unsigned long seconds
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / N;  // index of batch
    if (i_batch >= B) { return; }
    int i_pt = index - N * i_batch;
    if (i_pt < in_actual_numpoints[i_batch]) {
        int coor[3];
        const scalar_t *p_pt = in_data + index * 3;
        coor[0] = (int) floor((p_pt[0] - d_coord_shift[0]) / d_voxel_size[0]);
        coor[1] = (int) floor((p_pt[1] - d_coord_shift[1]) / d_voxel_size[1]);
        coor[2] = (int) floor((p_pt[2] - d_coord_shift[2]) / d_voxel_size[2]);
        // printf("p_pt %f %f %f %f; ", p_pt[2], d_coord_shift[2], d_coord_shift[0], d_coord_shift[1]);
        if (coor[0] < 0 || coor[0] >= d_grid_size[0] || coor[1] < 0 || coor[1] >= d_grid_size[1] || coor[2] < 0 || coor[2] >= d_grid_size[2]) { return; }
        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
        
        int voxel_idx = coor_2_occ[coor_indx_b];
        if (voxel_idx == -1) {  // found an empty voxel
            int old_voxel_num = atomicCAS(
                    &coor_2_occ[coor_indx_b],
                    -1, 0
            );
            if (old_voxel_num == -1) {
                // CAS -> old val, if old val is -1
                // if we get -1, this thread is the one who obtain a new voxel
                // so only this thread should do the increase operator below
                int tmp = atomicAdd(occ_idx+i_batch, 1); // increase the counter, return old counter
                    // increase the counter, return old counter
                if (tmp < max_o) {
                    int coord_inds = (i_batch * max_o + tmp) * 3;
                    occ_2_coor[coord_inds] = coor[0];
                    occ_2_coor[coord_inds + 1] = coor[1];
                    occ_2_coor[coord_inds + 2] = coor[2];
                } else {
                    curandState state;
                    curand_init(index+2*seconds, 0, 0, &state);
                    int insrtidx = ceilf(curand_uniform(&state) * (tmp+1)) - 1;
                    if(insrtidx < max_o){
                        int coord_inds = (i_batch * max_o + insrtidx) * 3;
                        occ_2_coor[coord_inds] = coor[0];
                        occ_2_coor[coord_inds + 1] = coor[1];
                        occ_2_coor[coord_inds + 2] = coor[2];
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void map_coor2occ_kernel(
    const int B,
    const int *d_grid_size,       // 3
    const int *kernel_size,       // 3
    const int grid_size_vol,
    const int max_o,
    int* occ_idx, // B, all -1
    int *coor_occ,  // B * 400 * 400 * 400
    int *coor_2_occ,  // B * 400 * 400 * 400
    int *occ_2_coor  // B * max_o * 3
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / max_o;  // index of batch
    if (i_batch >= B) { return; }
    int i_pt = index - max_o * i_batch;
    if (i_pt < occ_idx[i_batch] && i_pt < max_o) {
        int coor[3];
        coor[0] = occ_2_coor[index*3];
        if (coor[0] < 0) { return; }
        coor[1] = occ_2_coor[index*3+1];
        coor[2] = occ_2_coor[index*3+2];
        
        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
        coor_2_occ[coor_indx_b] = i_pt;
        // printf("kernel_size[0] %d", kernel_size[0]);
        for (int coor_x = max(0, coor[0] - kernel_size[0] / 2) ; coor_x < min(d_grid_size[0], coor[0] + (kernel_size[0] + 1) / 2); coor_x++)    {
            for (int coor_y = max(0, coor[1] - kernel_size[1] / 2) ; coor_y < min(d_grid_size[1], coor[1] + (kernel_size[1] + 1) / 2); coor_y++)   {
                for (int coor_z = max(0, coor[2] - kernel_size[2] / 2) ; coor_z < min(d_grid_size[2], coor[2] + (kernel_size[2] + 1) / 2); coor_z++) {
                    coor_indx_b = i_batch * grid_size_vol + coor_x * (d_grid_size[1] * d_grid_size[2]) + coor_y * d_grid_size[2] + coor_z;
                    if (coor_occ[coor_indx_b] > 0) { continue; }
                    atomicCAS(coor_occ + coor_indx_b, 0, 1);
                }
            }
        }   
    }
}

template <typename scalar_t>
__global__ void fill_occ2pnts_kernel(
    const scalar_t* in_data,   // B * N * 3
    const int* in_actual_numpoints, // B 
    const int B,
    const int N,
    const int P,
    const scalar_t *d_coord_shift,     // 3
    const scalar_t *d_voxel_size,      // 3
    const int *d_grid_size,       // 3
    const int grid_size_vol,
    const int max_o,
    int *coor_2_occ,  // B * 400 * 400 * 400, all -1
    int *occ_2_pnts,  // B * max_o * P, all -1
    int *occ_numpnts,  // B * max_o, all 0
    unsigned long seconds
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / N;  // index of batch
    if (i_batch >= B) { return; }
    int i_pt = index - N * i_batch;
    if (i_pt < in_actual_numpoints[i_batch]) {
        int coor[3];
        const scalar_t *p_pt = in_data + index * 3;
        coor[0] = (int) floor((p_pt[0] - d_coord_shift[0]) / d_voxel_size[0]);
        coor[1] = (int) floor((p_pt[1] - d_coord_shift[1]) / d_voxel_size[1]);
        coor[2] = (int) floor((p_pt[2] - d_coord_shift[2]) / d_voxel_size[2]);
        if (coor[0] < 0 || coor[0] >= d_grid_size[0] || coor[1] < 0 || coor[1] >= d_grid_size[1] || coor[2] < 0 || coor[2] >= d_grid_size[2]) { return; }
        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
        
        int voxel_idx = coor_2_occ[coor_indx_b];
        if (voxel_idx > 0) {  // found an claimed coor2occ
            int occ_indx_b = i_batch * max_o + voxel_idx;
            int tmp = atomicAdd(occ_numpnts + occ_indx_b, 1); // increase the counter, return old counter
            if (tmp < P) {
                occ_2_pnts[occ_indx_b * P + tmp] = i_pt;
            } else {
                curandState state;
                curand_init(index+2*seconds, 0, 0, &state);
                int insrtidx = ceilf(curand_uniform(&state) * (tmp+1)) - 1;
                if(insrtidx < P){
                    occ_2_pnts[occ_indx_b * P + insrtidx] = i_pt;
                }
            }
        }
    }
}

    
template <typename scalar_t>        
__global__ void mask_raypos_kernel(
    scalar_t *raypos,    // [B, 2048, 400, 3]
    int *coor_occ,    // B * 400 * 400 * 400
    const int B,       // 3
    const int R,       // 3
    const int D,       // 3
    const int grid_size_vol,
    const scalar_t *d_coord_shift,     // 3
    const int *d_grid_size,       // 3
    const scalar_t *d_voxel_size,      // 3
    int *raypos_mask    // B, R, D
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / (R * D);  // index of batch
    if (i_batch >= B) { return; }
    int coor[3];
    coor[0] = (int) floor((raypos[index*3] - d_coord_shift[0]) / d_voxel_size[0]);
    coor[1] = (int) floor((raypos[index*3+1] - d_coord_shift[1]) / d_voxel_size[1]);
    coor[2] = (int) floor((raypos[index*3+2] - d_coord_shift[2]) / d_voxel_size[2]);
    // printf(" %f %f %f;", raypos[index*3], raypos[index*3+1], raypos[index*3+2]);
    if ((coor[0] >= 0) && (coor[0] < d_grid_size[0]) && (coor[1] >= 0) && (coor[1] < d_grid_size[1]) && (coor[2] >= 0) && (coor[2] < d_grid_size[2])) { 
        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
        raypos_mask[index] = coor_occ[coor_indx_b];
    }
}


template <typename scalar_t>
__global__ void get_shadingloc_kernel(
    const scalar_t *raypos,    // [B, 2048, 400, 3]
    const int *raypos_mask,    // B, R, D
    const int B,       // 3
    const int R,       // 3
    const int D,       // 3
    const int SR,       // 3
    scalar_t *sample_loc,       // B * R * SR * 3
    int *sample_loc_mask       // B * R * SR
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / (R * D);  // index of batch
    if (i_batch >= B) { return; }
    int temp = raypos_mask[index];
    if (temp >= 0) {
        int r = (index - i_batch * R * D) / D;
        int loc_inds = i_batch * R * SR + r * SR + temp;
        sample_loc[loc_inds * 3] = raypos[index * 3];
        sample_loc[loc_inds * 3 + 1] = raypos[index * 3 + 1];
        sample_loc[loc_inds * 3 + 2] = raypos[index * 3 + 2];
        sample_loc_mask[loc_inds] = 1;
    }
}


template <typename scalar_t>
__global__ void query_along_ray_kernel(
    const scalar_t* in_data,   // B * N * 3
    const int B,
    const int SR,               // num. samples along each ray e.g., 128
    const int R,               // e.g., 1024
    const int max_o,
    const int P,
    const int K,                // num.  neighbors
    const int grid_size_vol,
    const scalar_t radius_limit2,
    const scalar_t *d_coord_shift,     // 3
    const int *d_grid_size,
    const scalar_t *d_voxel_size,      // 3
    const int *kernel_size,
    const int *occ_numpnts,    // B * max_o
    const int *occ_2_pnts,            // B * max_o * P
    const int *coor_2_occ,      // B * 400 * 400 * 400 
    const scalar_t *sample_loc,       // B * R * SR * 3
    const int *sample_loc_mask,       // B * R * SR
    int *sample_pidx       // B * R * SR * K
) {
    int index =  blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / (R * SR);  // index of batch
    if (i_batch >= B || sample_loc_mask[index] <= 0) { return; }
    scalar_t centerx = sample_loc[index * 3];
    scalar_t centery = sample_loc[index * 3 + 1];
    scalar_t centerz = sample_loc[index * 3 + 2];
    int frustx = (int) floor((centerx - d_coord_shift[0]) / d_voxel_size[0]);
    int frusty = (int) floor((centery - d_coord_shift[1]) / d_voxel_size[1]);
    int frustz = (int) floor((centerz - d_coord_shift[2]) / d_voxel_size[2]);
                        
    centerx = sample_loc[index * 3];
    centery = sample_loc[index * 3 + 1];
    centerz = sample_loc[index * 3 + 2];
                        
    int kid = 0, far_ind = 0, coor_z, coor_y, coor_x;
    scalar_t far2 = 0.0;
    scalar_t xyz2Buffer[K];
    for (int layer = 0; layer < (kernel_size[0]+1)/2; layer++){                        
        for (int x = max(-frustx, -layer); x < min(d_grid_size[0] - frustx, layer + 1); x++) {
            coor_x = frustx + x;
            for (int y = max(-frusty, -layer); y < min(d_grid_size[1] - frusty, layer + 1); y++) {
                coor_y = frusty + y;
                for (int z =  max(-frustz, -layer); z < min(d_grid_size[2] - frustz, layer + 1); z++) {
                    coor_z = z + frustz;
                    if (max(abs(z), max(abs(x), abs(y))) != layer) continue;
                    int coor_indx_b = i_batch * grid_size_vol + coor_x * (d_grid_size[1] * d_grid_size[2]) + coor_y * d_grid_size[2] + coor_z;
                    int occ_indx = coor_2_occ[coor_indx_b] + i_batch * max_o;
                    if (occ_indx >= 0) {
                        for (int g = 0; g < min(P, occ_numpnts[occ_indx]); g++) {
                            int pidx = occ_2_pnts[occ_indx * P + g];
                            scalar_t x_v = (in_data[pidx*3]-centerx);
                            scalar_t y_v = (in_data[pidx*3 + 1]-centery);
                            scalar_t z_v = (in_data[pidx*3 + 2]-centerz);
                            scalar_t xyz2 = x_v * x_v + y_v * y_v + z_v * z_v;
                            if ((radius_limit2 == 0 || xyz2 <= radius_limit2)){
                                if (kid++ < K) {
                                    sample_pidx[index * K + kid - 1] = pidx;
                                    xyz2Buffer[kid-1] = xyz2;
                                    if (xyz2 > far2){
                                        far2 = xyz2;
                                        far_ind = kid - 1;
                                    }
                                } else {
                                    if (xyz2 < far2) {
                                        sample_pidx[index * K + far_ind] = pidx;
                                        xyz2Buffer[far_ind] = xyz2;
                                        far2 = xyz2;
                                        for (int i = 0; i < K; i++) {
                                            if (xyz2Buffer[i] > far2) {
                                                far2 = xyz2Buffer[i];
                                                far_ind = i;
                                            }
                                        }
                                    } 
                                }
                            }
                        }
                    }
                }
            }
        }
        if (kid >= K) break;
    }
}