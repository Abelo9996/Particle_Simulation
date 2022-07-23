#include "common.h"
#include <cuda.h>

#define NUM_THREADS 256

int* node_list;
int bin_size;
int bin_count;
int* node_size_list;

// Put any static global variables here that you will use throughout the simulation.
int blks;
int blks_bin;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__device__ void interact(particle_t* parts, int* node_list, int* node_size_list, int row, int col, int particle, int bin_size) {
    if (row >= 0 && row < bin_size && col >= 0 && col < bin_size){
        int i = row*bin_size+col;
        for (int idx = 0; idx < node_size_list[i];idx++) {
            apply_force_gpu(parts[particle], parts[node_list[i*16+idx]]);
        }
    }
}

__global__ void compute_forces_gpu(particle_t* parts, int* node_list, int* node_size_list, int bin_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < bin_size*bin_size){
        int i = tid / bin_size;
        int j = tid % bin_size;
        for (int idx = 0; idx < node_size_list[tid]; idx++) {
            int particle = node_list[16*tid+idx];
            parts[particle].ax = 0;
            parts[particle].ay = 0;
            interact(parts, node_list, node_size_list, i-1, j-1, particle, bin_size);
            interact(parts, node_list, node_size_list, i-1, j, particle, bin_size);
            interact(parts, node_list, node_size_list, i-1, j+1, particle, bin_size);
            interact(parts, node_list, node_size_list, i, j-1, particle, bin_size);
            interact(parts, node_list, node_size_list, i, j, particle, bin_size);
            interact(parts, node_list, node_size_list, i, j+1, particle, bin_size);
            interact(parts, node_list, node_size_list, i+1, j-1, particle, bin_size);
            interact(parts, node_list, node_size_list, i+1, j, particle, bin_size);
            interact(parts, node_list, node_size_list, i+1, j+1, particle, bin_size);
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    bin_size = (size/cutoff)+1;
    bin_count = bin_size*bin_size;
    cudaMalloc(&node_list, 16*bin_count*sizeof(int));
    cudaMalloc(&node_size_list, bin_count*sizeof(int));
    blks_bin = (bin_count + NUM_THREADS - 1) / NUM_THREADS;
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
}

__global__ void reset(int* node_size_list, int bin_count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    node_size_list[tid] = (tid < bin_count) ? 0 : node_size_list[tid];
}

__global__ void rebin(particle_t* parts, int* node_list, int* node_size_list, int bin_size, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_parts){
        int idx = (floor(parts[tid].x/cutoff) * bin_size)+floor(parts[tid].y/cutoff);
        node_list[(16*idx)+atomicAdd(&node_size_list[idx], 1)] = tid;
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    reset<<<blks_bin, NUM_THREADS>>>(node_size_list, bin_count);
    rebin<<<blks, NUM_THREADS>>>(parts, node_list, node_size_list, bin_size, num_parts);
    compute_forces_gpu<<<blks_bin, NUM_THREADS>>>(parts, node_list, node_size_list, bin_size);
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}