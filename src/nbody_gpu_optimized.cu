#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include "common.h"

__global__ void compute_forces_gpu_optimized(
    const float3 *pos,
    const float *mass,
    float3 *acc,
    int N)
{
    extern __shared__ float smem[];
    float *sx = smem;
    float *sy = sx + blockDim.x;
    float *sz = sy + blockDim.x;
    float *sm = sz + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    if (i >= N)
        return;

    float pix = pos[i].x;
    float piy = pos[i].y;
    float piz = pos[i].z;

    float aix = 0.0f;
    float aiy = 0.0f;
    float aiz = 0.0f;

    for (int tile = 0; tile < N; tile += blockDim.x)
    {

        int idx = tile + tid;
        if (idx < N)
        {
            sx[tid] = pos[idx].x;
            sy[tid] = pos[idx].y;
            sz[tid] = pos[idx].z;
            sm[tid] = mass[idx];
        }
        __syncthreads();

        int tileSize = min(blockDim.x, N - tile);
        int self = i - tile;

        for (int j = 0; j < tileSize; ++j)
        {
            if (j == self)
                continue;

            float rx = sx[j] - pix;
            float ry = sy[j] - piy;
            float rz = sz[j] - piz;

            float dist2 = rx * rx + ry * ry + rz * rz + SOFTENING;
            float invDist = rsqrtf(dist2);
            float invDist3 = invDist * invDist * invDist;

            float s = G * sm[j] * invDist3;
            aix += rx * s;
            aiy += ry * s;
            aiz += rz * s;
        }
        __syncthreads();
    }

    acc[i] = {aix, aiy, aiz};
}

__global__ void integrate_gpu(
    float3 *pos,
    float3 *vel,
    const float3 *acc,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    vel[i].x += acc[i].x * DT;
    vel[i].y += acc[i].y * DT;
    vel[i].z += acc[i].z * DT;

    pos[i].x += vel[i].x * DT;
    pos[i].y += vel[i].y * DT;
    pos[i].z += vel[i].z * DT;
}

int main(int argc, char **argv)
{
    int N = 8192;
    if (argc > 1)
        N = std::stoi(argv[1]);

    const int BLOCK_SIZE = 128; // try 128 and 256

    std::vector<float3> h_pos(N), h_vel(N), h_acc(N);
    std::vector<float> h_mass(N);

    initialize_particles(h_pos, h_vel, h_mass);

    float3 *d_pos, *d_vel, *d_acc;
    float *d_mass;

    cudaMalloc(&d_pos, N * sizeof(float3));
    cudaMalloc(&d_vel, N * sizeof(float3));
    cudaMalloc(&d_acc, N * sizeof(float3));
    cudaMalloc(&d_mass, N * sizeof(float));

    cudaMemcpy(d_pos, h_pos.data(), N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, h_vel.data(), N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_mass.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    size_t sharedBytes = BLOCK_SIZE * 4 * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 5; ++i)
    {
        compute_forces_gpu_optimized<<<grid, block, sharedBytes>>>(d_pos, d_mass, d_acc, N);
        integrate_gpu<<<grid, block>>>(d_pos, d_vel, d_acc, N);
    }
    cudaDeviceSynchronize();

    const int iters = 10;

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
    {
        compute_forces_gpu_optimized<<<grid, block, sharedBytes>>>(d_pos, d_mass, d_acc, N);
        integrate_gpu<<<grid, block>>>(d_pos, d_vel, d_acc, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_pos.data(), d_pos, N * sizeof(float3), cudaMemcpyDeviceToHost);

    std::cout << "GPU optimized\n";
    std::cout << "Particles: " << N << "\n";
    std::cout << "Block size: " << BLOCK_SIZE << "\n";
    std::cout << "Time per step: "
              << (ms / iters) / 1000.0f << " s\n";
    std::cout << "pos[0]: "
              << h_pos[0].x << " "
              << h_pos[0].y << " "
              << h_pos[0].z << "\n";

    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_acc);
    cudaFree(d_mass);

    {
        std::ofstream csv("results_nbody.csv", std::ios::app);
        csv << N << ",," << (ms / 1000.0f) << "," << block.x << "\n";
    }

    return 0;
}
