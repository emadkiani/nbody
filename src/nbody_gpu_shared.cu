#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "common.h"

__global__ void compute_forces_gpu_shared(
    const float3 *pos,
    const float *mass,
    float3 *acc,
    int N)
{
    extern __shared__ char smem[];
    float3 *spos = (float3 *)smem;
    float *smass = (float *)(spos + blockDim.x);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    float3 pi = pos[i];
    float3 ai = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < N; tile += blockDim.x)
    {

        int idx = tile + threadIdx.x;
        if (idx < N)
        {
            spos[threadIdx.x] = pos[idx];
            smass[threadIdx.x] = mass[idx];
        }
        __syncthreads();

        int tileSize = min(blockDim.x, N - tile);
        for (int j = 0; j < tileSize; ++j)
        {

            float3 r;
            r.x = spos[j].x - pi.x;
            r.y = spos[j].y - pi.y;
            r.z = spos[j].z - pi.z;

            float dist2 = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING;
            float invDist = rsqrtf(dist2);
            float invDist3 = invDist * invDist * invDist;

            float s = G * smass[j] * invDist3;
            ai.x += r.x * s;
            ai.y += r.y * s;
            ai.z += r.z * s;
        }
        __syncthreads();
    }

    acc[i] = ai;
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
    int N = 4096;
    if (argc > 1)
        N = std::stoi(argv[1]);

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

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    size_t sharedBytes =
        block.x * (sizeof(float3) + sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    compute_forces_gpu_shared<<<grid, block, sharedBytes>>>(d_pos, d_mass, d_acc, N);
    integrate_gpu<<<grid, block>>>(d_pos, d_vel, d_acc, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_pos.data(), d_pos, N * sizeof(float3), cudaMemcpyDeviceToHost);

    std::cout << "GPU shared memory\n";
    std::cout << "Particles: " << N << "\n";
    std::cout << "Time: " << ms / 1000.0f << " s\n";
    std::cout << "pos[0]: "
              << h_pos[0].x << " "
              << h_pos[0].y << " "
              << h_pos[0].z << "\n";

    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_acc);
    cudaFree(d_mass);

    return 0;
}
