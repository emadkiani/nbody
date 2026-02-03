#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <cstddef>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
struct float3
{
    float x, y, z;
};
#endif

constexpr float G = 6.67430e-3f;
constexpr float SOFTENING = 1e-5f;
constexpr float DT = 0.01f;

inline float3 operator+(const float3 &a, const float3 &b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline float3 operator-(const float3 &a, const float3 &b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline float3 operator*(const float3 &a, float s)
{
    return {a.x * s, a.y * s, a.z * s};
}

inline void initialize_particles(
    std::vector<float3> &pos,
    std::vector<float3> &vel,
    std::vector<float> &mass)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> mdist(0.5f, 10.0f);

    for (size_t i = 0; i < pos.size(); ++i)
    {
        pos[i] = {dist(gen), dist(gen), dist(gen)};
        vel[i] = {0.0f, 0.0f, 0.0f};
        mass[i] = mdist(gen);
    }
}
