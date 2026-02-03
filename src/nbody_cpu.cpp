#include <iostream>
#include <chrono>
#include <fstream>
#include "common.h"

void compute_forces_cpu(
    const std::vector<float3> &pos,
    const std::vector<float> &mass,
    std::vector<float3> &acc)
{
    const size_t N = pos.size();

    for (size_t i = 0; i < N; ++i)
    {
        float3 a = {0.0f, 0.0f, 0.0f};

        for (size_t j = 0; j < N; ++j)
        {
            if (i == j)
                continue;

            float3 r = pos[j] - pos[i];
            float dist2 = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING;
            float invDist = 1.0f / std::sqrt(dist2);
            float invDist3 = invDist * invDist * invDist;

            float s = G * mass[j] * invDist3;
            a = a + r * s;
        }
        acc[i] = a;
    }
}

void integrate_cpu(
    std::vector<float3> &pos,
    std::vector<float3> &vel,
    const std::vector<float3> &acc)
{
    for (size_t i = 0; i < pos.size(); ++i)
    {
        vel[i] = vel[i] + acc[i] * DT;
        pos[i] = pos[i] + vel[i] * DT;
    }
}

int main(int argc, char **argv)
{
    size_t N = 4096;
    if (argc > 1)
        N = std::stoul(argv[1]);

    std::vector<float3> pos(N), vel(N), acc(N);
    std::vector<float> mass(N);

    initialize_particles(pos, vel, mass);

    auto start = std::chrono::high_resolution_clock::now();

    compute_forces_cpu(pos, mass, acc);
    integrate_cpu(pos, vel, acc);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "CPU N-body\n";
    std::cout << "Particles: " << N << "\n";
    std::cout << "Time: " << elapsed.count() << " s\n";

    std::cout << "pos[0]: "
              << pos[0].x << " "
              << pos[0].y << " "
              << pos[0].z << "\n";

    {
        std::ofstream csv("results_nbody.csv", std::ios::app);
        csv << N << "," << elapsed.count() << ",,\n";
    }

    return 0;
}
