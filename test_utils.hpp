
#pragma once

#include <chrono>
#include <random>

struct Timer {
    std::chrono::time_point<std::chrono::steady_clock> start;

    void tick()
    {
        start = std::chrono::steady_clock::now();
    }
    double tock()
    {
        auto end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9);
    }
};

template<typename T>
void random_arr(double *a, T len)
{
    // std::random_device rnd;
    // std::mt19937_64 gen(rnd());
    // std::uniform_real_distribution<> dis(-1, 1);
    // for (T i = 0; i < len; i++) {
    //     a[i] = dis(gen);
    // }

    for (T i = 0; i < len; i++) {
        // a[i] = 0.1;
        a[i] = (double)i / 10;
    }
}
