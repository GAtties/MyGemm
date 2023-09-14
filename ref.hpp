
#pragma once

#include <cstdint>

using i64 = std::int64_t;

#ifndef USE_MKL

inline void ref_dgemm(i64 m, i64 n, i64 k, const double *a, const double *b, double *c)
{
    for (i64 i = 0; i < m; i++) {
        for (i64 j = 0; j < n; j++) {
            for (i64 p = 0; p < k; p++) {
                c[i + j * m] += a[i + p * m] * b[p + j * k];
            }
        }
    }
}

#else

#include <mkl.h>

inline void ref_dgemm(i64 m, i64 n, i64 k, const double *a, const double *b, double *c)
{
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, m, b, k, 1, c, m);
}

#endif
