
// avx2- dgemm impl
// assmue alpha = 1, beta = 1
// col-major matrix, NN

constexpr bool dgemm_check = true;

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <cmath>
#include <cassert>
#include <cstring>

#include <omp.h>

#include "test_utils.hpp"
#include "ref.hpp"

using std::cout;
using std::endl;
using std::size_t;
using std::int32_t;
using std::int64_t;

constexpr unsigned int simd_bit = 256;
constexpr unsigned int simd_size = simd_bit / 8;
constexpr unsigned int simd_len = simd_size / sizeof(double);

constexpr unsigned int mr = simd_len * 2;
constexpr unsigned int nr = 4;

using i64 = int64_t;

// these parameters are associated to L1 L2 L3 cache hw prefetch and cpu micro-architecture
constexpr i64 mc = 320;
constexpr i64 kc = 256;
constexpr i64 nc = 4096;

constexpr size_t mem_alignment = 512;

// GF limit for 12400F: 4.4 * 16 = 70.4 GF
// 16: 4 double * 2 fsu * 2 (Multiply-Add-Fused)
constexpr double frequency = 4.4;
constexpr double gf_limit = frequency * 16;

// I compile this code on windows with mingw64, and I want it possible to be compiled on linux
#ifdef _WIN32
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size);
#define aligned_free(ptr) free(ptr)
#endif

// avx2 have 16 simd regs, each reg have 256 bits
// the limit block of avx2 may be v2*v4, will cost 14 regs, 2 more regs for a to do M1-M2 kernel

// Actally, we can save the regs from A, becuase we can do 'fmadd' without load data to regs in x64

// Asm code example:
// vfmadd132pd a_v0, b_v0, c_v0 (pseudocode)
// vfmadd132pd %ymm0, %ymm1, %ymm2 (Explicitly use reg to represent A)
// vfmadd132pd (%rdx), %ymm1, %ymm2 (Use register addressing to represent A), and one reg have been saved

// In fact, the actual number of physical simd registers in cpu core, is not just ymm0 ~ ymm15, we do not need to
// worry about that saving registers may cause lower performance. We can also make a program to test the number
// of real registers just in case.

// kernel: 8*4

#include <immintrin.h>

// assume size aligned as kernel size

// v2 * v4
static void opt_dgemm_micro_kernel_8x4(i64 k, const double *pa, const double *pb, double *c, i64 ldc)
{
    // use loadu to prevent error when C is not aligned as simd_size
    __m256d c_v00 = _mm256_loadu_pd(c + 0 + 0 * ldc);
    __m256d c_v01 = _mm256_loadu_pd(c + 4 + 0 * ldc);

    __m256d c_v10 = _mm256_loadu_pd(c + 0 + 1 * ldc);
    __m256d c_v11 = _mm256_loadu_pd(c + 4 + 1 * ldc);

    __m256d c_v20 = _mm256_loadu_pd(c + 0 + 2 * ldc);
    __m256d c_v21 = _mm256_loadu_pd(c + 4 + 2 * ldc);

    __m256d c_v30 = _mm256_loadu_pd(c + 0 + 3 * ldc);
    __m256d c_v31 = _mm256_loadu_pd(c + 4 + 3 * ldc);

    for (i64 i = 0; i < k; i++) {
        __m256d a_v0 = _mm256_load_pd(pa);
        __m256d a_v1 = _mm256_load_pd(pa + 4);

        __m256d b_v0 = _mm256_broadcast_sd(pb + 0);
        __m256d b_v1 = _mm256_broadcast_sd(pb + 1);
        __m256d b_v2 = _mm256_broadcast_sd(pb + 2);
        __m256d b_v3 = _mm256_broadcast_sd(pb + 3);

        // outer product store to register
        c_v00 = _mm256_fmadd_pd(a_v0, b_v0, c_v00);
        c_v01 = _mm256_fmadd_pd(a_v1, b_v0, c_v01);

        c_v10 = _mm256_fmadd_pd(a_v0, b_v1, c_v10);
        c_v11 = _mm256_fmadd_pd(a_v1, b_v1, c_v11);

        c_v20 = _mm256_fmadd_pd(a_v0, b_v2, c_v20);
        c_v21 = _mm256_fmadd_pd(a_v1, b_v2, c_v21);

        c_v30 = _mm256_fmadd_pd(a_v0, b_v3, c_v30);
        c_v31 = _mm256_fmadd_pd(a_v1, b_v3, c_v31);

        pa += 8;
        pb += 4;
    }

    // save v2xv4
    _mm256_storeu_pd(c + 0 + 0 * ldc, c_v00);
    _mm256_storeu_pd(c + 4 + 0 * ldc, c_v01);

    _mm256_storeu_pd(c + 0 + 1 * ldc, c_v10);
    _mm256_storeu_pd(c + 4 + 1 * ldc, c_v11);

    _mm256_storeu_pd(c + 0 + 2 * ldc, c_v20);
    _mm256_storeu_pd(c + 4 + 2 * ldc, c_v21);

    _mm256_storeu_pd(c + 0 + 3 * ldc, c_v30);
    _mm256_storeu_pd(c + 4 + 3 * ldc, c_v31);
}

double kernel_cost = 0.0;

static void dgemm_kernel(i64 m, i64 n, i64 k, const double *pa, const double *pb, double *c, i64 ldc)
{
    // TODO: add tail process
    assert(m % mr == 0);
    assert(n % nr == 0);

    Timer kernel_timer;

    kernel_timer.tick();

    i64 i, j;
    for (i = 0; i <= n - nr; i += nr) {
        for (j = 0; j <= m - mr; j += mr) {
            opt_dgemm_micro_kernel_8x4(k, pa + j * k, pb + i * k, c + j + i * ldc, ldc);
        }

        // tail process
    }

    // tail process

    double cost = kernel_timer.tock();
    kernel_cost += cost;
}

template<typename T>
concept CopySliceIter = requires(T func, int idx) {
    func(idx);
};

template<CopySliceIter Func, int start, int end>
static void repeat(Func func)
{
    if constexpr (start != end) {
        func(start);
        repeat<Func, start + 1, end>(func);
    }
}

template<CopySliceIter Func, int num>
static void repeat(Func func)
{
    repeat<Func, 0, num>(func);
}

template<int copy_size>
static void dgemm_tcopy_slice(const double *(&in), i64 ld, double *(&out))
{
    // C style
    // #pragma omp simd
    // for (int i = 0; i < copy_size; i++) {
    //     out[i] = in[i];
    // }

    // C++ style
    auto unroll_expr = [=](int idx) {
        out[idx] = in[idx];
    };
    repeat<decltype(unroll_expr), copy_size>(unroll_expr);

    in += ld;
    out += copy_size;
}

template<int copy_size>
static void dgemm_ncopy_slice(const double *(&in), i64 ld, double *(&out))
{
    // C style
    // for (int i = 0; i < copy_size; i++) {
    //     out[i] = in[i * ld];
    // }

    // C++ style
    auto unroll_expr = [=](int idx) {
        out[idx] = in[idx * ld];
    };
    repeat<decltype(unroll_expr), copy_size>(unroll_expr);

    in ++;
    out += copy_size;
}

using CopySlice = void (*)(const double *(&), i64, double *(&));

template<uint32_t i, int idx>
constexpr inline int clz_helper()
{
    static_assert(idx >= 0);

    if constexpr (idx != 0) {
        if constexpr (i & (1 << idx) != 0) {
            return idx;
        } else {
            return clz_helper<i, idx - 1>();
        }
    } else {
        return 0;
    }
}

// get highest bit, maybe same as gcc __builtin_clz
template<uint32_t i>
constexpr inline int clz()
{
    return clz_helper<i, 31>();
}

template<unsigned int copy_size>
constexpr inline int next_copy_size()
{
    return copy_size & (copy_size - 1) == 0 ? copy_size >> 1 : 1 << clz<copy_size>();
}

template<bool trans, unsigned int copy_size>
i64 get_slice_offset(i64 ld)
{
    if constexpr (!trans) {
        return ld;
    } else {
        return 1;
    }
}

template<bool trans, unsigned int copy_size>
i64 get_vertical_offset(i64 ld)
{
    if constexpr (!trans) {
        return 1;
    } else {
        return ld;
    }
}

template<bool trans, unsigned int copy_size>
struct CopySliceTraits {
};

template<unsigned int copy_size>
struct CopySliceTraits<false, copy_size> {
    static constexpr auto func = dgemm_ncopy_slice<copy_size>;
};

template<unsigned int copy_size>
struct CopySliceTraits<true, copy_size> {
    static constexpr auto func = dgemm_tcopy_slice<copy_size>;
};

template<bool trans, unsigned int copy_size>
static void dgemm_copy_tail(i64 tail_m, i64 n, const double *in, i64 ld, double *out)
{
    constexpr auto slice_func = CopySliceTraits<trans, copy_size>::func;
    if constexpr (copy_size != 0) {
        i64 slice_offset = get_slice_offset<trans, copy_size>(ld);
        if ((tail_m & copy_size) != 0) {
            for (i64 i = 0; i < n; i ++) {
                const double *in_ptr = in + i * slice_offset;
                slice_func(in_ptr, ld, out);
            }
            tail_m += copy_size;
        }
        dgemm_copy_tail<trans, next_copy_size<copy_size>()>(tail_m, n, in, ld, out);
    }
}

template<bool trans, unsigned int copy_size>
static void dgemm_copy(i64 m, i64 n, const double *in, i64 ld, double *out)
{
    constexpr auto slice_func = CopySliceTraits<trans, copy_size>::func;
    i64 slice_offset = get_slice_offset<trans, copy_size>(ld);

    i64 i;
    for (i = 0; i <= n - copy_size; i += copy_size) {
        const double *in_ptr = in + i * slice_offset;
        for (i64 j = 0; j < m; j++) {
            slice_func(in_ptr, ld, out);
        }
    }
    dgemm_copy_tail<trans, next_copy_size<copy_size>()>(m - i, n, in, ld, out);
}

void opt_dgemm(i64 m, i64 n, i64 k, const double *a, const double *b, double *c)
{
    double *pa = (double *)aligned_malloc(mc * kc * sizeof(double), mem_alignment);
    double *pb = (double *)aligned_malloc(nc * kc * sizeof(double), mem_alignment);

    auto dgemm_itcopy = dgemm_copy<true, mr>;  // naming style is from OpenBLAS
    auto dgemm_oncopy = dgemm_copy<false, nr>;

    i64 i, j, p;

    // n -> k -> m: We can get best performance with this direction, cause for the 'pa' is small enough to store in L2 cache.
    // So we can get better performance than doing redundant packing with 'pb'
    for (i = 0; i < n; i += nc) {
        i64 nb = std::min(nc, n - i);

        for (j = 0; j < k; j += kc) {
            i64 kb = std::min(kc, k - j);

            dgemm_oncopy(kb, nb, b + j + i * k, k, pb);

            for (p = 0; p < m; p += mc) {
                i64 mb = std::min(mc, m - p);

                dgemm_itcopy(kb, mb, a + p + j * m, m, pa);
                dgemm_kernel(mb, nb, kb, pa, pb, c + p + i * m, m);
            }
        }
    }

    aligned_free(pa);
    aligned_free(pb);
}

size_t align(size_t size, size_t alignment)
{
    assert((size_t)pow((size_t)log2(alignment), 2) == alignment);
    return (size + alignment - 1) & ~(alignment - 1);
}

bool compare(double *a, double *b, size_t len)
{
    constexpr double eps = 1e-7;

    for (size_t i = 0; i < len; i++) {
        // relative error
        if ((b[i] - a[i]) > eps * std::abs(a[i] + b[i])) {
            cout << "compare failed, i: " << i << ", left = " << a[i] << ", right = " << b[i] << endl;
            return false;
        }
    }
    return true;
}

// for test
int main(int argc, const char *argv[])
{
    i64 m, n, k;
    m = n = k = 1024;

    omp_set_num_threads(1);

    i64 *p[] = { &m, &n, &k };

    for (int i = 0; i < std::min(argc - 1, 3); i++) {
        *(p[i]) = std::atol(argv[i + 1]);
    }

    cout << "m = " << m << ", n = " << n << ", k = " << k << endl;

    double *a = (double *)aligned_malloc(align(m * k * sizeof(double), mem_alignment), mem_alignment);
    double *b = (double *)aligned_malloc(align(n * k * sizeof(double), mem_alignment), mem_alignment);
    double *c1 = (double *)aligned_malloc(align(m * n * sizeof(double), mem_alignment), mem_alignment);
    double *c2 = (double *)aligned_malloc(align(m * n * sizeof(double), mem_alignment), mem_alignment);

    random_arr(a, m * k);
    random_arr(b, n * k);
    random_arr(c1, m * n);
    memcpy(c2, c1, m * n * sizeof(double));

    // warm up
    if constexpr (dgemm_check) {
        ref_dgemm(m, n, k, a, b, c1);
        opt_dgemm(m, n, k, a, b, c2);
        if (!compare(c1, c2, m * n)) {
            aligned_free(a);
            aligned_free(b);
            aligned_free(c1);
            aligned_free(c2);
            return 1;
        }
    }

    Timer timer;
    if constexpr (dgemm_check) {
        ref_dgemm(m, n, k, a, b, c1);
        timer.tick();
        ref_dgemm(m, n, k, a, b, c1);
        double ref = timer.tock();
        cout << "ref gf: " << 2 * m * n * k / (ref * 1e+9)  << endl;
        opt_dgemm(m, n, k, a, b, c2);
    }

    kernel_cost = 0.0;
    timer.tick();
    opt_dgemm(m, n, k, a, b, c2);
    double opt = timer.tock();

    double opt_gf = 2 * m * n * k / (opt * 1e+9);
    double opt_kernel_gf = 2 * m * n * k / (kernel_cost * 1e+9);

    cout << "my gemm gf: " << opt_gf << endl;

    cout << "cp usage = " << opt_gf * 100 / gf_limit << "%" << endl;
    cout << "kernel cp usage = " << opt_kernel_gf * 100 / gf_limit << "%" << endl;

    aligned_free(a);
    aligned_free(b);
    aligned_free(c1);
    aligned_free(c2);
}
