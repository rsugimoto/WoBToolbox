/*
    Walk-on-Boundary Toolbox
    This file is a part of the public release of the Walk-on-Boundary (WoB) Toolbox.

    Author:         Ryusuke Sugimoto
    Affiliation:    University of Waterloo
    Date:           July 2023
    File Name:      utils.cuh
    Description:    This file defines a few utility functions used internally in the WoB Toolbox.
*/

#ifndef __WOB_UTILS_CUH__
#define __WOB_UTILS_CUH__

#include <Eigen/Core>

#ifdef __CUDACC__
#include <curand_kernel.h>
#else
#include <random>
#endif

namespace wob {

#ifdef __CUDACC__
using randomState_t = curandState_t;
#else
using randomState_t = std::mt19937;
#endif

namespace utils {

#ifdef __CUDACC__
template <typename T> inline __device__ T rand_uniform(wob::randomState_t *random_state_ptr) noexcept;
template <> inline __device__ float rand_uniform<float>(wob::randomState_t *random_state_ptr) noexcept {
    return curand_uniform(random_state_ptr);
}
template <> inline __device__ double rand_uniform<double>(wob::randomState_t *random_state_ptr) noexcept {
    return curand_uniform_double(random_state_ptr);
}

template <typename T> inline __device__ T rand_normal(wob::randomState_t *random_state_ptr) noexcept;
template <> inline __device__ float rand_normal<float>(wob::randomState_t *random_state_ptr) noexcept {
    return curand_normal(random_state_ptr);
}
template <> inline __device__ double rand_normal<double>(wob::randomState_t *random_state_ptr) noexcept {
    return curand_normal_double(random_state_ptr);
}
#else
template <typename T> inline __device__ T rand_uniform(wob::randomState_t *rand_state) noexcept;
template <> inline __device__ float rand_uniform<float>(wob::randomState_t *rand_state) noexcept {
    return 1.0f - std::uniform_real_distribution<float>()(*rand_state);
}
template <> inline __device__ double rand_uniform<double>(wob::randomState_t *rand_state) noexcept {
    return 1.0 - std::uniform_real_distribution<double>()(*rand_state);
}

template <typename T> inline __device__ T rand_normal(wob::randomState_t *rand_state) noexcept;
template <> inline __device__ float rand_normal<float>(wob::randomState_t *rand_state) noexcept {
    return std::normal_distribution<float>()(*rand_state);
}
template <> inline __device__ double rand_normal<double>(wob::randomState_t *rand_state) noexcept {
    return std::normal_distribution<double>()(*rand_state);
}
#endif

template <unsigned int Dim>
inline __host__ __device__ Eigen::Matrix<unsigned int, Dim, 1>
unflatten(unsigned int idx, unsigned int grid_res) noexcept;
template <>
inline __host__ __device__ Eigen::Matrix<unsigned int, 3, 1>
unflatten<3>(unsigned int idx, unsigned int grid_res) noexcept {
    return Eigen::Matrix<unsigned int, 3, 1>(idx % grid_res, (idx / grid_res) % grid_res, idx / (grid_res * grid_res));
}
template <>
inline __host__ __device__ Eigen::Matrix<unsigned int, 2, 1>
unflatten<2>(unsigned int idx, unsigned int grid_res) noexcept {
    return Eigen::Matrix<unsigned int, 2, 1>(idx % grid_res, idx / grid_res);
}

// returns a point in (-domain_size/2, domain_size/2)^Dim
template <typename ScalarType, unsigned int Dim>
inline __host__ __device__ Eigen::Matrix<ScalarType, Dim, 1>
idx_to_domain_point(unsigned int idx, unsigned int grid_res, ScalarType domain_size) noexcept {
    const ScalarType dx = domain_size / grid_res;
    Eigen::Matrix<unsigned int, Dim, 1> x = unflatten<Dim>(idx, grid_res);
    return (x.template cast<ScalarType>() + (ScalarType)0.5 * Eigen::Matrix<ScalarType, Dim, 1>::Ones()) * dx -
           (domain_size / 2) * Eigen::Matrix<ScalarType, Dim, 1>::Ones();
}

template <typename T> inline __host__ __device__ constexpr T zero();
template <> inline __host__ __device__ constexpr float zero<float>() { return 0.0f; }
template <> inline __host__ __device__ constexpr double zero<double>() { return 0.0; }
template <> inline __host__ __device__ Eigen::Vector2f zero<Eigen::Vector2f>() { return Eigen::Vector2f::Zero(); }
template <> inline __host__ __device__ Eigen::Vector2d zero<Eigen::Vector2d>() { return Eigen::Vector2d::Zero(); }
template <> inline __host__ __device__ Eigen::Vector3f zero<Eigen::Vector3f>() { return Eigen::Vector3f::Zero(); }
template <> inline __host__ __device__ Eigen::Vector3d zero<Eigen::Vector3d>() { return Eigen::Vector3d::Zero(); }
template <> inline __host__ __device__ Eigen::Vector4f zero<Eigen::Vector4f>() { return Eigen::Vector4f::Zero(); }
template <> inline __host__ __device__ Eigen::Vector4d zero<Eigen::Vector4d>() { return Eigen::Vector4d::Zero(); }
template <> inline __host__ __device__ Eigen::Matrix2f zero<Eigen::Matrix2f>() { return Eigen::Matrix2f::Zero(); }
template <> inline __host__ __device__ Eigen::Matrix2d zero<Eigen::Matrix2d>() { return Eigen::Matrix2d::Zero(); }
template <> inline __host__ __device__ Eigen::Matrix3f zero<Eigen::Matrix3f>() { return Eigen::Matrix3f::Zero(); }
template <> inline __host__ __device__ Eigen::Matrix3d zero<Eigen::Matrix3d>() { return Eigen::Matrix3d::Zero(); }

template <typename T> inline __host__ __device__ T one();
template <> inline __host__ __device__ constexpr float one<float>() { return 1.0f; }
template <> inline __host__ __device__ constexpr double one<double>() { return 1.0; }
template <> inline __host__ __device__ Eigen::Vector2f one<Eigen::Vector2f>() { return Eigen::Vector2f::Ones(); }
template <> inline __host__ __device__ Eigen::Vector2d one<Eigen::Vector2d>() { return Eigen::Vector2d::Ones(); }
template <> inline __host__ __device__ Eigen::Vector3f one<Eigen::Vector3f>() { return Eigen::Vector3f::Ones(); }
template <> inline __host__ __device__ Eigen::Vector3d one<Eigen::Vector3d>() { return Eigen::Vector3d::Ones(); }
template <> inline __host__ __device__ Eigen::Vector4f one<Eigen::Vector4f>() { return Eigen::Vector4f::Ones(); }
template <> inline __host__ __device__ Eigen::Vector4d one<Eigen::Vector4d>() { return Eigen::Vector4d::Ones(); }
template <> inline __host__ __device__ Eigen::Matrix2f one<Eigen::Matrix2f>() { return Eigen::Matrix2f::Identity(); }
template <> inline __host__ __device__ Eigen::Matrix2d one<Eigen::Matrix2d>() { return Eigen::Matrix2d::Identity(); }
template <> inline __host__ __device__ Eigen::Matrix3f one<Eigen::Matrix3f>() { return Eigen::Matrix3f::Identity(); }
template <> inline __host__ __device__ Eigen::Matrix3d one<Eigen::Matrix3d>() { return Eigen::Matrix3d::Identity(); }

// Compensated sum algorithm by Kahan
template <typename T> struct KahanSum {
    T sum, c;
    inline __host__ __device__ KahanSum() {
        sum = zero<T>();
        c = zero<T>();
    };
    inline __host__ __device__ KahanSum(const KahanSum &other) {
        this->sum = other.sum;
        this->c = other.c;
    }
    inline __host__ __device__ KahanSum<T> &operator+=(const T &value) {
        T y = value - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
        return *this;
    };
    inline __host__ __device__ operator T() const { return sum; }
};

} // namespace utils

} // namespace wob

#endif // __WOB_UTILS_CUH__