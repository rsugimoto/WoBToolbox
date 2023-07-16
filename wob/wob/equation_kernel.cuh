/*
    Walk-on-Boundary Toolbox
    This file is a part of the public release of the Walk-on-Boundary (WoB) Toolbox.

    Author:         Ryusuke Sugimoto
    Affiliation:    University of Waterloo
    Date:           July 2023
    File Name:      equation_kernel.cuh
    Description:    This file defines the kernel functions for WoB.
*/

#ifndef __WOB_EQUATION_KERNEL_CUH__
#define __WOB_EQUATION_KERNEL_CUH__

#include <Eigen/Core>
#include <string>

namespace wob {

enum EquationKernelType { PoissonKernel, EquationKernelTypeCount };

template <EquationKernelType T> constexpr __host__ __device__ bool is_vector_problem();
template <> constexpr __host__ __device__ bool is_vector_problem<PoissonKernel>() { return false; };

EquationKernelType string_to_equation_kernel_type(const std::string &str) {
    if (str == "PoissonKernel") return PoissonKernel;
    return EquationKernelTypeCount;
}
std::string equation_kernel_type_to_string(const EquationKernelType equation_kernel) {
    switch (equation_kernel) {
    case PoissonKernel: return "PoissonKernel";
    default: return "InvalidEqiuationKernelType";
    }
}

template <typename ScalarType, unsigned int Dim, bool IsVectorProblemm> struct kernel_type;
template <typename ScalarType, unsigned int Dim> struct kernel_type<ScalarType, Dim, true> {
    using type = Eigen::Matrix<ScalarType, Dim, Dim>;
};
template <typename ScalarType, unsigned int Dim> struct kernel_type<ScalarType, Dim, false> {
    using type = ScalarType;
};

template <typename ScalarType, unsigned int Dim, EquationKernelType T> class EquationKernel {
  public:
    using VectorDs = Eigen::Matrix<ScalarType, Dim, 1>;

    inline __host__ __device__ EquationKernel() : eps(std::numeric_limits<ScalarType>::epsilon()){};
    inline __host__ __device__ EquationKernel(ScalarType eps) : eps(eps){};

    inline __host__ __device__ ScalarType G(const VectorDs &y, const VectorDs &x) const;

    inline __host__ __device__ VectorDs dG_dx(const VectorDs &y, const VectorDs &x) const;

    inline __host__ __device__ ScalarType dG_dny(const VectorDs &y, const VectorDs &x, const VectorDs &ny) const;

    inline __host__ __device__ VectorDs d2G_dxdny(const VectorDs &y, const VectorDs &x, const VectorDs &ny) const;

    inline __host__ __device__ ScalarType
    d2G_dnxdny(const VectorDs &y, const VectorDs &x, const VectorDs &ny, const VectorDs &nx) const;

    const ScalarType eps;
};

// -------------------- 2D Poisson (Begin) --------------------

template <typename ScalarType> class EquationKernel<ScalarType, 2, PoissonKernel> {
  public:
    using Vector2s = Eigen::Matrix<ScalarType, 2, 1>;

    inline __host__ __device__ EquationKernel() : eps(std::numeric_limits<ScalarType>::epsilon()){};
    inline __host__ __device__ EquationKernel(ScalarType eps) : eps(eps){};

    inline __host__ __device__ ScalarType G(const Vector2s &y, const Vector2s &x) const {
        return -1.0f / (2.0f * (ScalarType)M_PI) * std::log(std::max((y - x).norm(), eps));
    };

    inline __host__ __device__ Vector2s dG_dx(const Vector2s &y, const Vector2s &x) const {
        const Vector2s r = y - x;
        return 1.0f / (2.0f * (ScalarType)M_PI) * r.normalized() / std::max(r.norm(), eps);
    };

    inline __host__ __device__ ScalarType dG_dny(const Vector2s &y, const Vector2s &x, const Vector2s &ny) const {
        const Vector2s r = y - x;
        return -1.0f / (2.0f * (ScalarType)M_PI) * r.normalized().dot(ny) / std::max(r.norm(), eps);
    };

    inline __host__ __device__ Vector2s d2G_dxdny(const Vector2s &y, const Vector2s &x, const Vector2s &ny) const {
        const Vector2s r = y - x;
        const Vector2s r_hat = r.normalized();
        return 1.0f / (2.0f * (ScalarType)M_PI * std::max(r.squaredNorm(), eps)) * (ny - 2.0f * r_hat.dot(ny) * r_hat);
    };

    inline __host__ __device__ ScalarType
    d2G_dnxdny(const Vector2s &y, const Vector2s &x, const Vector2s &ny, const Vector2s &nx) const {
        const Vector2s r = y - x;
        const Vector2s r_hat = r.normalized();
        return 1.0f / (2.0f * (ScalarType)M_PI * std::max(r.squaredNorm(), eps)) *
               (nx.dot(ny) - 2.0f * r_hat.dot(nx) * r_hat.dot(ny));
    }

    ScalarType eps;
};

// -------------------- 2D Poisson (End) --------------------

// -------------------- 3D Poisson (Begin) --------------------

template <typename ScalarType> class EquationKernel<ScalarType, 3, PoissonKernel> {
  public:
    using Vector3s = Eigen::Matrix<ScalarType, 3, 1>;
    inline __host__ __device__ EquationKernel() : eps(std::numeric_limits<ScalarType>::epsilon()){};
    inline __host__ __device__ EquationKernel(ScalarType eps) : eps(eps){};

    inline __host__ __device__ ScalarType G(const Vector3s &y, const Vector3s &x) const {
        return 1.0f / (4.0f * (ScalarType)M_PI * std::max((y - x).norm(), eps));
    };

    inline __host__ __device__ Vector3s dG_dx(const Vector3s &y, const Vector3s &x) const {
        const Vector3s r = y - x;
        return 1.0f / (4.0f * (ScalarType)M_PI) * r.normalized() / std::max(r.squaredNorm(), eps);
    };

    inline __host__ __device__ ScalarType dG_dny(const Vector3s &y, const Vector3s &x, const Vector3s &ny) const {
        const Vector3s r = y - x;
        return -1.0f / (4.0f * (ScalarType)M_PI) * r.normalized().dot(ny) / std::max(r.squaredNorm(), eps);
    };

    inline __host__ __device__ Vector3s d2G_dxdny(const Vector3s &y, const Vector3s &x, const Vector3s &ny) const {
        const Vector3s r = y - x;
        const Vector3s r_hat = r.normalized();
        return 1.0f / (4.0f * (ScalarType)M_PI * std::max(r.norm() * r.squaredNorm(), eps)) *
               (ny - 3.0f * r_hat.dot(ny) * r_hat);
    };

    inline __host__ __device__ ScalarType
    d2G_dnxdny(const Vector3s &y, const Vector3s &x, const Vector3s &ny, const Vector3s &nx) const {
        const Vector3s r = y - x;
        const Vector3s r_hat = r.normalized();
        return 1.0f / (4.0f * (ScalarType)M_PI * std::max(r.norm() * r.squaredNorm(), eps)) *
               (nx.dot(ny) - 3.0f * r_hat.dot(nx) * r_hat.dot(ny));
    }

    ScalarType eps;
};

// -------------------- 3D Poisson (End) --------------------

} // namespace wob

#endif // __WOB_EQUATION_KERNEL_CUH__
