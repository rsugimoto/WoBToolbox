/*
    Walk-on-Boundary Toolbox
    This file is a part of the public release of the Walk-on-Boundary (WoB) Toolbox.

    Author:         Ryusuke Sugimoto
    Affiliation:    University of Waterloo
    Date:           July 2023
    File Name:      bie_models.cuh
    Description:    This file defines different types of BIE used for WoB and how some of the terms should be sampled..
*/

#ifndef __WOB_BIE_MODELS_CUH__
#define __WOB_BIE_MODELS_CUH__

#include "equation_kernel.cuh"
#include "points.hpp"
#include "utils.cuh"

namespace wob {

enum BIEModelType { IndirectSingleLayer, IndirectDoubleLayer, Direct, BIEModelTypeCount };
enum DomainType { InteriorDomain, ExteriorDomain, DomainTypeCount };

BIEModelType string_to_bie_model_type(const std::string &str) {
    if (str == "IndirectSingleLayer") return IndirectSingleLayer;
    if (str == "IndirectDoubleLayer") return IndirectDoubleLayer;
    if (str == "Direct") return Direct;
    return BIEModelTypeCount;
}
std::string bie_model_type_to_string(const BIEModelType bie_model_type) {
    switch (bie_model_type) {
    case IndirectSingleLayer: return "IndirectSingleLayer";
    case IndirectDoubleLayer: return "IndirectDoubleLayer";
    case Direct: return "Direct";
    default: return "InvalidBIEModelType";
    }
}

DomainType string_to_domain_type(const std::string &str) {
    if (str == "InteriorDomain") return InteriorDomain;
    if (str == "ExteriorDomain") return ExteriorDomain;
    return DomainTypeCount;
}

std::string domain_type_to_string(const DomainType domain_type) {
    switch (domain_type) {
    case InteriorDomain: return "InteriorDomain";
    case ExteriorDomain: return "ExteriorDomain";
    default: return "InvalidDomainType";
    }
}

template <typename ScalarType, unsigned int Dim, EquationKernelType equationKernel, DomainType domainType>
class BIEModelBase {
  public:
    using BoundaryPointXs = BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()>;
    using DomainPointXs = DomainPoint<ScalarType, Dim>;
    using ValueType = typename value_type<ScalarType, Dim, is_vector_problem<equationKernel>()>::type;

    inline __host__ __device__ BIEModelBase() : equation_kernel(), first_kind_equation_scaling_constant(1.0f){};
    inline __host__ __device__ BIEModelBase(ScalarType eps)
        : equation_kernel(eps), first_kind_equation_scaling_constant(1.0f){};
    inline __host__ __device__ BIEModelBase(ScalarType eps, ScalarType first_kind_equation_scaling_constant)
        : equation_kernel(eps), first_kind_equation_scaling_constant(first_kind_equation_scaling_constant){};

    constexpr inline __device__ ScalarType domain_type_coefficient() const {
        if constexpr (domainType == InteriorDomain)
            return 1.0;
        else // if constexpr (domainType == ExteriorDomain)
            return -1.0;
    }

    constexpr inline __device__ ScalarType integral_free_term(const ScalarType interior_angle) const {
        if constexpr (domainType == InteriorDomain)
            return interior_angle;
        else // if constexpr (domainType == ExteriorDomain)
            return 1.f - interior_angle;
    }
    template <class Func>
    inline __device__ auto estimate_volume_term(
        const BoundaryPointXs &boundary_point, unsigned int num_volume_samples, Func volume_sample_sampler
    ) const {
        if (num_volume_samples == 0) return utils::zero<ValueType>();

        auto [volume_sample_point, inv_pdf] = volume_sample_sampler();
        ValueType result = utils::zero<ValueType>();
        for (unsigned int i = 0; i < num_volume_samples; i++) {
            if (boundary_point.boundary_type == Dirichlet) {
                result +=
                    equation_kernel.G(boundary_point.p, volume_sample_point.p) * volume_sample_point.value * inv_pdf;
            } else if (boundary_point.boundary_type == Neumann) {
                result += equation_kernel.dG_dny(boundary_point.p, volume_sample_point.p, boundary_point.n) *
                          volume_sample_point.value * inv_pdf;
            } else { // if (boundary_point.boundary_type == Robin)
                result += (equation_kernel.dG_dny(boundary_point.p, volume_sample_point.p, boundary_point.n) +
                           boundary_point.robin_alpha * equation_kernel.G(volume_sample_point.p, boundary_point.p)) *
                          volume_sample_point.value * inv_pdf;
            }
        }

        result = result / num_volume_samples;
        return result;
    }

    template <class Func>
    inline __device__ auto estimate_volume_term_init(
        const BoundaryPointXs &boundary_point, unsigned int num_volume_samples, Func volume_sample_sampler
    ) const {
        if (num_volume_samples == 0) return utils::zero<ValueType>();

        auto [volume_sample_point, inv_pdf] = volume_sample_sampler();
        ValueType result = utils::zero<ValueType>();
        for (unsigned int i = 0; i < num_volume_samples; i++) {
            result -= equation_kernel.G(boundary_point.p, volume_sample_point.p) * volume_sample_point.value * inv_pdf;
        }

        result = result / num_volume_samples;
        return result;
    }

    template <class Func>
    inline __device__ auto estimate_volume_term_init(
        const DomainPointXs &domain_point, unsigned int num_volume_samples, Func volume_sample_sampler
    ) const {
        if (num_volume_samples == 0) return utils::zero<ValueType>();

        auto [volume_sample_point, inv_pdf] = volume_sample_sampler();
        ValueType result = utils::zero<ValueType>();
        for (unsigned int i = 0; i < num_volume_samples; i++) {
            result -= equation_kernel.G(volume_sample_point.p, domain_point.p) * volume_sample_point.value * inv_pdf;
        }

        result = result / num_volume_samples;
        return result;
    }

    template <class Func>
    inline __device__ auto estimate_volume_term_init_derivative(
        const DomainPointXs &domain_point, unsigned int num_volume_samples, Func volume_sample_sampler
    ) const {
        using ResultType = Eigen::Matrix<ScalarType, Dim, is_vector_problem<equationKernel>() ? Dim : 1>;
        if (num_volume_samples == 0) return utils::zero<ResultType>();

        auto [volume_sample_point, inv_pdf] = volume_sample_sampler();

        ResultType result = ResultType::Zero();
        for (unsigned int i = 0; i < num_volume_samples; i++) {
            result -=
                equation_kernel.dG_dx(volume_sample_point.p, domain_point.p) * volume_sample_point.value * inv_pdf;
        }

        result = result / num_volume_samples;
        return result;
    }

    EquationKernel<ScalarType, Dim, equationKernel> equation_kernel;
    const ScalarType first_kind_equation_scaling_constant;
};

template <
    typename ScalarType, unsigned int Dim, EquationKernelType equationKernel, BIEModelType bieModel,
    DomainType domainType>
class BIEModel : public BIEModelBase<ScalarType, Dim, equationKernel, domainType> {
  public:
    using BoundaryPointXs = BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()>;
    using DomainPointXs = DomainPoint<ScalarType, Dim>;
    inline __host__ __device__ BIEModel() : BIEModelBase<ScalarType, Dim, equationKernel, domainType>(){};
    inline __host__ __device__ BIEModel(ScalarType eps)
        : BIEModelBase<ScalarType, Dim, equationKernel, domainType>(eps){};
    inline __host__ __device__ BIEModel(ScalarType eps, ScalarType first_kind_equation_scaling_constant)
        : BIEModelBase<ScalarType, Dim, equationKernel, domainType>(eps, first_kind_equation_scaling_constant){};

    inline __device__ auto
    solution_domain_unknown(const DomainPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf) const;
    template <class Func>
    inline __device__ auto solution_domain_known(
        const DomainPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const;

    inline __device__ auto
    gradient_domain_unknown(const DomainPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf) const;
    template <class Func>
    inline __device__ auto gradient_domain_known(
        const DomainPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const;

    inline __device__ auto
    solution_boundary_unknown(const BoundaryPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf) const;
    template <class Func>
    inline __device__ auto solution_boundary_known(
        const BoundaryPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const;

    inline __device__ auto fredholm_equation_boundary_unknown(
        BoundaryPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf, wob::randomState_t *random_state_ptr,
        const bool is_backward_sampling
    ) const;
    template <class Func>
    inline __device__ auto fredholm_equation_boundary_known(
        const BoundaryPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const;
};

// -------------------- Indirect Sigle Layer (Begin) --------------------
template <typename ScalarType, unsigned int Dim, EquationKernelType equationKernel, DomainType domainType>
class BIEModel<ScalarType, Dim, equationKernel, IndirectSingleLayer, domainType>
    : public BIEModelBase<ScalarType, Dim, equationKernel, domainType> {
  public:
    inline __host__ __device__ BIEModel() : BIEModelBase<ScalarType, Dim, equationKernel, domainType>(){};
    inline __host__ __device__ BIEModel(ScalarType eps)
        : BIEModelBase<ScalarType, Dim, equationKernel, domainType>(eps){};
    inline __host__ __device__ BIEModel(ScalarType eps, ScalarType first_kind_equation_scaling_constant)
        : BIEModelBase<ScalarType, Dim, equationKernel, domainType>(eps, first_kind_equation_scaling_constant){};

    using BoundaryPointXs = BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()>;
    using DomainPointXs = DomainPoint<ScalarType, Dim>;
    using KernelType = typename kernel_type<ScalarType, Dim, is_vector_problem<equationKernel>()>::type;
    using ValueType = typename value_type<ScalarType, Dim, is_vector_problem<equationKernel>()>::type;
    using BIEModelBase<ScalarType, Dim, equationKernel, domainType>::equation_kernel;
    using BIEModelBase<ScalarType, Dim, equationKernel, domainType>::first_kind_equation_scaling_constant;
    using BIEModelBase<ScalarType, Dim, equationKernel, domainType>::domain_type_coefficient;
    using BIEModelBase<ScalarType, Dim, equationKernel, domainType>::integral_free_term;

    inline __device__ KernelType
    solution_domain_unknown(const DomainPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf) const {
        return equation_kernel.G(y.p, x.p) * inv_pdf;
    }
    template <class Func>
    inline __device__ ValueType solution_domain_known(
        const DomainPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const {
        return this->estimate_volume_term_init(x, num_volume_samples, volume_sample_sampler);
    };

    inline __device__ auto
    gradient_domain_unknown(const DomainPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf) const {
        return (equation_kernel.dG_dx(y.p, x.p) * inv_pdf).eval();
    }
    template <class Func>
    inline __device__ auto gradient_domain_known(
        const DomainPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const {
        return this->estimate_volume_term_init_derivative(x, num_volume_samples, volume_sample_sampler);
    }

    inline __device__ KernelType
    solution_boundary_unknown(const BoundaryPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf) const {
        if (x.boundary_type == Dirichlet) return utils::zero<KernelType>();

        return equation_kernel.G(y.p, x.p) * inv_pdf;
    }
    template <class Func>
    inline __device__ ValueType solution_boundary_known(
        const BoundaryPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const {
        if (x.boundary_type == Dirichlet) return x.boundary_value;

        return this->estimate_volume_term_init(x, num_volume_samples, volume_sample_sampler);
    }

    inline __device__ KernelType fredholm_equation_boundary_unknown(
        BoundaryPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf, wob::randomState_t *random_state_ptr,
        const bool is_backward_sampling
    ) const {
        if (is_backward_sampling && x.boundary_type == Dirichlet) {
            if (utils::rand_uniform<ScalarType>(random_state_ptr) * (1.0f + first_kind_equation_scaling_constant) <
                1.0f) {
                y = x;
                return (1.0f + first_kind_equation_scaling_constant) * utils::one<KernelType>();
            } else {
                inv_pdf *= (1.0f + first_kind_equation_scaling_constant) / first_kind_equation_scaling_constant;
            }
        } else if (!is_backward_sampling && y.boundary_type == Dirichlet) {
            if (utils::rand_uniform<ScalarType>(random_state_ptr) * (1.0f + first_kind_equation_scaling_constant) <
                1.0f) {
                x = y;
                return (1.0f + first_kind_equation_scaling_constant) * utils::one<KernelType>();
            } else {
                inv_pdf *= (1.0f + first_kind_equation_scaling_constant) / first_kind_equation_scaling_constant;
            }
        }

        if (x.boundary_type == Dirichlet) {
            return -2.f * first_kind_equation_scaling_constant * inv_pdf * equation_kernel.G(y.p, x.p);
        } else if (x.boundary_type == Neumann) {
            return -domain_type_coefficient() / integral_free_term(x.interior_angle) * inv_pdf *
                   equation_kernel.dG_dny(x.p, y.p, x.n);
        } else /* if (x.boundary_type == Robin) */
        {
            return -domain_type_coefficient() / integral_free_term(x.interior_angle) * inv_pdf *
                   (equation_kernel.dG_dny(x.p, y.p, x.n) + x.robin_alpha * equation_kernel.G(y.p, x.p));
        }
    }
    template <class Func>
    inline __device__ ValueType fredholm_equation_boundary_known(
        const BoundaryPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const {
        if (x.boundary_type == Dirichlet) {
            return 2.f * first_kind_equation_scaling_constant *
                   (x.boundary_value + this->estimate_volume_term(x, num_volume_samples, volume_sample_sampler));
        } else if (x.boundary_type == Neumann) {
            return domain_type_coefficient() / integral_free_term(x.interior_angle) *
                   (x.boundary_value + this->estimate_volume_term(x, num_volume_samples, volume_sample_sampler));
        } else /* if (x.boundary_type == Robin) */ {
            return domain_type_coefficient() / integral_free_term(x.interior_angle) *
                   (x.boundary_value + this->estimate_volume_term(x, num_volume_samples, volume_sample_sampler));
        }
    }
};

// -------------------- Indirect Sigle Layer (End) --------------------

// -------------------- Indirect Double Layer (Begin) --------------------
// This implementaiton does not support the Dirichlet exterior domain problems in general cases.
template <typename ScalarType, unsigned int Dim, EquationKernelType equationKernel, DomainType domainType>
class BIEModel<ScalarType, Dim, equationKernel, IndirectDoubleLayer, domainType>
    : public BIEModelBase<ScalarType, Dim, equationKernel, domainType> {
  public:
    inline __host__ __device__ BIEModel() : BIEModelBase<ScalarType, Dim, equationKernel, domainType>(){};
    inline __host__ __device__ BIEModel(ScalarType eps)
        : BIEModelBase<ScalarType, Dim, equationKernel, domainType>(eps){};
    inline __host__ __device__ BIEModel(ScalarType eps, ScalarType first_kind_equation_scaling_constant)
        : BIEModelBase<ScalarType, Dim, equationKernel, domainType>(eps, first_kind_equation_scaling_constant){};

    using BoundaryPointXs = BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()>;
    using DomainPointXs = DomainPoint<ScalarType, Dim>;
    using KernelType = typename kernel_type<ScalarType, Dim, is_vector_problem<equationKernel>()>::type;
    using ValueType = typename value_type<ScalarType, Dim, is_vector_problem<equationKernel>()>::type;
    using BIEModelBase<ScalarType, Dim, equationKernel, domainType>::equation_kernel;
    using BIEModelBase<ScalarType, Dim, equationKernel, domainType>::domain_type_coefficient;
    using BIEModelBase<ScalarType, Dim, equationKernel, domainType>::integral_free_term;

    inline __device__ KernelType
    solution_domain_unknown(const DomainPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf) const {
        return -equation_kernel.dG_dny(y.p, x.p, y.n) * inv_pdf;
    }
    template <class Func>
    inline __device__ ValueType solution_domain_known(
        const DomainPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const {
        return this->estimate_volume_term_init(x, num_volume_samples, volume_sample_sampler);
    };

    inline __device__ auto
    gradient_domain_unknown(const DomainPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf) const {
        return (-equation_kernel.d2G_dxdny(y.p, x.p, y.n) * inv_pdf).eval();
    }
    template <class Func>
    inline __device__ auto gradient_domain_known(
        const DomainPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const {
        return this->estimate_volume_term_init_derivative(x, num_volume_samples, volume_sample_sampler);
    }

    inline __device__ KernelType
    solution_boundary_unknown(const BoundaryPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf) const {
        // Assumes Dirichlet problem
        return utils::zero<KernelType>();
    }
    template <class Func>
    inline __device__ ValueType solution_boundary_known(
        const BoundaryPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const {
        // Assumes Dirichlet problem
        return x.boundary_value;
    }

    inline __device__ KernelType fredholm_equation_boundary_unknown(
        BoundaryPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf, wob::randomState_t *random_state_ptr,
        const bool is_backward_sampling
    ) const {
        assert(x.boundary_type == Dirichlet); // Only Dirichlet boundary is supported with the double layer formulation.
                                              // This is to avoid hypersingular integrals.

        return domain_type_coefficient() / (1.f - integral_free_term(x.interior_angle)) * inv_pdf *
               equation_kernel.dG_dny(y.p, x.p, y.n);
    }
    template <class Func>
    inline __device__ ValueType fredholm_equation_boundary_known(
        const BoundaryPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const {
        assert(x.boundary_type == Dirichlet); // Only Dirichlet boundary is supported with the double layer formulation.
                                              // This is to avoid hypersingular integrals.

        return domain_type_coefficient() * 1.f / (1.f - integral_free_term(x.interior_angle)) *
               (x.boundary_value + this->estimate_volume_term(x, num_volume_samples, volume_sample_sampler));
    }
};

// -------------------- Indirect Double Layer (End) --------------------

// -------------------- Direct (Begin) --------------------

// The sign flip for exterior problems need to be handled carefully. After flipping the sign, some are cancelled to
// further flip the sign of the first kind equaiton constant.
template <typename ScalarType, unsigned int Dim, EquationKernelType equationKernel, DomainType domainType>
class BIEModel<ScalarType, Dim, equationKernel, Direct, domainType>
    : public BIEModelBase<ScalarType, Dim, equationKernel, domainType> {
  public:
    inline __host__ __device__ BIEModel() : BIEModelBase<ScalarType, Dim, equationKernel, domainType>(){};
    inline __host__ __device__ BIEModel(ScalarType eps)
        : BIEModelBase<ScalarType, Dim, equationKernel, domainType>(eps){};
    inline __host__ __device__ BIEModel(ScalarType eps, ScalarType first_kind_equation_scaling_constant)
        : BIEModelBase<ScalarType, Dim, equationKernel, domainType>(eps, first_kind_equation_scaling_constant){};

    using BoundaryPointXs = BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()>;
    using DomainPointXs = DomainPoint<ScalarType, Dim>;
    using KernelType = typename kernel_type<ScalarType, Dim, is_vector_problem<equationKernel>()>::type;
    using ValueType = typename value_type<ScalarType, Dim, is_vector_problem<equationKernel>()>::type;
    using BIEModelBase<ScalarType, Dim, equationKernel, domainType>::equation_kernel;
    using BIEModelBase<ScalarType, Dim, equationKernel, domainType>::first_kind_equation_scaling_constant;
    using BIEModelBase<ScalarType, Dim, equationKernel, domainType>::domain_type_coefficient;
    using BIEModelBase<ScalarType, Dim, equationKernel, domainType>::integral_free_term;

    inline __device__ KernelType
    solution_domain_unknown(const DomainPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf) const {
        if (y.boundary_type == Dirichlet)
            return domain_type_coefficient() * inv_pdf * equation_kernel.G(y.p, x.p);
        else if (y.boundary_type == Neumann)
            return -domain_type_coefficient() * inv_pdf * equation_kernel.dG_dny(y.p, x.p, y.n);
        else // if (y.boundary_type == Robin)
            return -domain_type_coefficient() * inv_pdf *
                   (equation_kernel.dG_dny(y.p, x.p, y.n) + y.robin_alpha * equation_kernel.G(y.p, x.p));
    }
    template <class Func>
    inline __device__ ValueType solution_domain_known(
        const DomainPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const {
        ValueType result = this->estimate_volume_term_init(x, num_volume_samples, volume_sample_sampler);
        if (y.boundary_type == Dirichlet)
            result += -domain_type_coefficient() * inv_pdf * equation_kernel.dG_dny(y.p, x.p, y.n) *
                      (y.boundary_value + this->estimate_volume_term(y, num_volume_samples, volume_sample_sampler));
        else // if (y.boundary_type == Neumann || y.boundary_type == Robin)
            result += domain_type_coefficient() * inv_pdf * equation_kernel.G(y.p, x.p) *
                      (y.boundary_value + this->estimate_volume_term(y, num_volume_samples, volume_sample_sampler));
        return result;
    };

    inline __device__ auto
    gradient_domain_unknown(const DomainPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf) const {
        if (y.boundary_type == Dirichlet)
            return (domain_type_coefficient() * inv_pdf * equation_kernel.dG_dx(y.p, x.p)).eval();
        else if (y.boundary_type == Neumann)
            return (-domain_type_coefficient() * inv_pdf * equation_kernel.d2G_dxdny(y.p, x.p, y.n)).eval();
        else // if (y.boundary_type == Robin)
            return (-domain_type_coefficient() * inv_pdf *
                    (equation_kernel.d2G_dxdny(y.p, x.p, y.n) + y.robin_alpha * equation_kernel.dG_dx(y.p, x.p)))
                .eval();
    }
    template <class Func>
    inline __device__ auto gradient_domain_known(
        const DomainPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const {
        auto result = this->estimate_volume_term_init_derivative(x, num_volume_samples, volume_sample_sampler);
        if (y.boundary_type == Dirichlet)
            result += (-domain_type_coefficient() * inv_pdf * equation_kernel.d2G_dxdny(y.p, x.p, y.n) *
                       (y.boundary_value + this->estimate_volume_term(y, num_volume_samples, volume_sample_sampler)))
                          .eval();
        else // if (y.boundary_type == Neumann || y.boundary_type == Robin)
            result += (domain_type_coefficient() * inv_pdf * equation_kernel.dG_dx(y.p, x.p) *
                       (y.boundary_value + this->estimate_volume_term(y, num_volume_samples, volume_sample_sampler)))
                          .eval();
        return result;
    }

    inline __device__ KernelType
    solution_boundary_unknown(const BoundaryPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf) const {
        if (x.boundary_type == Dirichlet) return utils::zero<KernelType>();

        if (y.boundary_type == Dirichlet)
            return domain_type_coefficient() / integral_free_term(x.interior_angle) * inv_pdf *
                   equation_kernel.G(y.p, x.p);
        else if (y.boundary_type == Neumann)
            return -domain_type_coefficient() / integral_free_term(x.interior_angle) * inv_pdf *
                   equation_kernel.dG_dny(y.p, x.p, y.n);
        else // if (y.boundary_type == Robin)
            return -domain_type_coefficient() / integral_free_term(x.interior_angle) * inv_pdf *
                   (equation_kernel.dG_dny(y.p, x.p, y.n) + y.robin_alpha * equation_kernel.G(y.p, x.p));
    }
    template <class Func>
    inline __device__ ValueType solution_boundary_known(
        const BoundaryPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const {
        if (x.boundary_type == Dirichlet) return x.boundary_value;

        ValueType result = this->estimate_volume_term_init(x, num_volume_samples, volume_sample_sampler);
        if (y.boundary_type == Dirichlet)
            result += -domain_type_coefficient() / integral_free_term(x.interior_angle) * inv_pdf *
                      equation_kernel.dG_dny(y.p, x.p, y.n) *
                      (y.boundary_value + this->estimate_volume_term(y, num_volume_samples, volume_sample_sampler));
        else // if (y.boundary_type == Neumann || y.boundary_type == Robin)
            result += domain_type_coefficient() / integral_free_term(x.interior_angle) * inv_pdf *
                      equation_kernel.G(y.p, x.p) *
                      (y.boundary_value + this->estimate_volume_term(y, num_volume_samples, volume_sample_sampler));
        return result;
    }

    inline __device__ KernelType fredholm_equation_boundary_unknown(
        BoundaryPointXs &x, BoundaryPointXs &y, ScalarType inv_pdf, wob::randomState_t *random_state_ptr,
        const bool is_backward_sampling
    ) const {
        if (is_backward_sampling && x.boundary_type == Dirichlet) {
            if (utils::rand_uniform<ScalarType>(random_state_ptr) * (1.0f + first_kind_equation_scaling_constant) <
                1.0f) {
                y = x;
                return (1.0f + first_kind_equation_scaling_constant) * utils::one<KernelType>();
            } else {
                inv_pdf *= (1.0f + first_kind_equation_scaling_constant) / first_kind_equation_scaling_constant;
            }
        } else if (!is_backward_sampling && y.boundary_type == Dirichlet) {
            if (utils::rand_uniform<ScalarType>(random_state_ptr) * (1.0f + first_kind_equation_scaling_constant) <
                1.0f) {
                x = y;
                return (1.0f + first_kind_equation_scaling_constant) * utils::one<KernelType>();
            } else {
                inv_pdf *= (1.0f + first_kind_equation_scaling_constant) / first_kind_equation_scaling_constant;
            }
        }

        if (x.boundary_type == Dirichlet) {
            if (y.boundary_type == Dirichlet)
                return -first_kind_equation_scaling_constant / integral_free_term(x.interior_angle) * inv_pdf *
                       equation_kernel.G(y.p, x.p);
            else if (y.boundary_type == Neumann)
                return first_kind_equation_scaling_constant / integral_free_term(x.interior_angle) * inv_pdf *
                       equation_kernel.dG_dny(y.p, x.p, y.n);
            else // if (y.boundary_type == Robin)
                return first_kind_equation_scaling_constant / integral_free_term(x.interior_angle) * inv_pdf *
                       (equation_kernel.dG_dny(y.p, x.p, y.n) + y.robin_alpha * equation_kernel.G(y.p, x.p));
        } else { // if (x.boundary_type == Neumann || x.boundary_type == Robin)
            if (y.boundary_type == Dirichlet)
                return domain_type_coefficient() / integral_free_term(x.interior_angle) * inv_pdf *
                       equation_kernel.G(y.p, x.p);
            else if (y.boundary_type == Neumann)
                return -domain_type_coefficient() / integral_free_term(x.interior_angle) * inv_pdf *
                       equation_kernel.dG_dny(y.p, x.p, y.n);
            else // if (y.boundary_type == Robin)
                return -domain_type_coefficient() / integral_free_term(x.interior_angle) * inv_pdf *
                       (equation_kernel.dG_dny(y.p, x.p, y.n) + y.robin_alpha * equation_kernel.G(y.p, x.p));
        }
    }
    template <class Func>
    inline __device__ ValueType fredholm_equation_boundary_known(
        const BoundaryPointXs &x, const BoundaryPointXs &y, ScalarType inv_pdf, unsigned int num_volume_samples,
        Func volume_sample_sampler
    ) const {
        if (x.boundary_type == Dirichlet) {
            if (y.boundary_type == Dirichlet)
                return domain_type_coefficient() * first_kind_equation_scaling_constant *
                       ((x.boundary_value + this->estimate_volume_term(x, num_volume_samples, volume_sample_sampler)) +
                        domain_type_coefficient() / integral_free_term(x.interior_angle) * inv_pdf *
                            equation_kernel.dG_dny(y.p, x.p, y.n) *
                            (y.boundary_value + this->estimate_volume_term(y, num_volume_samples, volume_sample_sampler)
                            ));
            else // if (y.boundary_type == Neumann || y.boundary_type == Robin)
                return domain_type_coefficient() * first_kind_equation_scaling_constant *
                       ((x.boundary_value + this->estimate_volume_term(x, num_volume_samples, volume_sample_sampler)) -
                        domain_type_coefficient() / integral_free_term(x.interior_angle) * inv_pdf *
                            equation_kernel.G(y.p, x.p) *
                            (y.boundary_value + this->estimate_volume_term(y, num_volume_samples, volume_sample_sampler)
                            ));
        } else { // if (x.boundary_type == Neumann || x.boundary_type == Robin)
            if (y.boundary_type == Dirichlet)
                return -domain_type_coefficient() / integral_free_term(x.interior_angle) * inv_pdf *
                       equation_kernel.dG_dny(y.p, x.p, y.n) *
                       (y.boundary_value + this->estimate_volume_term(y, num_volume_samples, volume_sample_sampler));
            else // if (y.boundary_type == Neumann || y.boundary_type == Robin)
                return domain_type_coefficient() / integral_free_term(x.interior_angle) * inv_pdf *
                       equation_kernel.G(y.p, x.p) *
                       (y.boundary_value + this->estimate_volume_term(y, num_volume_samples, volume_sample_sampler));
        }
    }
};

// -------------------- Direct (End) --------------------

} // namespace wob

#endif //__WOB_BIE_MODELS_CUH__