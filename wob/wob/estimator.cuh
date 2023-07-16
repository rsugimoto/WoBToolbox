/*
    Walk-on-Boundary Toolbox
    This file is a part of the public release of the Walk-on-Boundary (WoB) Toolbox.

    Author:         Ryusuke Sugimoto
    Affiliation:    University of Waterloo
    Date:           July 2023
    File Name:      estimator.cuh
    Description:    This file defines forward and backward estimators for the WoB method. The estimator class also
                    determines the specififc sampling method used.
*/

#ifndef __WOB_ESTIMATOR_CUH__
#define __WOB_ESTIMATOR_CUH__

#include <string>
#include <thrust/tuple.h>

#include "bie_models.cuh"

namespace wob {
template <typename ScalarType, unsigned int Dim, bool IsVectorProblem> class Scene;

enum EstimatorType { BackwardEstimator, ForwardEstimator, EstimatorTypeCount };
EstimatorType string_to_estimator_type(const std::string &str) {
    if (str == "BackwardEstimator") return BackwardEstimator;
    if (str == "ForwardEstimator") return ForwardEstimator;
    return EstimatorTypeCount;
}
std::string estimator_type_to_string(const EstimatorType estimator_type) {
    switch (estimator_type) {
    case ForwardEstimator: return "ForwardEstimator";
    case BackwardEstimator: return "BackwardEstimator";
    default: return "InvalidEstimatorType";
    }
}

enum ProblemType { DirichletProblem, NeumannProblem, RobinProblem, MixedProblem, ProblemTypeCount };
ProblemType string_to_problem_type(const std::string &str) {
    if (str == "DirichletProblem") return DirichletProblem;
    if (str == "NeumannProblem") return NeumannProblem;
    if (str == "RobinProblem") return RobinProblem;
    if (str == "MixedProblem") return MixedProblem;
    return ProblemTypeCount;
}
std::string problem_type_to_string(const ProblemType problem_type) {
    switch (problem_type) {
    case DirichletProblem: return "DirichletProblem";
    case NeumannProblem: return "NeumannProblem";
    case RobinProblem: return "RobinProblem";
    case MixedProblem: return "MixedProblem";
    default: return "InvalidProblemType";
    }
}

template <
    typename ScalarType, unsigned int Dim, EquationKernelType equationKernel, BIEModelType bieModel,
    DomainType domainType, EstimatorType estimator, ProblemType problem>
class Estimator {
  public:
    inline __host__ __device__ Estimator(
        wob::randomState_t *random_state_ptr, const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene,
        unsigned int path_length
    )
        : random_state_ptr(random_state_ptr), scene(scene), path_length(path_length), bie_model(),
          num_volume_samples(0){};
    inline __host__ __device__ Estimator(
        wob::randomState_t *random_state_ptr, const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene,
        unsigned int path_length, unsigned int num_resampling_candidates
    )
        : random_state_ptr(random_state_ptr), scene(scene), path_length(path_length),
          num_resampling_candidates(num_resampling_candidates), bie_model(), num_volume_samples(0){};
    inline __host__ __device__ Estimator(
        wob::randomState_t *random_state_ptr, const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene,
        unsigned int path_length, unsigned int num_resampling_candidates, ScalarType eps
    )
        : random_state_ptr(random_state_ptr), scene(scene), path_length(path_length),
          num_resampling_candidates(num_resampling_candidates), bie_model(eps), num_volume_samples(0){};
    inline __host__ __device__ Estimator(
        wob::randomState_t *random_state_ptr, const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene,
        unsigned int path_length, unsigned int num_resampling_candidates, ScalarType eps,
        ScalarType first_kind_equation_scaling_constant
    )
        : random_state_ptr(random_state_ptr), scene(scene), path_length(path_length),
          num_resampling_candidates(num_resampling_candidates), bie_model(eps, first_kind_equation_scaling_constant),
          num_volume_samples(0){};

    inline __host__ __device__ Estimator(
        wob::randomState_t *random_state_ptr, const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene,
        unsigned int path_length, unsigned int num_resampling_candidates, ScalarType eps,
        ScalarType first_kind_equation_scaling_constant, unsigned int num_volume_samples
    )
        : random_state_ptr(random_state_ptr), scene(scene), path_length(path_length),
          num_resampling_candidates(num_resampling_candidates), bie_model(eps, first_kind_equation_scaling_constant),
          num_volume_samples(num_volume_samples){};

    inline __device__ auto compute_sample_path_contribution_domain(const DomainPoint<ScalarType, Dim> &x);

    inline __device__ auto compute_sample_path_contribution_boundary(
        const BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()> &x
    );

    wob::randomState_t *random_state_ptr;
    const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene;
    unsigned int path_length;
    unsigned int num_resampling_candidates;
    BIEModel<ScalarType, Dim, equationKernel, bieModel, domainType> bie_model;
    unsigned int num_volume_samples;
};

// ---------- Backward Estimator (Begin) ----------

template <
    unsigned int Dim, typename ScalarType, EquationKernelType equationKernel, BIEModelType bieModel, DomainType domain,
    ProblemType problem>
class Estimator<ScalarType, Dim, equationKernel, bieModel, domain, BackwardEstimator, problem> {
  public:
    inline __host__ __device__ Estimator(
        wob::randomState_t *random_state_ptr, const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene,
        unsigned int path_length
    )
        : random_state_ptr(random_state_ptr), scene(scene), path_length(path_length), bie_model(),
          num_volume_samples(0){};
    inline __host__ __device__ Estimator(
        wob::randomState_t *random_state_ptr, const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene,
        unsigned int path_length, unsigned int num_resampling_candidates
    )
        : random_state_ptr(random_state_ptr), scene(scene), path_length(path_length),
          num_resampling_candidates(num_resampling_candidates), bie_model(), num_volume_samples(0){};
    inline __host__ __device__ Estimator(
        wob::randomState_t *random_state_ptr, const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene,
        unsigned int path_length, unsigned int num_resampling_candidates, ScalarType eps
    )
        : random_state_ptr(random_state_ptr), scene(scene), path_length(path_length),
          num_resampling_candidates(num_resampling_candidates), bie_model(eps), num_volume_samples(0){};
    inline __host__ __device__ Estimator(
        wob::randomState_t *random_state_ptr, const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene,
        unsigned int path_length, unsigned int num_resampling_candidates, ScalarType eps,
        ScalarType first_kind_equation_scaling_constant
    )
        : random_state_ptr(random_state_ptr), scene(scene), path_length(path_length),
          num_resampling_candidates(num_resampling_candidates), bie_model(eps, first_kind_equation_scaling_constant),
          num_volume_samples(0){};

    inline __host__ __device__ Estimator(
        wob::randomState_t *random_state_ptr, const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene,
        unsigned int path_length, unsigned int num_resampling_candidates, ScalarType eps,
        ScalarType first_kind_equation_scaling_constant, unsigned int num_volume_samples
    )
        : random_state_ptr(random_state_ptr), scene(scene), path_length(path_length),
          num_resampling_candidates(num_resampling_candidates), bie_model(eps, first_kind_equation_scaling_constant),
          num_volume_samples(num_volume_samples){};

    inline __device__ auto compute_sample_path_contribution_domain(const DomainPoint<ScalarType, Dim> &x) {
        BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()> y, y1;
        ScalarType inv_pdf, inv_pdf1;

        if constexpr (bieModel == IndirectDoubleLayer && problem == DirichletProblem && equationKernel == PoissonKernel) {
            thrust::tie(y, inv_pdf) = scene.sample_boundary_line_intersection(random_state_ptr, x);
            y1 = y;
            inv_pdf1 = inv_pdf;
        } else if constexpr (bieModel == Direct && problem == NeumannProblem && equationKernel == PoissonKernel) {
            thrust::tie(y, inv_pdf) = scene.sample_boundary_line_intersection(random_state_ptr, x);
            thrust::tie(y1, inv_pdf1) = scene.sample_boundary_boundary_value_sampling(random_state_ptr);
        } else {
            thrust::tie(y, inv_pdf) = scene.sample_boundary_uniform(random_state_ptr);
            thrust::tie(y1, inv_pdf1) = scene.sample_boundary_uniform(random_state_ptr);
        }

        auto volume_sample_sampler = [this]() {
            return scene.sample_volme_cache_cache_value_sampling(random_state_ptr);
        };

        if constexpr (is_vector_problem<equationKernel>()) {
            Eigen::Matrix<ScalarType, Dim, 1> result =
                bie_model.solution_domain_known(x, y1, inv_pdf1, num_volume_samples, volume_sample_sampler);
            Eigen::Matrix<ScalarType, Dim, 1> sample_path_contribution = compute_sample_unknown_boundary(y);
            result += bie_model.solution_domain_unknown(x, y, inv_pdf) * sample_path_contribution;
            return result;
        } else {
            Eigen::Matrix<ScalarType, Dim + 1, 1> result;
            result[0] = bie_model.solution_domain_known(x, y1, inv_pdf1, num_volume_samples, volume_sample_sampler);
            result.template tail<Dim>() =
                bie_model.gradient_domain_known(x, y1, inv_pdf1, num_volume_samples, volume_sample_sampler);

            ScalarType sample_path_contribution = compute_sample_unknown_boundary(y);
            result[0] += sample_path_contribution * bie_model.solution_domain_unknown(x, y, inv_pdf);
            result.template tail<Dim>() += sample_path_contribution * bie_model.gradient_domain_unknown(x, y, inv_pdf);
            return result;
        }
    }

    inline __device__ ScalarType compute_sample_path_contribution_boundary(
        const BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()> &x
    ) {
        BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()> y, y1;
        ScalarType inv_pdf, inv_pdf1;

        if constexpr (bieModel == IndirectDoubleLayer && problem == DirichletProblem && equationKernel == PoissonKernel) {
            thrust::tie(y, inv_pdf) = scene.sample_boundary_line_intersection(random_state_ptr, x);
            y1 = y;
            inv_pdf1 = inv_pdf;
        } else if constexpr (bieModel == Direct && problem == NeumannProblem && equationKernel == PoissonKernel) {
            thrust::tie(y, inv_pdf) = scene.sample_boundary_line_intersection(random_state_ptr, x);
            thrust::tie(y1, inv_pdf1) = scene.sample_boundary_boundary_value_sampling(random_state_ptr);
        } else {
            thrust::tie(y, inv_pdf) = scene.sample_boundary_uniform(random_state_ptr);
            thrust::tie(y1, inv_pdf1) = scene.sample_boundary_uniform(random_state_ptr);
        }

        auto volume_sample_sampler = [this]() {
            return scene.sample_volme_cache_cache_value_sampling(random_state_ptr);
        };

        ScalarType result;
        result = bie_model.solution_boundary_known(x, y1, inv_pdf1, num_volume_samples, volume_sample_sampler);

        ScalarType sample_path_contribution = compute_sample_unknown_boundary(y);
        result += sample_path_contribution * bie_model.solution_boundary_unknown(x, y, inv_pdf);

        return result;
    }

    wob::randomState_t *random_state_ptr;
    const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene;
    unsigned int path_length;
    unsigned int num_resampling_candidates;
    BIEModel<ScalarType, Dim, equationKernel, bieModel, domain> bie_model;
    unsigned int num_volume_samples;

    //   private:
    inline __device__ auto
    compute_sample_unknown_boundary(const BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()> &x) {
        typename value_type<ScalarType, Dim, is_vector_problem<equationKernel>()>::type sample_path_contribution;
        typename kernel_type<ScalarType, Dim, is_vector_problem<equationKernel>()>::type path_weight;

        if constexpr (is_vector_problem<equationKernel>()) {
            sample_path_contribution.setZero();
            path_weight.setIdentity();
        } else {
            sample_path_contribution = 0.0;
            path_weight = 1.0;
        }

        auto volume_sample_sampler = [this]() {
            return scene.sample_volme_cache_cache_value_sampling(random_state_ptr);
        };

        BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()> y_prev = x;
        BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()> y;
        ScalarType inv_pdf;
        for (unsigned int m = 0; m < path_length; m++) {
            if (m == path_length - 1) path_weight = 0.5f * path_weight;

            if constexpr (bieModel == IndirectDoubleLayer && problem == DirichletProblem && equationKernel == PoissonKernel) {
                thrust::tie(y, inv_pdf) = scene.sample_boundary_line_intersection(random_state_ptr, y_prev);
                sample_path_contribution +=
                    path_weight * bie_model.fredholm_equation_boundary_known(
                                      y_prev, y, inv_pdf, num_volume_samples, volume_sample_sampler
                                  );
                path_weight *= bie_model.fredholm_equation_boundary_unknown(y_prev, y, inv_pdf, random_state_ptr, true);

                y_prev = y;
            } else if constexpr (bieModel == Direct && problem == NeumannProblem && equationKernel == PoissonKernel) {
                thrust::tie(y, inv_pdf) = scene.sample_boundary_boundary_value_sampling(random_state_ptr);
                sample_path_contribution +=
                    path_weight * bie_model.fredholm_equation_boundary_known(
                                      y_prev, y, inv_pdf, num_volume_samples, volume_sample_sampler
                                  );

                thrust::tie(y, inv_pdf) = scene.sample_boundary_line_intersection(random_state_ptr, y_prev);
                path_weight = path_weight *
                              bie_model.fredholm_equation_boundary_unknown(y_prev, y, inv_pdf, random_state_ptr, true);

                y_prev = y;
            } else if constexpr (bieModel == IndirectSingleLayer) {
                // thrust::tie(y, inv_pdf) = scene.sample_boundary_uniform(random_state_ptr); // unnecessary
                sample_path_contribution +=
                    path_weight * bie_model.fredholm_equation_boundary_known(
                                      y_prev, y, inv_pdf, num_volume_samples, volume_sample_sampler
                                  );

                if (num_resampling_candidates == 1) {
                    thrust::tie(y, inv_pdf) = scene.sample_boundary_uniform(random_state_ptr);
                } else {
                    switch (y_prev.boundary_type) {
                    case Dirichlet:
                        thrust::tie(y, inv_pdf) = scene.sample_boundary_resampled_importance_sampling(
                            random_state_ptr,
                            [&](const BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()> &y) {
                                if constexpr (is_vector_problem<equationKernel>())
                                    return bie_model.equation_kernel.G(y.p, y_prev.p).norm();
                                else
                                    return abs(bie_model.equation_kernel.G(y.p, y_prev.p));
                            },
                            num_resampling_candidates
                        );
                        break;
                    case Neumann:
                        thrust::tie(y, inv_pdf) = scene.sample_boundary_resampled_importance_sampling(
                            random_state_ptr,
                            [&](const BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()> &y) {
                                if constexpr (is_vector_problem<equationKernel>())
                                    return bie_model.equation_kernel.dG_dny(y_prev.p, y.p, y_prev.n).norm();
                                else
                                    return abs(bie_model.equation_kernel.dG_dny(y_prev.p, y.p, y_prev.n));
                            },
                            num_resampling_candidates
                        );
                        break;
                    case Robin:
                        thrust::tie(y, inv_pdf) = scene.sample_boundary_resampled_importance_sampling(
                            random_state_ptr,
                            [&](const BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()> &y) {
                                if constexpr (is_vector_problem<equationKernel>())
                                    return (bie_model.equation_kernel.dG_dny(y_prev.p, y.p, y_prev.n) +
                                            y_prev.robin_alpha * bie_model.equation_kernel.G(y.p, y_prev.p))
                                        .norm();
                                else
                                    return abs(
                                        bie_model.equation_kernel.dG_dny(y_prev.p, y.p, y_prev.n) +
                                        y_prev.robin_alpha * bie_model.equation_kernel.G(y.p, y_prev.p)
                                    );
                            },
                            num_resampling_candidates
                        );
                        break;
                    }
                }

                path_weight = path_weight *
                              bie_model.fredholm_equation_boundary_unknown(y_prev, y, inv_pdf, random_state_ptr, true);

                y_prev = y;
            } else {
                thrust::tie(y, inv_pdf) = scene.sample_boundary_uniform(random_state_ptr);
                sample_path_contribution +=
                    path_weight * bie_model.fredholm_equation_boundary_known(
                                      y_prev, y, inv_pdf, num_volume_samples, volume_sample_sampler
                                  );

                thrust::tie(y, inv_pdf) = scene.sample_boundary_uniform(random_state_ptr);
                path_weight = path_weight *
                              bie_model.fredholm_equation_boundary_unknown(y_prev, y, inv_pdf, random_state_ptr, true);

                y_prev = y;
            }

            if constexpr (is_vector_problem<equationKernel>()) {
                if (path_weight.norm() == 0.0f) break;
            } else {
                if (path_weight == 0.0f) break;
            }
        }

        return sample_path_contribution;
    }
};

// ---------- Backward Estimator (End) ----------

// ---------- Forward Estimator (Begin) ----------

template <
    unsigned int Dim, typename ScalarType, EquationKernelType equationKernel, BIEModelType bieModel, DomainType domain,
    ProblemType problem>
class Estimator<ScalarType, Dim, equationKernel, bieModel, domain, ForwardEstimator, problem> {
  public:
    inline __host__ __device__ Estimator(
        wob::randomState_t *random_state_ptr, const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene,
        unsigned int path_length
    )
        : random_state_ptr(random_state_ptr), scene(scene), path_length(path_length), bie_model(),
          num_volume_samples(0){};
    inline __host__ __device__ Estimator(
        wob::randomState_t *random_state_ptr, const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene,
        unsigned int path_length, unsigned int num_resampling_candidates
    )
        : random_state_ptr(random_state_ptr), scene(scene), path_length(path_length),
          num_resampling_candidates(num_resampling_candidates), bie_model(), num_volume_samples(0){};
    inline __host__ __device__ Estimator(
        wob::randomState_t *random_state_ptr, const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene,
        unsigned int path_length, unsigned int num_resampling_candidates, ScalarType eps
    )
        : random_state_ptr(random_state_ptr), scene(scene), path_length(path_length),
          num_resampling_candidates(num_resampling_candidates), bie_model(eps), num_volume_samples(0){};
    inline __host__ __device__ Estimator(
        wob::randomState_t *random_state_ptr, const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene,
        unsigned int path_length, unsigned int num_resampling_candidates, ScalarType eps,
        ScalarType first_kind_equation_scaling_constant
    )
        : random_state_ptr(random_state_ptr), scene(scene), path_length(path_length),
          num_resampling_candidates(num_resampling_candidates), bie_model(eps, first_kind_equation_scaling_constant),
          num_volume_samples(0){};

    inline __host__ __device__ Estimator(
        wob::randomState_t *random_state_ptr, const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene,
        unsigned int path_length, unsigned int num_resampling_candidates, ScalarType eps,
        ScalarType first_kind_equation_scaling_constant, unsigned int num_volume_samples
    )
        : random_state_ptr(random_state_ptr), scene(scene), path_length(path_length),
          num_resampling_candidates(num_resampling_candidates), bie_model(eps, first_kind_equation_scaling_constant),
          num_volume_samples(num_volume_samples){};

    inline __device__ auto compute_sample_path_contribution_domain(const DomainPoint<ScalarType, Dim> &x) {
        BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()> y, y2;
        ScalarType inv_pdf, inv_pdf2;

        Eigen::Matrix<ScalarType, is_vector_problem<equationKernel>() ? Dim : Dim + 1, 1> result;
        auto volume_sample_sampler = [this]() {
            return scene.sample_volme_cache_cache_value_sampling(random_state_ptr);
        };

        thrust::tie(y, inv_pdf) = scene.sample_boundary_uniform(random_state_ptr);
        if constexpr (is_vector_problem<equationKernel>()) {
            result = bie_model.solution_domain_known(x, y, inv_pdf, num_volume_samples, volume_sample_sampler);
        } else {
            result[0] = bie_model.solution_domain_known(x, y, inv_pdf, num_volume_samples, volume_sample_sampler);
            result.template tail<Dim>() =
                bie_model.gradient_domain_known(x, y, inv_pdf, num_volume_samples, volume_sample_sampler);
        }

        if constexpr (bieModel == IndirectSingleLayer && problem == NeumannProblem)
            thrust::tie(y, inv_pdf) = scene.sample_boundary_boundary_value_sampling(random_state_ptr);
        else
            thrust::tie(y, inv_pdf) = scene.sample_boundary_uniform(random_state_ptr);
        thrust::tie(y2, inv_pdf2) = scene.sample_boundary_uniform(random_state_ptr);

        typename value_type<ScalarType, Dim, is_vector_problem<equationKernel>()>::type path_weight =
            inv_pdf *
            bie_model.fredholm_equation_boundary_known(y, y2, inv_pdf2, num_volume_samples, volume_sample_sampler);

        BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()> y_prev;
        for (unsigned int m = 0; m < path_length; m++) {
            if (m == path_length - 1) path_weight *= 0.5f;

            if constexpr (is_vector_problem<equationKernel>()) {
                result += bie_model.solution_domain_unknown(x, y, 1.0f) * path_weight;
            } else {
                result[0] += path_weight * bie_model.solution_domain_unknown(x, y, 1.0f);
                result.template tail<Dim>() += path_weight * bie_model.gradient_domain_unknown(x, y, 1.0f);
            }

            y_prev = y;
            if constexpr (bieModel == IndirectSingleLayer && problem == NeumannProblem && equationKernel == PoissonKernel)
                thrust::tie(y, inv_pdf) = scene.sample_boundary_line_intersection(random_state_ptr, y_prev);
            else
                thrust::tie(y, inv_pdf) = scene.sample_boundary_uniform(random_state_ptr);
            path_weight =
                bie_model.fredholm_equation_boundary_unknown(y, y_prev, inv_pdf, random_state_ptr, false) * path_weight;
        }

        return result;
    }

    inline __device__ typename value_type<ScalarType, Dim, is_vector_problem<equationKernel>()>::type
    compute_sample_path_contribution_boundary(
        const BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()> &x
    ) {
        BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()> y, y2;
        ScalarType inv_pdf, inv_pdf2;

        auto volume_sample_sampler = [this]() {
            return scene.sample_volme_cache_cache_value_sampling(random_state_ptr);
        };

        thrust::tie(y, inv_pdf) = scene.sample_boundary_uniform(random_state_ptr);
        typename value_type<ScalarType, Dim, is_vector_problem<equationKernel>()>::type result =
            bie_model.solution_boundary_known(x, y, inv_pdf, num_volume_samples, volume_sample_sampler);

        if constexpr (bieModel == IndirectSingleLayer && problem == NeumannProblem)
            thrust::tie(y, inv_pdf) = scene.sample_boundary_boundary_value_sampling(random_state_ptr);
        else
            thrust::tie(y, inv_pdf) = scene.sample_boundary_uniform(random_state_ptr);
        thrust::tie(y2, inv_pdf2) = scene.sample_boundary_uniform(random_state_ptr);

        typename value_type<ScalarType, Dim, is_vector_problem<equationKernel>()>::type path_weight =
            inv_pdf *
            bie_model.fredholm_equation_boundary_known(y, y2, inv_pdf2, num_volume_samples, volume_sample_sampler);

        BoundaryPoint<ScalarType, Dim, is_vector_problem<equationKernel>()> y_prev;
        for (unsigned int m = 0; m < path_length; m++) {
            if (m == path_length - 1) path_weight *= 0.5f;
            result += bie_model.solution_boundary_unknown(x, y, 1.0f) * path_weight;

            y_prev = y;
            if constexpr (bieModel == IndirectSingleLayer && problem == NeumannProblem && equationKernel == PoissonKernel)
                thrust::tie(y, inv_pdf) = scene.sample_boundary_line_intersection(random_state_ptr, y_prev);
            else
                thrust::tie(y, inv_pdf) = scene.sample_boundary_uniform(random_state_ptr);
            path_weight =
                bie_model.fredholm_equation_boundary_unknown(y, y_prev, inv_pdf, random_state_ptr, false) * path_weight;
        }

        return result;
    }

    wob::randomState_t *random_state_ptr;
    const Scene<ScalarType, Dim, is_vector_problem<equationKernel>()> &scene;
    unsigned int path_length;
    unsigned int num_resampling_candidates;
    BIEModel<ScalarType, Dim, equationKernel, bieModel, domain> bie_model;
    unsigned int num_volume_samples;
};

// ---------- Forward Estimator (End) ----------

} // namespace wob

#endif //__WOB_ESTIMATOR_CUH__