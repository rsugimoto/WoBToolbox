/*
    Walk-on-Boundary Toolbox
    This file is a part of the public release of the Walk-on-Boundary (WoB) Toolbox.

    Author:         Ryusuke Sugimoto
    Affiliation:    University of Waterloo
    Date:           July 2023
    File Name:      scene.cuh
    Description:    This file defines classes and functions to perform boundary or volume sampling in the scene.
   Currently, the scene is assumed to consist of a set of traigles in 3D and a set of line segments in 2D. Therefore,
   there is only "local" BVH with no "global" BVH.
*/

#ifndef __WOB_SCENE_CUH__
#define __WOB_SCENE_CUH__

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <thrust/optional.h>
#include <thrust/pair.h>

#include "lbvh.cuh"

#include "points.hpp"
#include "utils.cuh"

namespace wob {

template <typename ScalarType>
inline __host__ __device__ unsigned int
binary_search(const ScalarType *array, unsigned int array_size, ScalarType key) {
    unsigned int low = 0, high = array_size - 1;
    while (low < high) {
        unsigned int mid = (high + low) / 2;
        if (key <= array[mid]) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return high;
}

// Copied from igl/barycentric_coordinates.cpp and __host__ __device__ qualifiers are added
template <typename DerivedP, typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedL>
__host__ __device__ void barycentric_coordinates(
    const Eigen::MatrixBase<DerivedP> &P, const Eigen::MatrixBase<DerivedA> &A, const Eigen::MatrixBase<DerivedB> &B,
    const Eigen::MatrixBase<DerivedC> &C, Eigen::PlainObjectBase<DerivedL> &L
) {
    using namespace Eigen;
    // http://gamedev.stackexchange.com/a/23745
    typedef Eigen::Array<typename DerivedP::Scalar, DerivedP::RowsAtCompileTime, DerivedP::ColsAtCompileTime> ArrayS;
    typedef Eigen::Array<typename DerivedP::Scalar, DerivedP::RowsAtCompileTime, 1> VectorS;

    const ArrayS v0 = B.array() - A.array();
    const ArrayS v1 = C.array() - A.array();
    const ArrayS v2 = P.array() - A.array();
    VectorS d00 = (v0 * v0).rowwise().sum();
    VectorS d01 = (v0 * v1).rowwise().sum();
    VectorS d11 = (v1 * v1).rowwise().sum();
    VectorS d20 = (v2 * v0).rowwise().sum();
    VectorS d21 = (v2 * v1).rowwise().sum();
    VectorS denom = d00 * d11 - d01 * d01;
    L.resize(P.rows(), 3);
    L.col(1) = (d11 * d20 - d01 * d21) / denom;
    L.col(2) = (d00 * d21 - d01 * d20) / denom;
    L.col(0) = 1.0f - (L.col(1) + L.col(2)).array();
}

template <unsigned int p, typename ScalarType> inline __host__ __device__ ScalarType int_pow(ScalarType x) {
    ScalarType result = 1.f;
#pragma unroll
    for (unsigned int i = 0; i < p; i++) result *= x;
    return result;
}

template <typename ScalarType, unsigned int Dim, bool IsVectorCache> struct VolumeCache {
    unsigned int grid_res;
    ScalarType domain_size;
    const typename value_type<ScalarType, Dim, IsVectorCache>::type *cache;
    const ScalarType *cdf;

    inline __device__ thrust::pair<
        DomainSamplePoint<ScalarType, Dim, typename value_type<ScalarType, Dim, IsVectorCache>::type>, ScalarType>
    sample_volme_cache_uniform(wob::randomState_t *random_state_ptr) const {
        ScalarType unif = 1.0f - utils::rand_uniform<ScalarType>(random_state_ptr); // make it [0.0, 1.0)
        unsigned int idx = int_pow<Dim>(grid_res) * unif;
        Eigen::Matrix<ScalarType, Dim, 1> p = utils::idx_to_domain_point<ScalarType, Dim>(idx, grid_res, domain_size);
#pragma unroll
        for (unsigned int i = 0; i < Dim; i++) {
            ScalarType unif = utils::rand_uniform<ScalarType>(random_state_ptr);
            p[i] += (unif - 0.5f) * domain_size / grid_res;
        }
        DomainSamplePoint<ScalarType, Dim, typename value_type<ScalarType, Dim, IsVectorCache>::type> result_point{
            p, cache[idx]};
        return {result_point, int_pow<Dim>(domain_size)};
    }

    inline __device__ thrust::pair<
        DomainSamplePoint<ScalarType, Dim, typename value_type<ScalarType, Dim, IsVectorCache>::type>, ScalarType>
    sample_volme_cache_cache_value_sampling(wob::randomState_t *random_state_ptr) const {
        ScalarType unif = utils::rand_uniform<ScalarType>(random_state_ptr);
        unsigned int idx = binary_search(cdf, int_pow<Dim>(grid_res), unif);
        Eigen::Matrix<ScalarType, Dim, 1> p = utils::idx_to_domain_point<ScalarType, Dim>(idx, grid_res, domain_size);
#pragma unroll
        for (unsigned int i = 0; i < Dim; i++) {
            ScalarType unif = utils::rand_uniform<ScalarType>(random_state_ptr);
            p[i] += (unif - 0.5f) * domain_size / grid_res;
        }
        DomainSamplePoint<ScalarType, Dim, typename value_type<ScalarType, Dim, IsVectorCache>::type> result_point{
            p, cache[idx]};

        ScalarType pdf = idx == 0 ? cdf[idx] : cdf[idx] - cdf[idx - 1];
        return {result_point, int_pow<Dim>(domain_size / grid_res) / pdf};
    }
};

template <typename ScalarType, unsigned int Dim, bool IsVectorCache> struct VolumeCacheHost {
    explicit VolumeCacheHost(
        unsigned int grid_res, ScalarType domain_size,
        const thrust::device_vector<typename value_type<ScalarType, Dim, IsVectorCache>::type> &cache
    )
        : grid_res(grid_res), domain_size(domain_size), cache(cache) {
        update();
    }

    void _update(const thrust::device_vector<ScalarType> &cache) {
        if (cdf.size() != 0 && cdf.size() != cache.size())
            std::cerr << "Caution: VolumeCache detected a change of size of vector. This may cause "
                         "a reallocation of memory."
                      << std::endl;
        cdf = cache;
        thrust::for_each(cdf.begin(), cdf.end(), [] __device__(auto &elem) { elem = abs(elem); });
        ScalarType total_value = thrust::reduce(cdf.begin(), cdf.end());
        if (total_value == 0.0)
            thrust::fill(cdf.begin(), cdf.end(), (ScalarType)1. / cache.size());
        else
            thrust::for_each(cdf.begin(), cdf.end(), [total_value] __device__(auto &elem) { elem /= total_value; });
        thrust::inclusive_scan(cdf.begin(), cdf.end(), cdf.begin());
    }

    void _update(const thrust::device_vector<Eigen::Matrix<ScalarType, Dim, 1>> &cache) {
        if (cdf.size() != 0 && cdf.size() != cache.size())
            std::cerr << "Caution: VolumeCache detected a change of size of vector. This may cause "
                         "a reallocation of memory."
                      << std::endl;
        cdf.resize(cache.size());
        thrust::transform(cache.begin(), cache.end(), cdf.begin(), [](auto &elem) { return elem.norm(); });
        update(cdf);
    }

    void update() { _update(cache); }

    VolumeCache<ScalarType, Dim, IsVectorCache> get_device_repr() {
        return {grid_res, domain_size, cache.data().get(), cdf.data().get()};
    }

    unsigned int grid_res;
    ScalarType domain_size;
    const thrust::device_vector<typename value_type<ScalarType, Dim, IsVectorCache>::type> &cache;
    thrust::device_vector<ScalarType> cdf;

  private:
};

template <typename ScalarType, unsigned int Dim, bool IsVectorProblem> struct ElementXs {};

template <typename ScalarType, unsigned int Dim, bool IsVectorProblem> struct ElementXsAABB {};

template <typename ScalarType, unsigned int Dim, bool IsVectorProblem> struct LineElementXsIntersect {
    thrust::pair<bool, Eigen::Matrix<ScalarType, Dim, 1>> __device__ __host__ operator()(
        const lbvh::Line<ScalarType, Dim> &line, const ElementXs<ScalarType, Dim, IsVectorProblem> &elem
    ) const noexcept;
};

template <typename ScalarType, unsigned int Dim, bool IsVectorProblem> struct SceneHost {
    SceneHost(const std::vector<wob::ElementXs<ScalarType, Dim, IsVectorProblem>> &elems)
        : bvh(elems.begin(), elems.end(), true), total_boundary_area(0.0), total_boundary_value(0.0) {

        const unsigned int num_objects = bvh.objects_host().size();
        const unsigned int num_internal_nodes = num_objects - 1;

        thrust::host_vector<ScalarType> cdf_host(num_objects);

        // compute area CDF
        wob::utils::KahanSum<ScalarType> boundary_area_sum;
        for (size_t i = 0; i < num_objects; i++) {
            ScalarType elem_size = elems[bvh.nodes_host()[i + num_internal_nodes].object_idx].area();
            boundary_area_sum += elem_size;
            cdf_host[i] = boundary_area_sum.sum;
        }
        total_boundary_area = boundary_area_sum;

        for (size_t i = 0; i < num_objects; i++) { cdf_host[i] /= total_boundary_area; }

        area_cdf = cdf_host;

        // compute boundary value CDF
        wob::utils::KahanSum<ScalarType> boundary_value_sum;
        for (size_t i = 0; i < num_objects; i++) {
            ScalarType elem_boundary_value =
                elems[bvh.nodes_host()[i + num_internal_nodes].object_idx].boundary_value();
            boundary_value_sum += elem_boundary_value;
            cdf_host[i] = boundary_value_sum.sum;
        }
        total_boundary_value = boundary_value_sum;

        for (size_t i = 0; i < num_objects; i++) { cdf_host[i] /= total_boundary_value; }

        boundary_value_cdf = cdf_host;
    }

    void set_volume_cache(
        unsigned int grid_res, ScalarType domain_size,
        const thrust::device_vector<typename value_type<ScalarType, Dim, IsVectorProblem>::type> &cache
    ) {
        volume_cache.emplace(grid_res, domain_size, cache);
    }

    Scene<ScalarType, Dim, IsVectorProblem> get_device_repr() {
        return {
            lbvh::bvh_device<ScalarType, Dim, ElementXs<ScalarType, Dim, IsVectorProblem>>(bvh.get_device_repr()),
            area_cdf.data().get(),
            total_boundary_area,
            boundary_value_cdf.data().get(),
            total_boundary_value,
            volume_cache
                ? thrust::optional<VolumeCache<ScalarType, Dim, IsVectorProblem>>(volume_cache->get_device_repr())
                : thrust::nullopt};
    }

    lbvh::bvh<
        ScalarType, Dim, ElementXs<ScalarType, Dim, IsVectorProblem>, ElementXsAABB<ScalarType, Dim, IsVectorProblem>>
        bvh;
    thrust::device_vector<ScalarType> area_cdf, boundary_value_cdf;
    ScalarType total_boundary_area, total_boundary_value;
    thrust::optional<VolumeCacheHost<ScalarType, Dim, IsVectorProblem>> volume_cache;
};

template <typename ScalarType, unsigned int Dim, bool IsVectorProblem> class Scene {
  public:
    inline __device__ thrust::pair<BoundaryPoint<ScalarType, Dim, IsVectorProblem>, ScalarType>
    sample_boundary_uniform(wob::randomState_t *random_state_ptr) const;

    inline __device__ thrust::pair<BoundaryPoint<ScalarType, Dim, IsVectorProblem>, ScalarType>
    sample_boundary_line_intersection(
        wob::randomState_t *random_state_ptr, const BoundaryPoint<ScalarType, Dim, IsVectorProblem> &x
    ) const;

    inline __device__ thrust::pair<BoundaryPoint<ScalarType, Dim, IsVectorProblem>, ScalarType>
    sample_boundary_line_intersection(wob::randomState_t *random_state_ptr, const DomainPoint<ScalarType, Dim> &x)
        const;

    // Variant that returns a boundary point and number of intersections
    inline __device__ thrust::pair<BoundaryPoint<ScalarType, Dim, IsVectorProblem>, unsigned int>
    sample_boundary_line_intersection2(
        wob::randomState_t *random_state_ptr, const BoundaryPoint<ScalarType, Dim, IsVectorProblem> &x
    ) const;

    inline __device__ thrust::pair<BoundaryPoint<ScalarType, Dim, IsVectorProblem>, unsigned int>
    sample_boundary_line_intersection2(wob::randomState_t *random_state_ptr, const DomainPoint<ScalarType, Dim> &x)
        const;

    template <class Func>
    inline __device__ thrust::pair<BoundaryPoint<ScalarType, Dim, IsVectorProblem>, ScalarType>
    sample_boundary_resampled_importance_sampling(
        wob::randomState_t *random_state_ptr, Func target_distribution_func,
        const unsigned int num_resampling_candidates
    ) const;

    inline __device__ thrust::pair<BoundaryPoint<ScalarType, Dim, IsVectorProblem>, ScalarType>
    sample_boundary_boundary_value_sampling(wob::randomState_t *random_state_ptr) const;

    inline __device__
        thrust::pair<DomainSamplePoint<ScalarType, Dim, value_type<ScalarType, Dim, IsVectorProblem>>, ScalarType>
        sample_volme_cache_uniform(wob::randomState_t *random_state_ptr) const;

    inline __device__
        thrust::pair<DomainSamplePoint<ScalarType, Dim, value_type<ScalarType, Dim, IsVectorProblem>>, ScalarType>
        sample_volme_cache_cache_value_sampling(wob::randomState_t *random_state_ptr) const;

    lbvh::bvh_device<ScalarType, Dim, ElementXs<ScalarType, Dim, IsVectorProblem>> bvh;
    ScalarType *area_cdf;
    ScalarType total_boundary_area;
    ScalarType *boundary_value_cdf;
    ScalarType total_boundary_value;
    thrust::optional<VolumeCache<ScalarType, Dim, IsVectorProblem>> volume_cache;
};

template <typename ScalarType, bool IsVectorProblem> struct ElementXs<ScalarType, 2, IsVectorProblem> {
    Eigen::Matrix<ScalarType, 2, 1> a, b, n;
    typename value_type<ScalarType, 2, IsVectorProblem>::type boundary_value_a, boundary_value_b;
    BoundaryType boundary_type;
    ScalarType robin_alpha_a, robin_alpha_b;

    inline __host__ __device__ ScalarType area() const noexcept { return (b - a).norm(); }
    // used for importance sampling
    inline __host__ __device__ ScalarType boundary_value() const noexcept {
        // This is technically incorrect if the sign of boundary value changes for points a and b.
        if constexpr (IsVectorProblem)
            return area() * (boundary_value_a.norm() + boundary_value_b.norm()) / 2.0f;
        else
            return area() * (std::abs(boundary_value_a) + std::abs(boundary_value_b)) / 2.0f;
    }
    template <typename T>
    inline __host__ __device__ T interpolate(const T &val1, const T &val2, ScalarType unif) const noexcept {
        return val1 + unif * (val2 - val1);
    }
};

template <typename ScalarType, bool IsVectorProblem> struct ElementXsAABB<ScalarType, 2, IsVectorProblem> {
    __device__ __host__ lbvh::aabb<ScalarType, 2> operator()(const ElementXs<ScalarType, 2, IsVectorProblem> &line
    ) const noexcept {
        lbvh::aabb<ScalarType, 2> aabb;
        if constexpr (std::is_same<ScalarType, float>::value) {
            aabb.upper = make_float2(std::max(line.a[0], line.b[0]), std::max(line.a[1], line.b[1]));
            aabb.lower = make_float2(std::min(line.a[0], line.b[0]), std::min(line.a[1], line.b[1]));
        } else {
            aabb.upper = make_double2(std::max(line.a[0], line.b[0]), std::max(line.a[1], line.b[1]));
            aabb.lower = make_double2(std::min(line.a[0], line.b[0]), std::min(line.a[1], line.b[1]));
        }
        return aabb;
    }
};

template <typename ScalarType, bool IsVectorProblem> struct LineElementXsIntersect<ScalarType, 2, IsVectorProblem> {
    thrust::pair<bool, Eigen::Matrix<ScalarType, 2, 1>> __device__ __host__ operator()(
        const lbvh::Line<ScalarType, 2> &line, const ElementXs<ScalarType, 2, IsVectorProblem> line_segment
    ) const noexcept {
        constexpr ScalarType eps = 10 * std::numeric_limits<ScalarType>::epsilon();
        Eigen::Matrix<ScalarType, 2, 1> line_dir(line.dir.x, line.dir.y);
        Eigen::Matrix<ScalarType, 2, 1> line_origin(line.origin.x, line.origin.y);

        // construct a linear equation with
        // line.origin + t * line.dir = line_segment.origin + a * line_segment.dir
        // and solve it. This may not be numerically the best way.
        Eigen::Matrix<ScalarType, 2, 2> mat;
        mat.col(0) = line_dir;
        mat.col(1) = line_segment.a - line_segment.b;

        Eigen::Matrix<ScalarType, 2, 1> ta = mat.inverse() * (line_segment.a - line_origin);

        if (ta[1] < (ScalarType)0.0 || ta[1] > (ScalarType)1.0 || std::abs(ta[0]) <= eps || std::isnan(ta[0]) ||
            std::isnan(ta[1]))
            return {false, Eigen::Matrix<ScalarType, 2, 1>()};
        return {true, line_origin + ta[0] * line_dir};
    }
};

template <typename ScalarType, bool IsVectorProblem> class Scene<ScalarType, 2, IsVectorProblem> {

  public:
    inline __device__ thrust::pair<BoundaryPoint<ScalarType, 2, IsVectorProblem>, ScalarType>
    sample_boundary_uniform(wob::randomState_t *random_state_ptr) const {
        const unsigned int num_objects = bvh.num_objects;
        const unsigned int num_internal_nodes = num_objects - 1;

        ScalarType unif1 = utils::rand_uniform<ScalarType>(random_state_ptr);
        unsigned int node_idx = binary_search(area_cdf, num_objects, unif1);
        const ElementXs<ScalarType, 2, IsVectorProblem> &element =
            bvh.objects[bvh.nodes[node_idx + num_internal_nodes].object_idx];

        ScalarType unif2 = utils::rand_uniform<ScalarType>(random_state_ptr);
        BoundaryPoint<ScalarType, 2, IsVectorProblem> result_point{
            element.interpolate(element.a, element.b, unif2),
            element.n,
            (ScalarType)0.5,
            element.interpolate(element.boundary_value_a, element.boundary_value_b, unif2),
            element.boundary_type,
            element.interpolate(element.robin_alpha_a, element.robin_alpha_b, unif2),
        };

        return {result_point, total_boundary_area};
    };

    inline __device__ thrust::pair<BoundaryPoint<ScalarType, 2, IsVectorProblem>, unsigned int>
    sample_boundary_line_intersection2(
        wob::randomState_t *random_state_ptr, const BoundaryPoint<ScalarType, 2, IsVectorProblem> &x
    ) const {
        typename lbvh::vector_of<ScalarType, 2>::type origin;
        origin.x = x.p[0];
        origin.y = x.p[1];
        ScalarType theta = utils::rand_uniform<ScalarType>(random_state_ptr);
        typename lbvh::vector_of<ScalarType, 2>::type dir;
#ifdef __CUDACC__
        sincospi(theta, &dir.x, &dir.y);
#else
        dir.x = std::cos(theta * M_PI);
        dir.y = std::sin(theta * M_PI);
#endif
        lbvh::Line<ScalarType, 2> line{origin, dir};

        constexpr unsigned int buffer_size = 64;
        thrust::pair<unsigned int, Eigen::Matrix<ScalarType, 2, 1>> intersection_points[buffer_size];
        unsigned int num_intersections = lbvh::query_device(
            bvh, lbvh::query_line_intersect<ScalarType, 2>(line),
            LineElementXsIntersect<ScalarType, 2, IsVectorProblem>(), intersection_points, buffer_size
        );

        if (num_intersections == 0) {
            // Setting a default value here is necessary to avoid fireflies caused with undefined values.
            return {
                BoundaryPoint<ScalarType, 2, IsVectorProblem>{
                    Eigen::Matrix<ScalarType, 2, 1>::Zero(), Eigen::Matrix<ScalarType, 2, 1>::Zero(), 0.5, 0.0,
                    Dirichlet, 0.0},
                0};
        }

        num_intersections = std::min(num_intersections, buffer_size);
        unsigned int object_idx;
        Eigen::Matrix<ScalarType, 2, 1> p;
        thrust::tie(object_idx, p) = intersection_points[std::min(
            num_intersections - 1, (unsigned int)(num_intersections * utils::rand_uniform<ScalarType>(random_state_ptr))
        )];

        const ElementXs<ScalarType, 2, IsVectorProblem> &element = bvh.objects[object_idx];
        ScalarType interp = (p - element.a).norm() / element.area();

        BoundaryPoint<ScalarType, 2, IsVectorProblem> result_point{
            p,
            element.n,
            0.5,
            element.interpolate(element.boundary_value_a, element.boundary_value_b, interp),
            element.boundary_type,
            element.interpolate(element.robin_alpha_a, element.robin_alpha_b, interp)};

        return {result_point, num_intersections};
    }

    inline __device__ thrust::pair<BoundaryPoint<ScalarType, 2, IsVectorProblem>, unsigned int>
    sample_boundary_line_intersection2(wob::randomState_t *random_state_ptr, const DomainPoint<ScalarType, 2> &x)
        const {
        return sample_boundary_line_intersection2(random_state_ptr, BoundaryPoint<ScalarType, 2, IsVectorProblem>{x.p});
    }

    inline __device__ thrust::pair<BoundaryPoint<ScalarType, 2, IsVectorProblem>, ScalarType>
    sample_boundary_line_intersection(
        wob::randomState_t *random_state_ptr, const BoundaryPoint<ScalarType, 2, IsVectorProblem> &x
    ) const {
        BoundaryPoint<ScalarType, 2, IsVectorProblem> result_point;
        unsigned int num_intersections;
        thrust::tie(result_point, num_intersections) = sample_boundary_line_intersection2(random_state_ptr, x);
        constexpr ScalarType eps = 10 * std::numeric_limits<ScalarType>::epsilon();
        ScalarType inv_pdf = num_intersections * (ScalarType)M_PI * (result_point.p - x.p).norm() /
                             std::max(std::abs((result_point.p - x.p).normalized().dot(result_point.n)), eps);
        return {result_point, inv_pdf};
    }

    inline __device__ thrust::pair<BoundaryPoint<ScalarType, 2, IsVectorProblem>, ScalarType>
    sample_boundary_line_intersection(wob::randomState_t *random_state_ptr, const DomainPoint<ScalarType, 2> &x) const {
        return sample_boundary_line_intersection(random_state_ptr, BoundaryPoint<ScalarType, 2, IsVectorProblem>{x.p});
    }

    template <class Func>
    inline __device__ thrust::pair<BoundaryPoint<ScalarType, 2, IsVectorProblem>, ScalarType>
    sample_boundary_resampled_importance_sampling(
        wob::randomState_t *random_state_ptr, Func target_distribution_func,
        const unsigned int num_resampling_candidates
    ) const {
        // Stream RIS described in ReSTIR. No spatial nor temporal reuse implemented.
        // Currently, the source distribution is fixed to the uniform distribution

        class Reservoir {
          public:
            BoundaryPoint<ScalarType, 2, IsVectorProblem> y; // the output sample
            wob::randomState_t *random_state_ptr;
            ScalarType w_sum;                       // the sum of weights
            unsigned int num_resampling_candidates; // the number of samples seen so far

            __device__ Reservoir(wob::randomState_t *random_state_ptr)
                : y(), random_state_ptr(random_state_ptr), w_sum(0.0f), num_resampling_candidates(0){};
            inline __device__ void
            update(const BoundaryPoint<ScalarType, 2, IsVectorProblem> &x_i, const ScalarType w_i) {
                w_sum += w_i;
                num_resampling_candidates += 1;
                if (utils::rand_uniform<ScalarType>(random_state_ptr) <= w_i / w_sum) y = x_i;
            };
        };

        Reservoir reservoir(random_state_ptr);

        for (unsigned int i = 0; i < num_resampling_candidates; i++) {
            BoundaryPoint<ScalarType, 2, IsVectorProblem> x_i;
            ScalarType x_i_inv_pdf;
            thrust::tie(x_i, x_i_inv_pdf) = sample_boundary_uniform(random_state_ptr);
            reservoir.update(x_i, target_distribution_func(x_i) * x_i_inv_pdf);
        }

        BoundaryPoint<ScalarType, 2, IsVectorProblem> res = reservoir.y;
        ScalarType inv_pdf = reservoir.w_sum / std::max(
                                                   target_distribution_func(res) * reservoir.num_resampling_candidates,
                                                   10 * std::numeric_limits<ScalarType>::epsilon()
                                               );

        return {res, inv_pdf};
    }

    inline __device__ thrust::pair<BoundaryPoint<ScalarType, 2, IsVectorProblem>, ScalarType>
    sample_boundary_boundary_value_sampling(wob::randomState_t *random_state_ptr) const {
        const unsigned int num_objects = bvh.num_objects;
        const unsigned int num_internal_nodes = bvh.num_objects - 1;

        ScalarType unif1 = utils::rand_uniform<ScalarType>(random_state_ptr);
        unsigned int node_idx = binary_search(boundary_value_cdf, num_objects, unif1);
        const ElementXs<ScalarType, 2, IsVectorProblem> &element =
            bvh.objects[bvh.nodes[node_idx + num_internal_nodes].object_idx];

        ScalarType unif2 = utils::rand_uniform<ScalarType>(random_state_ptr);
        ScalarType ba, bb;
        if constexpr (IsVectorProblem) {
            ba = element.boundary_value_a.norm();
            bb = element.boundary_value_b.norm();
        } else {
            ba = std::abs(element.boundary_value_a);
            bb = std::abs(element.boundary_value_b);
        }
        ScalarType interp;

        if (std::abs(ba - bb) < 10 * std::numeric_limits<ScalarType>::epsilon())
            interp = unif2;
        else
            interp = (-ba + std::sqrt(ba * ba + unif2 * (bb * bb - ba * ba))) / (bb - ba);

        BoundaryPoint<ScalarType, 2, IsVectorProblem> result_point{
            element.interpolate(element.a, element.b, interp),
            element.n,
            (ScalarType)0.5,
            element.interpolate(element.boundary_value_a, element.boundary_value_b, interp),
            element.boundary_type,
            element.interpolate(element.robin_alpha_a, element.robin_alpha_b, interp),
        };

        return {result_point, total_boundary_value / (ba + interp * (bb - ba))};
    }

    inline __device__ thrust::pair<
        DomainSamplePoint<ScalarType, 2, typename value_type<ScalarType, 2, IsVectorProblem>::type>, ScalarType>
    sample_volme_cache_uniform(wob::randomState_t *random_state_ptr) const {
        return volume_cache->sample_volume_cache_uniform(random_state_ptr);
    }

    inline __device__ thrust::pair<
        DomainSamplePoint<ScalarType, 2, typename value_type<ScalarType, 2, IsVectorProblem>::type>, ScalarType>
    sample_volme_cache_cache_value_sampling(wob::randomState_t *random_state_ptr) const {
        return volume_cache->sample_volme_cache_cache_value_sampling(random_state_ptr);
    }

    lbvh::bvh_device<ScalarType, 2, ElementXs<ScalarType, 2, IsVectorProblem>> bvh;
    ScalarType *area_cdf;
    ScalarType total_boundary_area;
    ScalarType *boundary_value_cdf;
    ScalarType total_boundary_value;
    thrust::optional<VolumeCache<ScalarType, 2, IsVectorProblem>> volume_cache;
};

// 3D

template <typename ScalarType, bool IsVectorProblem> struct ElementXs<ScalarType, 3, IsVectorProblem> {
    Eigen::Matrix<ScalarType, 3, 1> a, b, c, n;
    typename value_type<ScalarType, 3, IsVectorProblem>::type boundary_value_a, boundary_value_b, boundary_value_c;
    BoundaryType boundary_type;
    ScalarType robin_alpha_a, robin_alpha_b, robin_alpha_c;

    inline __host__ __device__ ScalarType area() const noexcept {
        return (c - a).cross(b - a).norm() / (ScalarType)2.0;
    }

    inline __host__ __device__ ScalarType boundary_value() const noexcept {
        // This is technically incorrect if the sign of boundary value changes for points a, b and c.
        if constexpr (IsVectorProblem)
            return area() * (boundary_value_a.norm() + boundary_value_b.norm() + boundary_value_c.norm()) / 3.0f;
        else
            return area() * (std::abs(boundary_value_a) + std::abs(boundary_value_b) + std::abs(boundary_value_c)) /
                   3.0f;
    }

    template <typename T>
    inline __host__ __device__ T
    interpolate(const T &val1, const T &val2, const T &val3, ScalarType unif1, ScalarType unif2) const noexcept {
        ScalarType sqrt_unif1 = std::sqrt(unif1);
        return ((ScalarType)1. - sqrt_unif1) * val1 + sqrt_unif1 * ((ScalarType)1. - unif2) * val2 +
               unif2 * sqrt_unif1 * val3;
    }

    template <typename T>
    inline __host__ __device__ T
    interpolate(const T &val1, const T &val2, const T &val3, const Eigen::Matrix<ScalarType, 1, 3> &bc) const noexcept {
        return bc[0] * val1 + bc[1] * val2 + bc[2] * val3;
    }
};

template <typename ScalarType, bool IsVectorProblem> struct ElementXsAABB<ScalarType, 3, IsVectorProblem> {
    __device__ __host__ lbvh::aabb<ScalarType, 3> operator()(const ElementXs<ScalarType, 3, IsVectorProblem> &tri
    ) const noexcept {
        lbvh::aabb<ScalarType, 3> aabb;
        if constexpr (std::is_same<ScalarType, float>::value) {
            aabb.upper = make_float4(
                std::max(tri.a[0], std::max(tri.b[0], tri.c[0])), std::max(tri.a[1], std::max(tri.b[1], tri.c[1])),
                std::max(tri.a[2], std::max(tri.b[2], tri.c[2])), 0.0f
            );
            aabb.lower = make_float4(
                std::min(tri.a[0], std::min(tri.b[0], tri.c[0])), std::min(tri.a[1], std::min(tri.b[1], tri.c[1])),
                std::min(tri.a[2], std::min(tri.b[2], tri.c[2])), 0.0f
            );
        } else {
            aabb.upper = make_double4(
                std::max(tri.a[0], std::max(tri.b[0], tri.c[0])), std::max(tri.a[1], std::max(tri.b[1], tri.c[1])),
                std::max(tri.a[2], std::max(tri.b[2], tri.c[2])), 0.0f
            );
            aabb.lower = make_double4(
                std::min(tri.a[0], std::min(tri.b[0], tri.c[0])), std::min(tri.a[1], std::min(tri.b[1], tri.c[1])),
                std::min(tri.a[2], std::min(tri.b[2], tri.c[2])), 0.0f
            );
        }
        return aabb;
    }
};

template <typename ScalarType, bool IsVectorProblem> struct LineElementXsIntersect<ScalarType, 3, IsVectorProblem> {
    thrust::pair<bool, Eigen::Matrix<ScalarType, 3, 1>> __device__ __host__ operator()(
        const lbvh::Line<ScalarType, 3> &line, const ElementXs<ScalarType, 3, IsVectorProblem> triangle
    ) const noexcept {
        constexpr ScalarType eps = 10 * std::numeric_limits<ScalarType>::epsilon();
        Eigen::Matrix<ScalarType, 3, 1> line_dir(line.dir.x, line.dir.y, line.dir.z);
        Eigen::Matrix<ScalarType, 3, 1> line_origin(line.origin.x, line.origin.y, line.origin.z);

        // construct a linear equation with
        // line.origin + t * line.dir = tri.origin + a * tri.d1 + b * tri.d2
        // and solve it. This may not be numerically the best way.
        Eigen::Matrix<ScalarType, 3, 3> mat;
        mat.col(0) = line_dir;
        mat.col(1) = triangle.a - triangle.b;
        mat.col(2) = triangle.a - triangle.c;

        Eigen::Matrix<ScalarType, 3, 1> tab = mat.inverse() * (triangle.a - line_origin);

        if (tab[1] < (ScalarType)0.0 || tab[2] < (ScalarType)0.0 || tab[1] + tab[2] > (ScalarType)1.0 ||
            std::abs(tab[0]) <= eps || std::isnan(tab[0]) || std::isnan(tab[1]) || std::isnan(tab[2]))
            return {false, Eigen::Matrix<ScalarType, 3, 1>()};
        return {true, line_origin + tab[0] * line_dir};
    }
};

template <typename ScalarType, bool IsVectorProblem> class Scene<ScalarType, 3, IsVectorProblem> {
  public:
    inline __device__ thrust::pair<BoundaryPoint<ScalarType, 3, IsVectorProblem>, ScalarType>
    sample_boundary_uniform(wob::randomState_t *random_state_ptr) const {
        const unsigned int num_objects = bvh.num_objects;
        const unsigned int num_internal_nodes = num_objects - 1;

        ScalarType unif1 = utils::rand_uniform<ScalarType>(random_state_ptr);
        unsigned int node_idx = binary_search(area_cdf, num_objects, unif1);
        ElementXs<ScalarType, 3, IsVectorProblem> element =
            bvh.objects[bvh.nodes[node_idx + num_internal_nodes].object_idx];

        ScalarType unif2 = utils::rand_uniform<ScalarType>(random_state_ptr);
        ScalarType unif3 = utils::rand_uniform<ScalarType>(random_state_ptr);

        BoundaryPoint<ScalarType, 3, IsVectorProblem> result_point{
            element.interpolate(element.a, element.b, element.c, unif2, unif3),
            element.n,
            (ScalarType)0.5,
            element.interpolate(
                element.boundary_value_a, element.boundary_value_b, element.boundary_value_c, unif2, unif3
            ),
            element.boundary_type,
            element.interpolate(element.robin_alpha_a, element.robin_alpha_b, element.robin_alpha_c, unif2, unif3)};

        return {result_point, total_boundary_area};
    };

    inline __device__ thrust::pair<BoundaryPoint<ScalarType, 3, IsVectorProblem>, unsigned int>
    sample_boundary_line_intersection2(
        wob::randomState_t *random_state_ptr, const BoundaryPoint<ScalarType, 3, IsVectorProblem> &x
    ) const {
        typename lbvh::vector_of<ScalarType, 3>::type origin;
        origin.x = x.p[0];
        origin.y = x.p[1];
        origin.z = x.p[2];

        // sample a direction from a unit hemisphere
        // https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations
        typename lbvh::vector_of<ScalarType, 3>::type dir;
        dir.z = utils::rand_uniform<ScalarType>(random_state_ptr);
        ScalarType phi = 2.f * utils::rand_uniform<ScalarType>(random_state_ptr);
#ifdef __CUDACC__
        sincospi(phi, &dir.x, &dir.y);
#else
        dir.x = std::cos(phi * M_PI);
        dir.y = std::sin(phi * M_PI);
#endif
        ScalarType r = std::sqrt(1.f - dir.z * dir.z);
        dir.x *= r;
        dir.y *= r;
        lbvh::Line<ScalarType, 3> line{origin, dir};

        constexpr unsigned int buffer_size = 64;
        thrust::pair<unsigned int, Eigen::Matrix<ScalarType, 3, 1>> intersection_points[buffer_size];
        unsigned int num_intersections = lbvh::query_device(
            bvh, lbvh::query_line_intersect<ScalarType, 3>(line),
            LineElementXsIntersect<ScalarType, 3, IsVectorProblem>(), intersection_points, buffer_size
        );

        if (num_intersections == 0) {
            // Setting a default value here is necessary to avoid fireflies caused with undefined values.
            return {
                BoundaryPoint<ScalarType, 3, IsVectorProblem>{
                    Eigen::Matrix<ScalarType, 3, 1>::Zero(), Eigen::Matrix<ScalarType, 3, 1>::Zero(), 0.5, 0.0,
                    Dirichlet, 0.0},
                0};
        }

        num_intersections = std::min(num_intersections, buffer_size);
        unsigned int object_idx;
        Eigen::Matrix<ScalarType, 3, 1> p;
        thrust::tie(object_idx, p) = intersection_points[std::min(
            num_intersections - 1, (unsigned int)(num_intersections * utils::rand_uniform<ScalarType>(random_state_ptr))
        )];

        const ElementXs<ScalarType, 3, IsVectorProblem> &element = bvh.objects[object_idx];
        Eigen::Matrix<ScalarType, 1, 3> bc;
        barycentric_coordinates(p.transpose(), element.a.transpose(), element.b.transpose(), element.c.transpose(), bc);

        BoundaryPoint<ScalarType, 3, IsVectorProblem> result_point{
            p,
            element.n,
            0.5,
            element.interpolate(element.boundary_value_a, element.boundary_value_b, element.boundary_value_c, bc),
            element.boundary_type,
            element.interpolate(element.robin_alpha_a, element.robin_alpha_b, element.robin_alpha_c, bc)};

        return {result_point, num_intersections};
    }

    inline __device__ thrust::pair<BoundaryPoint<ScalarType, 3, IsVectorProblem>, unsigned int>
    sample_boundary_line_intersection2(wob::randomState_t *random_state_ptr, const DomainPoint<ScalarType, 3> &x)
        const {
        return sample_boundary_line_intersection2(random_state_ptr, BoundaryPoint<ScalarType, 3, IsVectorProblem>{x.p});
    }

    inline __device__ thrust::pair<BoundaryPoint<ScalarType, 3, IsVectorProblem>, ScalarType>
    sample_boundary_line_intersection(
        wob::randomState_t *random_state_ptr, const BoundaryPoint<ScalarType, 3, IsVectorProblem> &x
    ) const {
        BoundaryPoint<ScalarType, 3, IsVectorProblem> result_point;
        unsigned int num_intersections;
        thrust::tie(result_point, num_intersections) = sample_boundary_line_intersection2(random_state_ptr, x);
        constexpr ScalarType eps = 10 * std::numeric_limits<ScalarType>::epsilon();
        ScalarType inv_pdf = num_intersections * 2.f * (ScalarType)M_PI * (result_point.p - x.p).squaredNorm() /
                             std::max(std::abs((result_point.p - x.p).normalized().dot(result_point.n)), eps);
        return {result_point, inv_pdf};
    }

    inline __device__ thrust::pair<BoundaryPoint<ScalarType, 3, IsVectorProblem>, ScalarType>
    sample_boundary_line_intersection(wob::randomState_t *random_state_ptr, const DomainPoint<ScalarType, 3> &x) const {
        return sample_boundary_line_intersection(random_state_ptr, BoundaryPoint<ScalarType, 3, IsVectorProblem>{x.p});
    }

    template <class Func>
    inline __device__ thrust::pair<BoundaryPoint<ScalarType, 3, IsVectorProblem>, ScalarType>
    sample_boundary_resampled_importance_sampling(
        wob::randomState_t *random_state_ptr, Func target_distribution_func,
        const unsigned int num_resampling_candidates
    ) const {
        // Stream RIS described in ReSTIR. No spatial nor temporal reuse implemented.
        // Currently, the source distribution is fixed to the uniform distribution

        class Reservoir {
          public:
            BoundaryPoint<ScalarType, 3, IsVectorProblem> y; // the output sample
            wob::randomState_t *random_state_ptr;
            ScalarType w_sum;                       // the sum of weights
            unsigned int num_resampling_candidates; // the number of samples seen so far

            __device__ Reservoir(wob::randomState_t *random_state_ptr)
                : y(), random_state_ptr(random_state_ptr), w_sum(0.0f), num_resampling_candidates(0){};
            inline __device__ void
            update(const BoundaryPoint<ScalarType, 3, IsVectorProblem> &x_i, const ScalarType w_i) {
                w_sum += w_i;
                num_resampling_candidates += 1;
                if (utils::rand_uniform<ScalarType>(random_state_ptr) <= w_i / w_sum) y = x_i;
            };
        };

        Reservoir reservoir(random_state_ptr);

        for (unsigned int i = 0; i < num_resampling_candidates; i++) {
            BoundaryPoint<ScalarType, 3, IsVectorProblem> x_i;
            ScalarType x_i_inv_pdf;
            thrust::tie(x_i, x_i_inv_pdf) = sample_boundary_uniform(random_state_ptr);
            reservoir.update(x_i, target_distribution_func(x_i) * x_i_inv_pdf);
        }

        BoundaryPoint<ScalarType, 3, IsVectorProblem> res = reservoir.y;
        ScalarType inv_pdf = reservoir.w_sum / std::max(
                                                   target_distribution_func(res) * reservoir.num_resampling_candidates,
                                                   10 * std::numeric_limits<ScalarType>::epsilon()
                                               );

        return {res, inv_pdf};
    }

    inline __device__ thrust::pair<BoundaryPoint<ScalarType, 3, IsVectorProblem>, ScalarType>
    sample_boundary_boundary_value_sampling(wob::randomState_t *random_state_ptr) const {
        const unsigned int num_objects = bvh.num_objects;
        const unsigned int num_internal_nodes = bvh.num_objects - 1;

        ScalarType unif1 = utils::rand_uniform<ScalarType>(random_state_ptr);
        unsigned int node_idx = binary_search(boundary_value_cdf, num_objects, unif1);
        const ElementXs<ScalarType, 3, IsVectorProblem> &element =
            bvh.objects[bvh.nodes[node_idx + num_internal_nodes].object_idx];
        ScalarType pmf = node_idx == 0 ? boundary_value_cdf[node_idx]
                                       : boundary_value_cdf[node_idx] - boundary_value_cdf[node_idx - 1];

        ScalarType unif2 = utils::rand_uniform<ScalarType>(random_state_ptr);
        ScalarType unif3 = utils::rand_uniform<ScalarType>(random_state_ptr);

        // currently, once the triangle is chosen, we uniformly choose a point inside of the triangle.
        BoundaryPoint<ScalarType, 3, IsVectorProblem> result_point{
            element.interpolate(element.a, element.b, element.c, unif2, unif3),
            element.n,
            (ScalarType)0.5,
            element.interpolate(
                element.boundary_value_a, element.boundary_value_b, element.boundary_value_c, unif2, unif3
            ),
            element.boundary_type,
            element.interpolate(element.robin_alpha_a, element.robin_alpha_b, element.robin_alpha_c, unif2, unif3)};

        return {result_point, element.area() / pmf};
    }

    inline __device__ thrust::pair<
        DomainSamplePoint<ScalarType, 3, typename value_type<ScalarType, 3, IsVectorProblem>::type>, ScalarType>
    sample_volme_cache_uniform(wob::randomState_t *random_state_ptr) const {
        return volume_cache->sample_volume_cache_uniform(random_state_ptr);
    }

    inline __device__ thrust::pair<
        DomainSamplePoint<ScalarType, 3, typename value_type<ScalarType, 3, IsVectorProblem>::type>, ScalarType>
    sample_volme_cache_cache_value_sampling(wob::randomState_t *random_state_ptr) const {
        return volume_cache->sample_volme_cache_cache_value_sampling(random_state_ptr);
    }
    lbvh::bvh_device<ScalarType, 3, ElementXs<ScalarType, 3, IsVectorProblem>> bvh;
    ScalarType *area_cdf;
    ScalarType total_boundary_area;
    ScalarType *boundary_value_cdf;
    ScalarType total_boundary_value;
    thrust::optional<VolumeCache<ScalarType, 3, IsVectorProblem>> volume_cache;
};

} // namespace wob

#endif //__WOB_SCENE_CUH__