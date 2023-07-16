/*
    Walk-on-Boundary Toolbox
    This file is a part of the public release of the Walk-on-Boundary (WoB) Toolbox.

    Author:         Ryusuke Sugimoto
    Affiliation:    University of Waterloo
    Date:           July 2023
    File Name:      runtime_utils.cuh
    Description:    This file defines several utility functions for applicaitons.
*/

#ifndef __WOB_RUNTIME_UTILS_CUH__
#define __WOB_RUNTIME_UTILS_CUH__

#include <Eigen/Core>
#include <chrono>
#include <ctime>
#include <fstream>
#include <random>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <vector>

#include "bie_models.cuh"
#include "equation_kernel.cuh"
#include "estimator.cuh"

namespace wob {

namespace utils {

template <typename T>
inline void save_field(const thrust::host_vector<T> &field, const std::string &filename, const unsigned int grid_size) {
    std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);

    out.write((const char *)&grid_size, sizeof(unsigned int));
    out.write((const char *)field.data(), sizeof(T) * field.size());
    out.close();
}

template <typename T>
inline void
save_field(const thrust::device_vector<T> &field, const std::string &filename, const unsigned int grid_size) {
    thrust::host_vector<T> field_host = field;
    save_field(field_host, filename, grid_size);
}

template <typename T>
inline void save_field(const std::vector<T> &field, const std::string &filename, const unsigned int grid_size) {
    std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);

    out.write((const char *)&grid_size, sizeof(unsigned int));
    out.write((const char *)field.data(), sizeof(T) * field.size());
    out.close();
}

template <typename T> inline void load_field(thrust::host_vector<T> &field, const std::string &filename) {
    std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);

    unsigned int grid_size;
    in.read((char *)&grid_size, sizeof(unsigned int));
    in.read((char *)field.data(), sizeof(T) * field.size());
    in.close();
}

template <typename T> inline void load_field(thrust::device_vector<T> &field, const std::string &filename) {
    thrust::host_vector<T> field_host = field;
    load_field(field_host, filename);
    field = field_host;
}

template <typename T> inline void laod_field(std::vector<T> &field, const std::string &filename) {
    std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);

    unsigned int grid_size;
    in.read((char *)&grid_size, sizeof(unsigned int));
    in.read((char *)field.data(), sizeof(T) * field.size());
    in.close();
}

#ifdef __CUDACC__
__device__ void random_init(unsigned long long seed, curandState_t *state) { curand_init(seed, 0, 0, state); }
#else
void random_init(unsigned long long seed, std::mt19937 *state) { *state = std::mt19937(seed); }
#endif

void random_states_init(thrust::device_vector<wob::randomState_t> &random_states) {
    std::mt19937_64 random;
    thrust::host_vector<unsigned long long> random_seeds(random_states.size());
    for (auto &seed : random_seeds) seed = random();
    thrust::device_vector<unsigned long long> random_seeds_device = random_seeds;
    wob::randomState_t *random_states_ptr = random_states.data().get();
    unsigned long long *random_seeds_ptr = random_seeds_device.data().get();
    thrust::for_each(
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(random_states.size()),
        [random_states_ptr, random_seeds_ptr] __device__(unsigned int idx) {
            random_init(random_seeds_ptr[idx], &random_states_ptr[idx]);
        }
    );
}

template <
    template <
        wob::EquationKernelType, wob::BIEModelType _bieModel, wob::DomainType _domain, wob::EstimatorType _estimator,
        wob::ProblemType _problem>
    class DeviceFunctor,
    class Args, wob::EquationKernelType _equationKernel = wob::EquationKernelType(0),
    wob::BIEModelType _bieModel = wob::BIEModelType(0), wob::DomainType _domain = wob::DomainType(0),
    wob::EstimatorType _estimator = wob::EstimatorType(0), wob::ProblemType _problem = wob::ProblemType(0)>
void run(
    Args args, unsigned int num_threads, wob::BIEModelType bieModel, wob::DomainType domain,
    wob::EstimatorType estimator, wob::ProblemType problem
) {
    if constexpr (_equationKernel == wob::EquationKernelTypeCount || _bieModel == wob::BIEModelTypeCount || _domain == wob::DomainTypeCount || _estimator == wob::EstimatorTypeCount || _problem == wob::ProblemTypeCount) {
        // This branch should be unreacheable.
        return;
    } else {
        if (bieModel > _bieModel) {
            run<DeviceFunctor, Args, _equationKernel, wob::BIEModelType(_bieModel + 1), _domain, _estimator, _problem>(
                args, num_threads, bieModel, domain, estimator, problem
            );
            return;
        }
        if (domain > _domain) {
            run<DeviceFunctor, Args, _equationKernel, _bieModel, wob::DomainType(_domain + 1), _estimator, _problem>(
                args, num_threads, bieModel, domain, estimator, problem
            );
            return;
        }
        if (estimator > _estimator) {
            run<DeviceFunctor, Args, _equationKernel, _bieModel, _domain, wob::EstimatorType(_estimator + 1), _problem>(
                args, num_threads, bieModel, domain, estimator, problem
            );
            return;
        }
        if (problem > _problem) {
            run<DeviceFunctor, Args, _equationKernel, _bieModel, _domain, _estimator, wob::ProblemType(_problem + 1)>(
                args, num_threads, bieModel, domain, estimator, problem
            );
            return;
        }

        thrust::for_each(
            thrust::make_counting_iterator<unsigned int>(0), thrust::make_counting_iterator<unsigned int>(num_threads),
            [args] __device__(unsigned int idx) {
                DeviceFunctor<_equationKernel, _bieModel, _domain, _estimator, _problem>()(idx, args);
            }
        );
    }
}

template <
    template <
        wob::EquationKernelType, wob::BIEModelType _bieModel, wob::DomainType _domain, wob::EstimatorType _estimator,
        wob::ProblemType _problem>
    class DeviceFunctor,
    wob::EquationKernelType _equationKernel, wob::BIEModelType _bieModel, wob::DomainType _domain,
    wob::EstimatorType _estimator, wob::ProblemType _problem, class Args>
void run(Args args, unsigned int num_threads) {
    thrust::for_each(
        thrust::make_counting_iterator<unsigned int>(0), thrust::make_counting_iterator<unsigned int>(num_threads),
        [args] __device__(unsigned int idx) {
            DeviceFunctor<_equationKernel, _bieModel, _domain, _estimator, _problem>()(idx, args);
        }
    );
}

} // namespace utils

} // namespace wob

inline std::ostream &operator<<(std::ostream &os, const std::chrono::time_point<std::chrono::system_clock> &time) {
    auto _time = std::chrono::system_clock::to_time_t(time);
    std::string time_str = std::ctime(&_time);
    time_str.resize(time_str.size() - 1);
    os << time_str;
    return os;
}

#endif // __WOB_RUNTIME_UTILS_CUH__