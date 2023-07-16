/*
    Walk-on-Boundary Example.

    Author:         Ryusuke Sugimoto
    Affiliation:    University of Waterloo
    Date:           July 2023
    File Name:      potential_flow_with_vpl.cu
    Description:    This sample demonstrates the use of a variant of the Virtual Point Light method (VPL) for WoB.
                    For the given Neumann problem, the solution and its gradient are computed by using the VPL method.
                    The single layer potential is used in combination with backward estimator using resampled importance
                    sampling of the integral kernel function.
*/

#include "wob/wob.cuh"

#include <filesystem>
#include <iostream>
#include <string>
#include <thrust/device_vector.h>
#include <vector>

#include <igl/readOBJ.h>
#include <igl/winding_number.h>
#include <nlohmann/json.hpp>

using ScalarType = float;
constexpr unsigned int Dim = 2;

struct DeviceArgs {
    unsigned int path_length;
    unsigned int num_path_computation_threads;
    unsigned int num_resampling_candidates;
    unsigned int grid_res;
    ScalarType eps;
    wob::Scene<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> scene;
    ScalarType *solution_buffer_ptr;
    Eigen::Matrix<ScalarType, Dim, 1> *gradient_buffer_ptr;
    wob::BoundaryPoint<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> *path_start_point_ptr;
    ScalarType *path_contribution_ptr;
    bool *evaluate_point_buffer_ptr;
    wob::randomState_t *random_state_buffer_ptr;
};

inline __device__ void AggregateFunc(const unsigned int idx, const DeviceArgs &args) {
    if (!args.evaluate_point_buffer_ptr[idx]) {
        args.solution_buffer_ptr[idx] = std::numeric_limits<ScalarType>::quiet_NaN();
        return;
    }

    wob::BIEModel<ScalarType, Dim, wob::PoissonKernel, wob::IndirectSingleLayer, wob::InteriorDomain> bie_model;

    wob::DomainPoint<ScalarType, Dim> x = {wob::utils::idx_to_domain_point<ScalarType, Dim>(idx, args.grid_res, 1.0)};

    wob::utils::KahanSum<Eigen::Matrix<ScalarType, Dim + 1, 1>> sum;
    for (unsigned int i = 0; i < args.num_path_computation_threads; i++) {
        wob::BoundaryPoint<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> y =
            args.path_start_point_ptr[i];
        ScalarType sample_path_contribution = args.path_contribution_ptr[i];

        Eigen::Matrix<ScalarType, Dim + 1, 1> result;
        result[0] = sample_path_contribution * bie_model.solution_domain_unknown(x, y, args.scene.total_boundary_area);
        result.template tail<Dim>() =
            sample_path_contribution * bie_model.gradient_domain_unknown(x, y, args.scene.total_boundary_area);
        sum += result;
    }

    Eigen::Matrix<ScalarType, Dim + 1, 1> &solution_gradient = sum.sum;

    args.solution_buffer_ptr[idx] += solution_gradient[0];
    args.gradient_buffer_ptr[idx] += solution_gradient.tail<Dim>();
}

inline __device__ void PathComputeFunc(const unsigned int idx, const DeviceArgs &args) {
    wob::randomState_t *random_state_ptr = &args.random_state_buffer_ptr[idx];

    wob::Estimator<
        ScalarType, Dim, wob::PoissonKernel, wob::IndirectSingleLayer, wob::InteriorDomain, wob::BackwardEstimator,
        wob::NeumannProblem>
        estimator(random_state_ptr, args.scene, args.path_length, args.num_resampling_candidates, args.eps);

    wob::BoundaryPoint<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> y;
    ScalarType inv_pdf;
    thrust::tie(y, inv_pdf) = args.scene.sample_boundary_uniform(random_state_ptr);
    ScalarType sample_path_contribution = estimator.compute_sample_unknown_boundary(y);

    args.path_start_point_ptr[idx] = y;
    args.path_contribution_ptr[idx] = sample_path_contribution;
}

int main(int argc, char *argv[]) {
    const std::string input_json_file = argv[1];
    using json = nlohmann::json;
    json config;
    {
        std::ifstream config_file(input_json_file);
        if (!config_file) {
            std::cout << "Failed to load config file: " << input_json_file << std::endl;
            return 0;
        }
        config_file >> config;
        config_file.close();
    }

    const std::string output_dir = config["output_dir"].get<std::string>();
    unsigned int path_length = config["path_length"].get<unsigned int>();
    unsigned int num_sample_paths = config["num_sample_paths"].get<unsigned int>();
    unsigned int num_resampling_candidates =
        config.contains("num_resampling_candidates") ? config["num_resampling_candidates"].get<unsigned int>() : 1;
    unsigned int grid_res = config["grid_res"].get<unsigned int>();
    ScalarType eps =
        config.contains("eps") ? config["eps"].get<ScalarType>() : std::numeric_limits<ScalarType>::epsilon();
    unsigned int num_evaluation_points = grid_res * grid_res;
    ScalarType max_time = config["max_time"].get<ScalarType>();

    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "input json file\t: " << input_json_file << std::endl;
    std::cout << "output directory\t: " << output_dir << std::endl;
    std::cout << "path length\t: " << path_length << std::endl;
    std::cout << "#sample paths per iteration\t: " << num_sample_paths << std::endl;
    std::cout << "#resampling candidates\t: " << num_resampling_candidates << std::endl;
    std::cout << "grid resolution\t: " << grid_res << std::endl;
    std::cout << "eps\t: " << eps << std::endl;
    std::cout << "precision\t: ";
    if constexpr (std::is_same<ScalarType, float>::value)
        std::cout << "float" << std::endl;
    else
        std::cout << "double" << std::endl;
    std::cout << "max time\t: " << max_time << " minutes" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    std::filesystem::create_directories(output_dir);
    std::filesystem::copy(
        input_json_file, output_dir + "/config.json", std::filesystem::copy_options::overwrite_existing
    );

    // define the scene
    std::vector<wob::ElementXs<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()>> elems;

    using Vector2s = Eigen::Matrix<ScalarType, 2, 1>;

    elems.push_back({Vector2s(-0.5, -0.5), Vector2s(-0.5, 0.5), Vector2s(-1.0, 0.0), -1.0, -1.0, wob::Neumann});
    elems.push_back({Vector2s(-0.5, 0.5), Vector2s(0.5, 0.5), Vector2s(0.0, 1.0), 0.0, 0.0, wob::Neumann});
    elems.push_back({Vector2s(0.5, 0.5), Vector2s(0.5, -0.5), Vector2s(1.0, 0.0), 1.0, 1.0, wob::Neumann});
    elems.push_back({Vector2s(0.5, -0.5), Vector2s(-0.5, -0.5), Vector2s(0.0, -1.0), 0.0, 0.0, wob::Neumann});

    struct Circle {
        Eigen::Matrix<ScalarType, Dim, 1> center;
        ScalarType radius;
        unsigned int num_segments;
    };

    std::vector<Circle> circles;
    circles.push_back({Eigen::Matrix<ScalarType, Dim, 1>(0.2, 0.1), 0.25, 360});
    circles.push_back({Eigen::Matrix<ScalarType, Dim, 1>(-0.25, 0.3), 0.15, 180});
    circles.push_back({Eigen::Matrix<ScalarType, Dim, 1>(-0.20, -0.25), 0.2, 180});
    circles.push_back({Eigen::Matrix<ScalarType, Dim, 1>(0.25, -0.3), 0.05, 90});

    for (const auto &circle : circles) {
        for (int i = 0; i < circle.num_segments; i++) {
            Eigen::Matrix<ScalarType, Dim, 1> v1 =
                circle.center + circle.radius * Eigen::Matrix<ScalarType, Dim, 1>(
                                                    std::cos((2 * M_PI) * i / circle.num_segments),
                                                    std::sin((2 * M_PI) * i / circle.num_segments)
                                                );
            Eigen::Matrix<ScalarType, Dim, 1> v2 =
                circle.center + circle.radius * Eigen::Matrix<ScalarType, Dim, 1>(
                                                    std::cos((2 * M_PI) * (i + 1) / circle.num_segments),
                                                    std::sin((2 * M_PI) * (i + 1) / circle.num_segments)
                                                );

            Eigen::Matrix<ScalarType, Dim, 1> t = (v1 - v2).normalized();
            Eigen::Matrix<ScalarType, Dim, 1> n(t[1], -t[0]);
            elems.push_back({v1, v2, n, 0.0, 0.0, wob::Neumann});
        }
    }

    thrust::host_vector<bool> evaluate_point_host(num_evaluation_points);
    for (size_t i = 0; i < num_evaluation_points; i++) {
        Eigen::Matrix<ScalarType, Dim, 1> x = wob::utils::idx_to_domain_point<ScalarType, Dim>(i, grid_res, 1.0);
        evaluate_point_host[i] = true;
        for (const auto &circle : circles) {
            if ((x - circle.center).squaredNorm() < circle.radius * circle.radius) evaluate_point_host[i] = false;
        }
    }
    thrust::device_vector<bool> evaluate_point(evaluate_point_host);

    wob::SceneHost<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> scene(elems);
    thrust::device_vector<ScalarType> solution(num_evaluation_points, wob::utils::zero<ScalarType>());
    thrust::device_vector<Eigen::Matrix<ScalarType, Dim, 1>> gradient(
        num_evaluation_points, wob::utils::zero<Eigen::Matrix<ScalarType, Dim, 1>>()
    );

    thrust::device_vector<wob::randomState_t> random_state_buffer(num_sample_paths);
    wob::utils::random_states_init(random_state_buffer);

    // These two buffers store the boundary cache location and the cache value.
    thrust::device_vector<wob::BoundaryPoint<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()>>
        cache_point_location_buffer(num_sample_paths);
    thrust::device_vector<ScalarType> cache_point_value_buffer(num_sample_paths);

    DeviceArgs args = {
        path_length,
        num_sample_paths,
        num_resampling_candidates,
        grid_res,
        eps,
        scene.get_device_repr(),
        solution.data().get(),
        gradient.data().get(),
        cache_point_location_buffer.data().get(),
        cache_point_value_buffer.data().get(),
        evaluate_point.data().get(),
        random_state_buffer.data().get()};

    auto start_time = std::chrono::system_clock::now();
    std::cout << "start time\t: " << start_time << std::endl;

    unsigned int num_iterations = 0;
    unsigned int total_sample_count = 0;
    double elapsed_time = 0.0;
    thrust::device_vector<ScalarType> _solution(num_evaluation_points, wob::utils::zero<ScalarType>());
    ScalarType *_solution_ptr = _solution.data().get();
    thrust::device_vector<Eigen::Matrix<ScalarType, Dim, 1>> _gradient(
        num_evaluation_points, wob::utils::zero<Eigen::Matrix<ScalarType, Dim, 1>>()
    );
    Eigen::Matrix<ScalarType, Dim, 1> *_gradient_ptr = _gradient.data().get();
    while (elapsed_time < max_time * 60.0) {
        num_iterations += 1;
        total_sample_count += num_sample_paths;
        auto begin_time = std::chrono::system_clock::now();

        thrust::for_each(
            thrust::make_counting_iterator<unsigned int>(0),
            thrust::make_counting_iterator<unsigned int>(num_sample_paths),
            [args] __device__(unsigned int idx) { PathComputeFunc(idx, args); }
        );

        thrust::for_each(
            thrust::make_counting_iterator<unsigned int>(0),
            thrust::make_counting_iterator<unsigned int>(num_evaluation_points),
            [args] __device__(unsigned int idx) { AggregateFunc(idx, args); }
        );

        auto end_time = std::chrono::system_clock::now();
        elapsed_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count() / 1000.0;
        std::cout << "\33[2K\r"; // clear line
        std::cout << "time " << elapsed_time << " s\t" << num_iterations << " iterations (" << total_sample_count
                  << " samples)\t" << std::flush;

        thrust::for_each(
            thrust::device, thrust::make_counting_iterator<unsigned int>(0),
            thrust::make_counting_iterator<unsigned int>(num_evaluation_points),
            [args, _solution_ptr, _gradient_ptr, total_sample_count] __device__(unsigned int idx) {
                _solution_ptr[idx] = args.solution_buffer_ptr[idx] / total_sample_count;
                _gradient_ptr[idx] = args.gradient_buffer_ptr[idx] / total_sample_count;
            }
        );

        wob::utils::save_field(
            _solution, output_dir + "/potential" + std::to_string(num_iterations) + ".scalar", grid_res
        );
        wob::utils::save_field(
            _gradient, output_dir + "/gradient" + std::to_string(num_iterations) + ".vector2", grid_res
        );
    }
    std::cout << std::endl;
}
