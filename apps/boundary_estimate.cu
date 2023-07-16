/*
    Walk-on-Boundary Example.

    Author:         Ryusuke Sugimoto
    Affiliation:    University of Waterloo
    Date:           July 2023
    File Name:      boundary_estimate.cu
    Description:    This example estimates the solution in the domain and exactly on the boundary for an interior Nemann
                    problem.
*/

#include "wob/wob.cuh"

#include <filesystem>
#include <iostream>
#include <string>
#include <thrust/device_vector.h>
#include <vector>

#include <igl/readOBJ.h>
#include <nlohmann/json.hpp>

using ScalarType = float;
constexpr unsigned int Dim = 2;

struct DeviceArgs {
    unsigned int path_length;
    unsigned int num_evaluations_per_iter;
    unsigned int grid_res;
    ScalarType eps;
    wob::Scene<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> scene;
    ScalarType *solution_buffer_ptr;
    Eigen::Matrix<ScalarType, Dim, 1> *gradient_buffer_ptr;
    bool *evaluate_point_buffer_ptr;
    wob::randomState_t *random_state_buffer_ptr;
};

inline __device__ void domainEstimatorFunc(const unsigned int idx, const DeviceArgs &args) {
    if (!args.evaluate_point_buffer_ptr[idx]) {
        args.solution_buffer_ptr[idx] = std::numeric_limits<ScalarType>::quiet_NaN();
        return;
    }
    wob::randomState_t random_state = args.random_state_buffer_ptr[idx];

    wob::Estimator<
        ScalarType, Dim, wob::PoissonKernel, wob::IndirectSingleLayer, wob::InteriorDomain, wob::ForwardEstimator,
        wob::NeumannProblem>
        estimator(&random_state, args.scene, args.path_length, 1, args.eps, 1.0);

    wob::utils::KahanSum<Eigen::Matrix<ScalarType, Dim + 1, 1>> sum;
    Eigen::Matrix<ScalarType, Dim, 1> x = wob::utils::idx_to_domain_point<ScalarType, Dim>(idx, args.grid_res, 1.0);
    for (unsigned int i = 0; i < args.num_evaluations_per_iter; i++) {
        sum += estimator.compute_sample_path_contribution_domain(wob::DomainPoint<ScalarType, Dim>{x});
    }

    Eigen::Matrix<ScalarType, Dim + 1, 1> &solution_gradient = sum.sum;

    args.solution_buffer_ptr[idx] += solution_gradient[0];
    args.gradient_buffer_ptr[idx][0] += solution_gradient[1];
    args.gradient_buffer_ptr[idx][1] += solution_gradient[2];

    args.random_state_buffer_ptr[idx] = random_state;
}

inline __device__ void boundaryEstimatorFunc(const unsigned int idx, const DeviceArgs &args) {
    wob::randomState_t random_state = args.random_state_buffer_ptr[idx];

    wob::Estimator<
        ScalarType, Dim, wob::PoissonKernel, wob::IndirectSingleLayer, wob::InteriorDomain, wob::ForwardEstimator,
        wob::NeumannProblem>
        estimator(&random_state, args.scene, args.path_length, 1, args.eps, 1.0);

    ScalarType radius = 0.5;
    unsigned int num_segments = 720;
    Eigen::Matrix<ScalarType, Dim, 1> v1 = Eigen::Matrix<ScalarType, Dim, 1>(
        radius * std::sin((2 * M_PI) * idx / num_segments), radius * std::cos((2 * M_PI) * idx / num_segments)
    );
    Eigen::Matrix<ScalarType, Dim, 1> v2 = Eigen::Matrix<ScalarType, Dim, 1>(
        radius * std::sin((2 * M_PI) * (idx + 1) / num_segments),
        radius * std::cos((2 * M_PI) * (idx + 1) / num_segments)
    );
    Eigen::Matrix<ScalarType, Dim, 1> t = (v1 - v2).normalized();
    Eigen::Matrix<ScalarType, Dim, 1> n(t[1], -t[0]);
    Eigen::Matrix<ScalarType, Dim, 1> x = (v1 + v2) / 2.0;

    wob::utils::KahanSum<ScalarType> sum;
    for (unsigned int i = 0; i < args.num_evaluations_per_iter; i++) {
        sum += estimator.compute_sample_path_contribution_boundary(
            wob::BoundaryPoint<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()>{
                x, n, 1. / 2, 0.0, wob::Neumann}
        );
    }

    ScalarType solution = sum.sum;

    args.solution_buffer_ptr[idx] += solution;
    args.random_state_buffer_ptr[idx] = random_state;
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

    // const std::string input_obj_file = config["input_obj_file"].get<std::string>();
    const std::string output_dir = config["output_dir"].get<std::string>();
    unsigned int path_length = config["path_length"].get<unsigned int>();
    unsigned int num_sample_paths_domain = config["num_sample_paths_domain"].get<unsigned int>();
    unsigned int num_sample_paths_boundary = config["num_sample_paths_boundary"].get<unsigned int>();

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
    std::cout << "#sample paths domain per iteration\t: " << num_sample_paths_domain << std::endl;
    std::cout << "#sample paths boundary per iteration\t: " << num_sample_paths_boundary << std::endl;
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

    unsigned int num_segments = 720;
    for (int i = 0; i < num_segments; i++) {
        ScalarType radius = 0.5;
        Eigen::Matrix<ScalarType, Dim, 1> v1 = Eigen::Matrix<ScalarType, Dim, 1>(
            radius * std::sin((2 * M_PI) * i / num_segments), radius * std::cos((2 * M_PI) * i / num_segments)
        );
        Eigen::Matrix<ScalarType, Dim, 1> v2 = Eigen::Matrix<ScalarType, Dim, 1>(
            radius * std::sin((2 * M_PI) * (i + 1) / num_segments),
            radius * std::cos((2 * M_PI) * (i + 1) / num_segments)
        );
        Eigen::Matrix<ScalarType, Dim, 1> t = (v1 - v2).normalized();
        Eigen::Matrix<ScalarType, Dim, 1> n(t[1], -t[0]);

        const ScalarType neumann_boundary_value = 200.0;
        if (i % 240 == 0 || i % 240 == 239)
            elems.push_back({v1, v2, -n, 1.0, neumann_boundary_value, wob::Neumann});
        else if (i % 240 == 120 || i % 240 == 119)
            elems.push_back({v1, v2, -n, -1.0, -neumann_boundary_value, wob::Neumann});
        else
            elems.push_back({v1, v2, -n, 0.0, 0.0, wob::Neumann});
    }

    thrust::host_vector<bool> evaluate_point_host(num_evaluation_points);
    for (size_t i = 0; i < num_evaluation_points; i++) {
        Eigen::Matrix<ScalarType, Dim, 1> x = wob::utils::idx_to_domain_point<ScalarType, Dim>(i, grid_res, 1.0);
        evaluate_point_host[i] = x.norm() <= 0.5;
    }

    wob::SceneHost<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> scene(elems);

    thrust::device_vector<ScalarType> solution(num_evaluation_points, wob::utils::zero<ScalarType>());
    thrust::device_vector<Eigen::Matrix<ScalarType, Dim, 1>> gradient(
        num_evaluation_points, wob::utils::zero<Eigen::Matrix<ScalarType, Dim, 1>>()
    );
    thrust::device_vector<bool> evaluate_point(evaluate_point_host);

    thrust::device_vector<ScalarType> boundary_solution(num_segments, wob::utils::zero<ScalarType>());

    thrust::device_vector<wob::randomState_t> random_state_buffer(std::max(num_evaluation_points, num_segments));
    wob::utils::random_states_init(random_state_buffer);

    DeviceArgs args = {
        path_length,
        num_sample_paths_domain,
        grid_res,
        eps,
        scene.get_device_repr(),
        solution.data().get(),
        gradient.data().get(),
        evaluate_point.data().get(),
        random_state_buffer.data().get()};

    unsigned int num_iterations = 0;
    unsigned int total_sample_count = 0;
    double elapsed_time = 0.0;
    thrust::device_vector<ScalarType> _solution(num_evaluation_points, wob::utils::zero<ScalarType>());
    ScalarType *_solution_ptr = _solution.data().get();
    thrust::device_vector<Eigen::Matrix<ScalarType, Dim, 1>> _gradient(
        num_evaluation_points, wob::utils::zero<Eigen::Matrix<ScalarType, Dim, 1>>()
    );
    Eigen::Matrix<ScalarType, Dim, 1> *_gradient_ptr = _gradient.data().get();
    std::cout << "computing solution in the domain" << std::endl;
    while (elapsed_time < max_time * 60.0) {
        num_iterations += 1;
        total_sample_count += num_sample_paths_domain;
        auto begin_time = std::chrono::system_clock::now();

        thrust::for_each(
            thrust::device, thrust::make_counting_iterator<unsigned int>(0),
            thrust::make_counting_iterator<unsigned int>(num_evaluation_points),
            [args] __device__(unsigned int idx) { domainEstimatorFunc(idx, args); }
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

    num_iterations = 0;
    total_sample_count = 0;
    elapsed_time = 0.0;
    thrust::device_vector<ScalarType> _boundary_solution(num_segments, wob::utils::zero<ScalarType>());
    ScalarType *_boundary_solution_ptr = _boundary_solution.data().get();
    args.solution_buffer_ptr = boundary_solution.data().get();
    args.num_evaluations_per_iter = num_sample_paths_boundary;
    std::cout << "computing solution on boundary" << std::endl;
    while (elapsed_time < max_time * 60.0) {
        num_iterations += 1;
        total_sample_count += num_sample_paths_boundary;
        auto begin_time = std::chrono::system_clock::now();

        thrust::for_each(
            thrust::device, thrust::make_counting_iterator<unsigned int>(0),
            thrust::make_counting_iterator<unsigned int>(num_segments),
            [args] __device__(unsigned int idx) { boundaryEstimatorFunc(idx, args); }
        );

        auto end_time = std::chrono::system_clock::now();
        elapsed_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count() / 1000.0;
        std::cout << "\33[2K\r"; // clear line
        std::cout << "time " << elapsed_time << " s\t" << num_iterations << " iterations (" << total_sample_count
                  << " samples)\t" << std::flush;

        thrust::for_each(
            thrust::device, thrust::make_counting_iterator<unsigned int>(0),
            thrust::make_counting_iterator<unsigned int>(num_segments),
            [args, _boundary_solution_ptr, total_sample_count] __device__(unsigned int idx) {
                _boundary_solution_ptr[idx] = args.solution_buffer_ptr[idx] / total_sample_count;
            }
        );

        // save the solution estimate on boundary
        {
            thrust::host_vector<ScalarType> boundary_solution_host = _boundary_solution;
            std::string filename(output_dir + "/boundary_solution" + std::to_string(num_iterations) + ".scalar");
            std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);
            out.write((const char *)&num_segments, sizeof(unsigned int));
            out.write((const char *)boundary_solution_host.data(), sizeof(ScalarType) * boundary_solution_host.size());
            out.close();
        }
    }
    std::cout << std::endl;
}
