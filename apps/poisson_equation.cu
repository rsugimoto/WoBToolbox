/*
    Walk-on-Boundary Example.

    Author:         Ryusuke Sugimoto
    Affiliation:    University of Waterloo
    Date:           July 2023
    File Name:      poisson_equation.cu
    Description:    This is a simple example of Poisson's equation solver with WoB in a box domain. The solution
                    correctly accounts for the constant source term. No validation code is included in this public
                    release. Note the WoB toolbox currently does not support more general cases.
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
    unsigned int num_evaluations_per_iter;
    unsigned int grid_res;
    ScalarType eps;
    unsigned int num_volume_samples;
    wob::Scene<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> scene;
    ScalarType *solution_buffer_ptr;
    Eigen::Matrix<ScalarType, Dim, 1> *gradient_buffer_ptr;
    wob::randomState_t *random_state_buffer_ptr;
};

inline __device__ void WoBFunc(const unsigned int idx, const DeviceArgs &args) {
    wob::randomState_t random_state = args.random_state_buffer_ptr[idx];

    wob::Estimator<
        ScalarType, Dim, wob::PoissonKernel, wob::IndirectDoubleLayer, wob::InteriorDomain, wob::BackwardEstimator,
        wob::DirichletProblem>
        estimator(&random_state, args.scene, args.path_length, 1, args.eps, 1.0, args.num_volume_samples);

    wob::utils::KahanSum<Eigen::Matrix<ScalarType, Dim + 1, 1>> sum;
    Eigen::Matrix<ScalarType, Dim, 1> x = wob::utils::idx_to_domain_point<ScalarType, Dim>(idx, args.grid_res, 2.0);

    for (unsigned int i = 0; i < args.num_evaluations_per_iter; i++)
        sum += estimator.compute_sample_path_contribution_domain(wob::DomainPoint<ScalarType, Dim>{x});

    Eigen::Matrix<ScalarType, Dim + 1, 1> &solution_gradient = sum.sum;

    args.solution_buffer_ptr[idx] += solution_gradient[0];
    args.gradient_buffer_ptr[idx][0] += solution_gradient[1];
    args.gradient_buffer_ptr[idx][1] += solution_gradient[2];

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

    const std::string output_dir = config["output_dir"].get<std::string>();
    unsigned int path_length = config["path_length"].get<unsigned int>();
    unsigned int num_sample_paths = config["num_sample_paths"].get<unsigned int>();
    unsigned int num_volume_samples = config["num_volume_samples"].get<unsigned int>();
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
    std::cout << "#num volume samples\t: " << num_volume_samples << std::endl;
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

    std::vector<wob::ElementXs<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()>> elems;

    using Vector2s = Eigen::Matrix<ScalarType, 2, 1>;

    // Define a square domain
    ScalarType half_domain_size = 1.0;
    elems.push_back(
        {Vector2s(-half_domain_size, -half_domain_size), Vector2s(-half_domain_size, half_domain_size),
         Vector2s(-1.0, 0.0), 1.0f, 1.0f, wob::Dirichlet}
    );
    elems.push_back(
        {Vector2s(-half_domain_size, half_domain_size), Vector2s(half_domain_size, half_domain_size),
         Vector2s(0.0, 1.0), 1.0f, 1.0f, wob::Dirichlet}
    );
    elems.push_back(
        {Vector2s(half_domain_size, half_domain_size), Vector2s(half_domain_size, -half_domain_size),
         Vector2s(1.0, 0.0), 1.0f, 1.0f, wob::Dirichlet}
    );
    elems.push_back(
        {Vector2s(half_domain_size, -half_domain_size), Vector2s(-half_domain_size, -half_domain_size),
         Vector2s(0.0, -1.0), 1.0f, 1.0f, wob::Dirichlet}
    );

    // This cache defines the source term inside the box domain. In this example, we use a constant source term of
    // value 5.
    thrust::device_vector<ScalarType> cache(num_evaluation_points, 5.0);
    wob::SceneHost<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> scene(elems);
    scene.set_volume_cache(grid_res, 2.0, cache);

    thrust::device_vector<ScalarType> solution(num_evaluation_points, wob::utils::zero<ScalarType>());
    thrust::device_vector<Eigen::Matrix<ScalarType, Dim, 1>> gradient(
        num_evaluation_points, wob::utils::zero<Eigen::Matrix<ScalarType, Dim, 1>>()
    );

    thrust::device_vector<wob::randomState_t> random_state_buffer(num_evaluation_points);
    wob::utils::random_states_init(random_state_buffer);

    DeviceArgs args = {
        path_length,
        num_sample_paths,
        grid_res,
        eps,
        num_volume_samples,
        scene.get_device_repr(),
        solution.data().get(),
        gradient.data().get(),
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
    while (elapsed_time < max_time * 60.0) {
        num_iterations += 1;
        total_sample_count += num_sample_paths;
        auto begin_time = std::chrono::system_clock::now();

        thrust::for_each(
            thrust::make_counting_iterator<unsigned int>(0),
            thrust::make_counting_iterator<unsigned int>(num_evaluation_points),
            [args] __device__(unsigned int idx) { WoBFunc(idx, args); }
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
