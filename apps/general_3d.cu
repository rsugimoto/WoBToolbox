/*
    Walk-on-Boundary Example.

    Author:         Ryusuke Sugimoto
    Affiliation:    University of Waterloo
    Date:           July 2023
    File Name:      general_3d.cu
    Description:    This example computes the solution of the Laplace equation in interior or exterior domain.
                    You can change the configuration and boudnary type using a configuraiotn file.
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
constexpr unsigned int Dim = 3;

struct DeviceArgs {
    unsigned int path_length;
    unsigned int num_sample_paths;
    unsigned int num_resampling_candidates;
    unsigned int grid_res;
    ScalarType eps, first_kind_equation_scaling_constant;
    wob::Scene<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> scene;
    ScalarType *solution_buffer_ptr;
    Eigen::Matrix<ScalarType, Dim, 1> *gradient_buffer_ptr;
    bool *evaluate_point_buffer_ptr;
    wob::randomState_t *random_state_buffer_ptr;
};

template <
    wob::EquationKernelType _equationKernel, wob::BIEModelType _bieModel, wob::DomainType _domain,
    wob::EstimatorType _estimator, wob::ProblemType _problem>
struct DeviceFunctor {
    inline __device__ void operator()(const unsigned int idx, const DeviceArgs &args) {
        if (!args.evaluate_point_buffer_ptr[idx]) {
            args.solution_buffer_ptr[idx] = std::numeric_limits<ScalarType>::quiet_NaN();
            return;
        }

        wob::randomState_t random_state = args.random_state_buffer_ptr[idx];

        wob::Estimator<ScalarType, Dim, _equationKernel, _bieModel, _domain, _estimator, _problem> estimator(
            &random_state, args.scene, args.path_length, args.num_resampling_candidates, args.eps,
            args.first_kind_equation_scaling_constant
        );

        auto x_2d = wob::utils::idx_to_domain_point<ScalarType, 2>(idx, args.grid_res, 1.0);
        Eigen::Matrix<ScalarType, Dim, 1> x(0.0, x_2d[1], x_2d[0]);

        wob::utils::KahanSum<Eigen::Matrix<ScalarType, Dim + 1, 1>> sum;
        for (unsigned int i = 0; i < args.num_sample_paths; i++)
            sum += estimator.compute_sample_path_contribution_domain(wob::DomainPoint<ScalarType, Dim>{x});

        Eigen::Matrix<ScalarType, Dim + 1, 1> &solution_gradient = sum.sum;

        args.solution_buffer_ptr[idx] += solution_gradient[0];
        args.gradient_buffer_ptr[idx][0] += solution_gradient[1];
        args.gradient_buffer_ptr[idx][1] += solution_gradient[2];
        args.gradient_buffer_ptr[idx][2] += solution_gradient[3];

        args.random_state_buffer_ptr[idx] = random_state;
    }
};

enum ANALYTICAL_SOLUTION_TYPE { INTERIOR_SOLUTION1, INTERIOR_SOLUTION2, EXTERIOR_SOLUTION1 };

Eigen::Matrix<ScalarType, Dim + 1, 1>
analytical_solution(ANALYTICAL_SOLUTION_TYPE solution_type, const Eigen::Matrix<ScalarType, Dim, 1> &y) {
    switch (solution_type) {
    case EXTERIOR_SOLUTION1:
        return Eigen::Matrix<ScalarType, Dim + 1, 1>(
            y[2] / (y.squaredNorm() * y.norm()), -3.f * y[2] * y[0] / (y.squaredNorm() * y.squaredNorm() * y.norm()),
            -3.f * y[2] * y[1] / (y.squaredNorm() * y.squaredNorm() * y.norm()),
            1.f / (y.squaredNorm() * y.norm()) - 3.f * y[2] * y[2] / (y.squaredNorm() * y.squaredNorm() * y.norm())
        );

    case INTERIOR_SOLUTION1: return Eigen::Matrix<ScalarType, Dim + 1, 1>(y[1], 0.0, 1.0, 0.0);

    case INTERIOR_SOLUTION2:
        const ScalarType scale = (ScalarType)M_PI;
        return Eigen::Matrix<ScalarType, Dim + 1, 1>(
            -std::cos(scale * y[1]) * std::sinh(scale * y[2]), 0.0,
            scale * std::sin(scale * y[1]) * std::sinh(scale * y[2]),
            -scale * std::cos(scale * y[1]) * std::cosh(scale * y[2])
        );
    }
    return Eigen::Matrix<ScalarType, Dim + 1, 1>();
}
ScalarType get_dirichlet_bc(ANALYTICAL_SOLUTION_TYPE solution_type, const Eigen::Matrix<ScalarType, Dim, 1> &y) {
    return analytical_solution(solution_type, y)[0];
}
ScalarType get_neumann_bc(
    ANALYTICAL_SOLUTION_TYPE solution_type, const Eigen::Matrix<ScalarType, Dim, 1> &y,
    const Eigen::Matrix<ScalarType, Dim, 1> &yn
) {
    Eigen::Matrix<ScalarType, Dim, 1> gradient = analytical_solution(solution_type, y).tail<3>();
    return gradient.dot(yn);
}

void shift_solution(
    thrust::device_vector<ScalarType> &_solution, const thrust::host_vector<bool> &evaluation_point,
    unsigned int grid_res, ANALYTICAL_SOLUTION_TYPE solution_type, wob::DomainType domain_type
) {
    thrust::host_vector<ScalarType> solution(_solution);
    unsigned int ref_point_idx = domain_type == wob::InteriorDomain ? grid_res * (grid_res / 2) + grid_res / 2 : 0;
    Eigen::Matrix<ScalarType, Dim, 1> x_ref =
        wob::utils::idx_to_domain_point<ScalarType, Dim>(ref_point_idx, grid_res, 1.0);
    Eigen::Matrix<ScalarType, Dim + 1, 1> analytical_solution_gradient_ref = analytical_solution(solution_type, x_ref);
    ScalarType offset = solution[ref_point_idx] - analytical_solution_gradient_ref[0];
    for (unsigned int i = 0; i < solution.size(); i++) {
        if (!evaluation_point[i]) continue;
        solution[i] -= offset;
    }
    _solution = solution;
}

std::tuple<ScalarType, ScalarType, ScalarType, ScalarType> compute_rmse(
    const thrust::device_vector<ScalarType> &_solution,
    const thrust::device_vector<Eigen::Matrix<ScalarType, Dim, 1>> &_gradient,
    const thrust::host_vector<bool> &evaluation_point, unsigned int grid_res, ANALYTICAL_SOLUTION_TYPE solution_type,
    wob::ProblemType problem
) {
    thrust::host_vector<ScalarType> solution(_solution);
    thrust::host_vector<Eigen::Matrix<ScalarType, Dim, 1>> gradient(_gradient);

    wob::utils::KahanSum<ScalarType> solution_squared_error_sum;
    wob::utils::KahanSum<ScalarType> x_gradient_squared_error_sum;
    wob::utils::KahanSum<ScalarType> y_gradient_squared_error_sum;
    wob::utils::KahanSum<ScalarType> z_gradient_squared_error_sum;

    unsigned int num_valid_grid_entries = 0;
    for (unsigned int i = 0; i < solution.size(); i++) {
        if (!evaluation_point[i]) continue;
        auto x_2d = wob::utils::idx_to_domain_point<ScalarType, 2>(i, grid_res, 1.0);
        Eigen::Matrix<ScalarType, Dim, 1> x(0.0, x_2d[1], x_2d[0]);
        Eigen::Matrix<ScalarType, Dim + 1, 1> analytical_solution_gradient = analytical_solution(solution_type, x);

        solution_squared_error_sum += pow(solution[i] - analytical_solution_gradient[0], 2);
        x_gradient_squared_error_sum += pow(gradient[i][0] - analytical_solution_gradient[1], 2);
        y_gradient_squared_error_sum += pow(gradient[i][1] - analytical_solution_gradient[2], 2);
        z_gradient_squared_error_sum += pow(gradient[i][2] - analytical_solution_gradient[3], 2);

        num_valid_grid_entries++;
    }

    ScalarType solution_rmse = std::sqrt(solution_squared_error_sum.sum / num_valid_grid_entries);
    ScalarType x_gradient_rmse = std::sqrt(x_gradient_squared_error_sum.sum / num_valid_grid_entries);
    ScalarType y_gradient_rmse = std::sqrt(y_gradient_squared_error_sum.sum / num_valid_grid_entries);
    ScalarType z_gradient_rmse = std::sqrt(z_gradient_squared_error_sum.sum / num_valid_grid_entries);

    return {solution_rmse, x_gradient_rmse, y_gradient_rmse, z_gradient_rmse};
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

    const std::string input_obj_file = config["input_obj_file"].get<std::string>();
    const std::string output_dir = config["output_dir"].get<std::string>();
    unsigned int path_length = config["path_length"].get<unsigned int>();
    unsigned int num_sample_paths = config["num_sample_paths"].get<unsigned int>();
    unsigned int num_resampling_candidates =
        config.contains("num_resampling_candidates") ? config["num_resampling_candidates"].get<unsigned int>() : 1;
    unsigned int grid_res = config["grid_res"].get<unsigned int>();
    ScalarType eps =
        config.contains("eps") ? config["eps"].get<ScalarType>() : std::numeric_limits<ScalarType>::epsilon();
    ScalarType first_kind_equation_scaling_constant =
        config.contains("first_kind_equation_scaling_constant")
            ? config["first_kind_equation_scaling_constant"].get<ScalarType>()
            : 1.0;
    unsigned int num_evaluation_points = grid_res * grid_res;
    ScalarType max_time = config["max_time"].get<ScalarType>();

    wob::EquationKernelType equation_kernel =
        wob::string_to_equation_kernel_type(config["equation_kernel"].get<std::string>());
    wob::BIEModelType bie_model = wob::string_to_bie_model_type(config["bie_model"].get<std::string>());
    wob::DomainType domain = wob::string_to_domain_type(config["domain"].get<std::string>());
    wob::EstimatorType estimator = wob::string_to_estimator_type(config["estimator"].get<std::string>());
    wob::ProblemType problem = wob::string_to_problem_type(config["problem"].get<std::string>());

    Eigen::Matrix<ScalarType, Eigen::Dynamic, 3> V;
    Eigen::Matrix<int, Eigen::Dynamic, 3> F;
    igl::readOBJ(input_obj_file, V, F);

    // resize the mesh so it fits in [-.5, .5]^3
    auto min = V.colwise().minCoeff().eval();
    auto max = V.colwise().maxCoeff().eval();
    auto size = (max - min).eval();
    ScalarType _size = size.maxCoeff();
    V.rowwise() -= (min + 0.5f * size);
    V *= 1. / _size;

    if (domain == wob::ExteriorDomain) {
        V *= 0.5f;
        V.col(1).array() += 0.05f;
    }

    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "input json file\t: " << input_json_file << std::endl;
    std::cout << "input obj file\t: " << input_obj_file << std::endl;
    std::cout << " ( #faces: " << F.rows() << " )" << std::endl;
    std::cout << "output directory\t: " << output_dir << std::endl;
    std::cout << "path length\t: " << path_length << std::endl;
    std::cout << "#sample paths\t: " << num_sample_paths << std::endl;
    std::cout << "#resampling candidates\t: " << num_resampling_candidates << std::endl;
    std::cout << "grid resolution\t: " << grid_res << std::endl;
    std::cout << "eps\t: " << eps << std::endl;
    std::cout << "first kind equation scaling constant\t: " << first_kind_equation_scaling_constant << std::endl;
    std::cout << "precision\t: ";
    if constexpr (std::is_same<ScalarType, float>::value)
        std::cout << "float" << std::endl;
    else
        std::cout << "double" << std::endl;
    std::cout << "max time\t: " << max_time << " minutes" << std::endl;
    std::cout << "equation type\t: " << wob::equation_kernel_type_to_string(equation_kernel) << std::endl;
    std::cout << "BIE model type\t: " << wob::bie_model_type_to_string(bie_model) << std::endl;
    std::cout << "domain type\t: " << wob::domain_type_to_string(domain) << std::endl;
    std::cout << "estimator type\t: " << wob::estimator_type_to_string(estimator) << std::endl;
    std::cout << "problem type\t: " << wob::problem_type_to_string(problem) << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    std::filesystem::create_directories(output_dir);
    std::filesystem::copy(
        input_json_file, output_dir + "/config.json", std::filesystem::copy_options::overwrite_existing
    );
    std::ofstream rmse_out(output_dir + "/rmse_list.csv");

    std::vector<wob::ElementXs<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()>> elems;
    ANALYTICAL_SOLUTION_TYPE solution_type = domain == wob::InteriorDomain ? INTERIOR_SOLUTION2 : EXTERIOR_SOLUTION1;

    Eigen::Matrix<ScalarType, Eigen::Dynamic, 3> N;
    igl::per_face_normals(V, F, N);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0, 2);
    for (size_t f = 0; f < F.rows(); f++) {
        Eigen::Matrix<ScalarType, Dim, 1> v1 = V.row(F(f, 0));
        Eigen::Matrix<ScalarType, Dim, 1> v2 = V.row(F(f, 1));
        Eigen::Matrix<ScalarType, Dim, 1> v3 = V.row(F(f, 2));
        Eigen::Matrix<ScalarType, Dim, 1> n = N.row(f);

        wob::BoundaryType boundary_type;
        if (problem == wob::MixedProblem)
            boundary_type = (wob::BoundaryType)dist(mt);
        else
            boundary_type = (wob::BoundaryType)problem;

        if (boundary_type == wob::Dirichlet)
            elems.push_back(
                {v1, v2, v3, n, get_dirichlet_bc(solution_type, v1), get_dirichlet_bc(solution_type, v2),
                 get_dirichlet_bc(solution_type, v3), wob::Dirichlet}
            );
        else if (boundary_type == wob::Neumann)
            elems.push_back(
                {v1, v2, v3, n, get_neumann_bc(solution_type, v1, n), get_neumann_bc(solution_type, v2, n),
                 get_neumann_bc(solution_type, v3, n), wob::Neumann}
            );
        else if (boundary_type == wob::Robin)
            elems.push_back(
                {v1, v2, v3, n, get_dirichlet_bc(solution_type, v1) + get_neumann_bc(solution_type, v1, n),
                 get_dirichlet_bc(solution_type, v2) + get_neumann_bc(solution_type, v2, n),
                 get_dirichlet_bc(solution_type, v3) + get_neumann_bc(solution_type, v3, n), wob::Robin, 1.0f, 1.0f}
            );
    }

    std::vector<ScalarType> analytical_solution_buffer(num_evaluation_points);
    std::vector<Eigen::Matrix<ScalarType, Dim, 1>> anaylical_gradient_buffer(num_evaluation_points);
    thrust::host_vector<bool> evaluate_point_host(num_evaluation_points);
    for (size_t i = 0; i < num_evaluation_points; i++) {
        auto x_2d = wob::utils::idx_to_domain_point<ScalarType, 2>(i, grid_res, 1.0);
        Eigen::Matrix<ScalarType, Dim, 1> x(0.0, x_2d[1], x_2d[0]);
        auto analytical_solution_gradient = analytical_solution(solution_type, x);

        ScalarType winding_number = igl::winding_number(V, F, x);

        if (domain == wob::InteriorDomain ? winding_number > (ScalarType)0.5 : winding_number <= (ScalarType)0.5) {
            analytical_solution_buffer[i] = analytical_solution_gradient[0];
            anaylical_gradient_buffer[i] = analytical_solution_gradient.tail<3>();
            evaluate_point_host[i] = true;
        } else {
            analytical_solution_buffer[i] = std::numeric_limits<ScalarType>::quiet_NaN();
            anaylical_gradient_buffer[i].setZero();
            evaluate_point_host[i] = false;
        }
    }
    wob::utils::save_field(analytical_solution_buffer, output_dir + "/potential_analytical.scalar", grid_res);
    wob::utils::save_field(anaylical_gradient_buffer, output_dir + "/gradient_analytical.vector3", grid_res);

    wob::SceneHost<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> scene(elems);

    thrust::device_vector<ScalarType> solution(num_evaluation_points, wob::utils::zero<ScalarType>());
    thrust::device_vector<Eigen::Matrix<ScalarType, Dim, 1>> gradient(
        num_evaluation_points, wob::utils::zero<Eigen::Matrix<ScalarType, Dim, 1>>()
    );
    thrust::device_vector<bool> evaluate_point(evaluate_point_host);

    thrust::device_vector<wob::randomState_t> random_state_buffer(num_evaluation_points);
    wob::utils::random_states_init(random_state_buffer);

    DeviceArgs args = {
        path_length,
        num_sample_paths,
        num_resampling_candidates,
        grid_res,
        eps,
        first_kind_equation_scaling_constant,
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
    while (elapsed_time < max_time * 60.0) {
        num_iterations += 1;
        total_sample_count += num_sample_paths;
        auto begin_time = std::chrono::system_clock::now();

        wob::utils::run<DeviceFunctor>(args, num_evaluation_points, bie_model, domain, estimator, problem);

        auto end_time = std::chrono::system_clock::now();
        elapsed_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count() / 1000.0;

        thrust::for_each(
            thrust::device, thrust::make_counting_iterator<unsigned int>(0),
            thrust::make_counting_iterator<unsigned int>(num_evaluation_points),
            [args, _solution_ptr, _gradient_ptr, total_sample_count] __device__(unsigned int idx) {
                _solution_ptr[idx] = args.solution_buffer_ptr[idx] / total_sample_count;
                _gradient_ptr[idx] = args.gradient_buffer_ptr[idx] / total_sample_count;
            }
        );

        // For Nemann problems, the solution is not unique, so we shift the solution by the difference between
        // the analytical solution and the estimated solution at the center to evaluate errors.
        if (problem == wob::NeumannProblem)
            shift_solution(_solution, evaluate_point_host, grid_res, solution_type, domain);

        auto [solution_rmse, x_gradient_rmse, y_gradient_rmse, z_gradient_rmse] =
            compute_rmse(solution, gradient, evaluate_point_host, grid_res, solution_type, problem);

        std::cout << "\33[2K\r"; // clear line
        std::cout << "time " << elapsed_time << " s\t" << num_iterations << " iterations (" << total_sample_count
                  << " samples)\t" << std::flush;
        std::cout << "RMSE : " << solution_rmse << " " << x_gradient_rmse << " " << y_gradient_rmse << " "
                  << z_gradient_rmse << std::flush;
        rmse_out << total_sample_count << "," << elapsed_time << "," << solution_rmse << "," << x_gradient_rmse << ","
                 << y_gradient_rmse << "," << z_gradient_rmse << std::endl;

        wob::utils::save_field(
            _solution, output_dir + "/potential" + std::to_string(num_iterations) + ".scalar", grid_res
        );
        wob::utils::save_field(
            _gradient, output_dir + "/gradient" + std::to_string(num_iterations) + ".vector3", grid_res
        );
    }
    std::cout << std::endl;
}
