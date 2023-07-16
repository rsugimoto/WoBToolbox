/*
    Walk-on-Boundary Example.

    Author:         Ryusuke Sugimoto
    Affiliation:    University of Waterloo
    Date:           July 2023
    File Name:      simple_wob_and_wos.cu
    Description:    This is the simplest example where the solution of the Dirichlet problem is computed inside the
                    bunny shape. WoBFunc functor implements the main algorithm of the WoB method. The solution is saved
                    in the output directory as binary files. Users need to use a Python script to visualize the results.
                    Other example files are strcutured similarly, so I recommend you understand this example file first.
                    A similarly optimized WoS implementation is also provided for comparison.
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
    wob::Scene<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> scene;
    ScalarType *solution_buffer_ptr;
    bool *evaluate_point_buffer_ptr;
    wob::randomState_t *random_state_buffer_ptr;
};

inline __device__ ScalarType sign(ScalarType x) {
    if (x > 0)
        return 1.0;
    else
        return -1.0;
}

// Define closest point query for WoS
__device__ inline auto closest_point_query(
    const lbvh::bvh_device<
        ScalarType, Dim, wob::ElementXs<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()>> &bvh_dev,
    const Eigen::Matrix<ScalarType, 2, 1> &x
) {
    struct LineSqrD {
        inline __device__ ScalarType operator()(
            const float2 &p, const wob::ElementXs<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> &line
        ) const noexcept {
            Eigen::Matrix<ScalarType, Dim, 1> lVec = line.b - line.a; // Line Vector
            Eigen::Matrix<ScalarType, Dim, 1> _p(p.x, p.y);
            ScalarType t = (_p - line.a).dot(lVec) / lVec.squaredNorm();
            t = std::min(std::max(t, 0.0f), 1.0f);
            return ((line.a + lVec * t) - _p).squaredNorm();
        }
    };
    const auto [object_id, sqr_R] = lbvh::query_device(bvh_dev, lbvh::nearest(make_float2(x[0], x[1])), LineSqrD());
    const wob::ElementXs<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> line =
        bvh_dev.objects[object_id];
    const ScalarType R = sqrt(sqr_R);

    // Check intersection point, and interpolate the boundary value
    Eigen::Matrix<ScalarType, Dim, 1> lVec = line.b - line.a; // Line Vector
    Eigen::Matrix<ScalarType, Dim, 1> _p(x[0], x[1]);
    ScalarType t = (_p - line.a).dot(lVec) / lVec.squaredNorm();
    t = std::min(std::max(t, 0.0f), 1.0f);
    ScalarType boundary_value = (line.boundary_value_b - line.boundary_value_a) * t + line.boundary_value_a;

    return thrust::make_pair(boundary_value, R);
};

// optimized WoB implementation for interior Dirichlet problem.
inline __device__ void WoBFunc(const unsigned int idx, const DeviceArgs &args) {
    if (!args.evaluate_point_buffer_ptr[idx]) {
        args.solution_buffer_ptr[idx] = std::numeric_limits<ScalarType>::quiet_NaN();
        return;
    }
    wob::randomState_t random_state = args.random_state_buffer_ptr[idx];

    wob::utils::KahanSum<ScalarType> sum;
    wob::DomainPoint<ScalarType, Dim> x{wob::utils::idx_to_domain_point<ScalarType, Dim>(idx, args.grid_res, 1.0)};
    wob::BoundaryPoint<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> y;
    int m;
    for (unsigned int i = 0; i < args.num_evaluations_per_iter; i++) {
        thrust::tie(y, m) = args.scene.sample_boundary_line_intersection2(&random_state, x);
        ScalarType weight = m * sign((y.p - x.p).dot(y.n));
        ScalarType path_contribution = weight * y.boundary_value;
        for (unsigned int j = 1; j < args.path_length; j++) {
            if (j == args.path_length - 1) weight *= 0.5f;
            Eigen::Matrix<ScalarType, Dim, 1> y_prev = y.p;
            thrust::tie(y, m) = args.scene.sample_boundary_line_intersection2(&random_state, y);
            weight *= -m * sign((y.p - y_prev).dot(y.n));
            path_contribution += weight * y.boundary_value;
        }
        sum += path_contribution;
    }

    args.solution_buffer_ptr[idx] += sum.sum;
    args.random_state_buffer_ptr[idx] = random_state;
}

// similarly optimized WoS implementation for interior Dirichlet problem.
inline __device__ void WoSFunc(const unsigned int idx, const DeviceArgs &args) {
    if (!args.evaluate_point_buffer_ptr[idx]) {
        args.solution_buffer_ptr[idx] = std::numeric_limits<ScalarType>::quiet_NaN();
        return;
    }
    wob::randomState_t random_state = args.random_state_buffer_ptr[idx];

    Eigen::Matrix<ScalarType, Dim, 1> _x = wob::utils::idx_to_domain_point<ScalarType, Dim>(idx, args.grid_res, 1.0);

    wob::utils::KahanSum<ScalarType> sum;
    for (unsigned int i = 0; i < args.num_evaluations_per_iter; i++) {
        Eigen::Matrix<ScalarType, Dim, 1> x = _x;
        while (true) {
            const auto [boundary_value, curr_R] = closest_point_query(args.scene.bvh, x);
            if (curr_R < args.eps) {
                sum += boundary_value; // assumes constant element
                break;
            }

            ScalarType theta = (ScalarType)(2.f * M_PI) * wob::utils::rand_uniform<ScalarType>(&random_state);
            x += curr_R * Eigen::Matrix<ScalarType, Dim, 1>(cos(theta), sin(theta));
        }
    }

    args.solution_buffer_ptr[idx] += sum.sum;
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

    const std::string input_obj_file = config["input_obj_file"].get<std::string>();
    const std::string output_dir = config["output_dir"].get<std::string>();
    bool is_WoS = config["is_wos"].get<bool>();
    bool evaluate_at_all_points =
        config.contains("evaluate_at_all_points") ? config["evaluate_at_all_points"].get<bool>() : false;
    unsigned int path_length = is_WoS ? 1 : config["path_length"].get<unsigned int>();
    unsigned int num_sample_paths = config["num_sample_paths"].get<unsigned int>();
    unsigned int grid_res = config["grid_res"].get<unsigned int>();
    ScalarType eps =
        config.contains("eps") ? config["eps"].get<ScalarType>() : std::numeric_limits<ScalarType>::epsilon();
    unsigned int num_evaluation_points = grid_res * grid_res;
    ScalarType max_time = config["max_time"].get<ScalarType>();

    // lod the geometry.
    Eigen::Matrix<ScalarType, Eigen::Dynamic, 2> V;
    Eigen::Matrix<int, Eigen::Dynamic, 2> F;
    Eigen::Matrix<ScalarType, Eigen::Dynamic, 3> _V;
    Eigen::Matrix<int, Eigen::Dynamic, 3> _F;
    igl::readOBJ(input_obj_file, _V, _F);
    V = _V.leftCols<2>();
    F = _F.leftCols<2>();

    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "input json file\t: " << input_json_file << std::endl;
    std::cout << "input obj file\t: " << input_obj_file << std::endl;
    std::cout << " ( #line segments: " << F.rows() << " )" << std::endl;
    std::cout << "output directory\t: " << output_dir << std::endl;
    if (!is_WoS) std::cout << "path length\t: " << path_length << std::endl;
    std::cout << "#sample paths per iteration\t: " << num_sample_paths << std::endl;
    std::cout << "grid resolution\t: " << grid_res << std::endl;
    std::cout << "eps\t: " << eps << std::endl;
    std::cout << "precision\t: ";
    if constexpr (std::is_same<ScalarType, float>::value)
        std::cout << "float" << std::endl;
    else
        std::cout << "double" << std::endl;
    std::cout << "max time\t: " << max_time << " minutes" << std::endl;
    std::cout << "is WoS\t: " << is_WoS << std::endl;
    std::cout << "evaluate at all points\t: " << evaluate_at_all_points << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    std::filesystem::create_directories(output_dir);
    std::filesystem::copy(
        input_json_file, output_dir + "/config.json", std::filesystem::copy_options::overwrite_existing
    );

    // Define the boundary condition for each line segment and construct the scene.
    ScalarType total_boundary_length = 0.0;
    for (size_t f = 0; f < F.rows(); f++) {
        Eigen::Matrix<ScalarType, Dim, 1> v1 = V.row(F(f, 0)).head<2>();
        Eigen::Matrix<ScalarType, Dim, 1> v2 = V.row(F(f, 1)).head<2>();
        total_boundary_length += (v1 - v2).norm();
    }

    unsigned int num_segments = 40;
    ScalarType boundary_length_sum = 0.0;
    std::vector<wob::ElementXs<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()>> elems;

    for (size_t f = 0; f < F.rows(); f++) {
        Eigen::Matrix<ScalarType, Dim, 1> v1 = V.row(F(f, 0)).head<2>();
        Eigen::Matrix<ScalarType, Dim, 1> v2 = V.row(F(f, 1)).head<2>();
        Eigen::Matrix<ScalarType, Dim, 1> t = (v1 - v2).normalized();
        Eigen::Matrix<ScalarType, Dim, 1> n(t[1], -t[0]);

        int segment_id = boundary_length_sum * num_segments / total_boundary_length;
        ScalarType boundary_value = segment_id % 2 == 1 ? 1.0 : -1.0;
        elems.push_back({v1, v2, n, boundary_value, boundary_value, wob::Dirichlet});
        boundary_length_sum += (v1 - v2).norm();
    }
    wob::SceneHost<ScalarType, Dim, wob::is_vector_problem<wob::PoissonKernel>()> scene(elems);

    // Precompute the interior points on the grid.
    thrust::host_vector<bool> evaluate_point_host(num_evaluation_points);
    if (evaluate_at_all_points)
        thrust::fill(evaluate_point_host.begin(), evaluate_point_host.end(), true);
    else {
        for (size_t i = 0; i < num_evaluation_points; i++) {
            Eigen::Matrix<ScalarType, Dim, 1> x = wob::utils::idx_to_domain_point<ScalarType, Dim>(i, grid_res, 1.0);
            ScalarType winding_number = igl::winding_number(V, F, x);
            evaluate_point_host[i] = winding_number < -0.5;
        }
    }

    thrust::device_vector<ScalarType> solution(num_evaluation_points, wob::utils::zero<ScalarType>());
    thrust::device_vector<bool> evaluate_point(evaluate_point_host);

    thrust::device_vector<wob::randomState_t> random_state_buffer(num_evaluation_points);
    wob::utils::random_states_init(random_state_buffer);

    DeviceArgs args = {
        path_length,
        num_sample_paths,
        grid_res,
        eps,
        scene.get_device_repr(),
        solution.data().get(),
        evaluate_point.data().get(),
        random_state_buffer.data().get()};

    unsigned int num_iterations = 0;
    unsigned int total_sample_count = 0;
    double elapsed_time = 0.0;
    thrust::device_vector<ScalarType> _solution(num_evaluation_points, wob::utils::zero<ScalarType>());
    ScalarType *_solution_ptr = _solution.data().get();
    while (elapsed_time < max_time * 60.0) {
        num_iterations += 1;
        total_sample_count += num_sample_paths;
        auto begin_time = std::chrono::system_clock::now();

        if (is_WoS) {
            thrust::for_each(
                thrust::make_counting_iterator<unsigned int>(0),
                thrust::make_counting_iterator<unsigned int>(num_evaluation_points),
                [args] __device__(unsigned int idx) { WoSFunc(idx, args); }
            );
        } else {
            thrust::for_each(
                thrust::make_counting_iterator<unsigned int>(0),
                thrust::make_counting_iterator<unsigned int>(num_evaluation_points),
                [args] __device__(unsigned int idx) { WoBFunc(idx, args); }
            );
        }
        auto end_time = std::chrono::system_clock::now();
        elapsed_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count() / 1000.0;
        std::cout << "\33[2K\r"; // clear line
        std::cout << "time " << elapsed_time << " s\t" << num_iterations << " iterations (" << total_sample_count
                  << " samples)\t" << std::flush;

        thrust::for_each(
            thrust::device, thrust::make_counting_iterator<unsigned int>(0),
            thrust::make_counting_iterator<unsigned int>(num_evaluation_points),
            [args, _solution_ptr, total_sample_count] __device__(unsigned int idx) {
                _solution_ptr[idx] = args.solution_buffer_ptr[idx] / total_sample_count;
            }
        );
        wob::utils::save_field(
            _solution, output_dir + "/potential" + std::to_string(num_iterations) + ".scalar", grid_res
        );

        if (elapsed_time > max_time * 60.0) break;
    }
    std::cout << std::endl;
}
