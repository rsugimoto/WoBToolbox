/*
    Walk-on-Boundary Toolbox
    This file is a part of the public release of the Walk-on-Boundary (WoB) Toolbox.

    Author:         Ryusuke Sugimoto
    Affiliation:    University of Waterloo
    Date:           July 2023
    File Name:      points.cuh
    Description:    This file defines sample points for the boundary and the domain.
                    - BoundaryPoint: a point on the boundary
                    - DomainPoint: a point in the domain
                    - DomainSamplePoint: a point in the domain with a value
*/

#ifndef __WOB_POINTS_HPP__
#define __WOB_POINTS_HPP__

#include <Eigen/Core>

namespace wob {

enum BoundaryType { Dirichlet, Neumann, Robin };

template <typename ScalarType, unsigned int Dim, bool IsVectorProblemm> struct value_type;
template <typename ScalarType, unsigned int Dim> struct value_type<ScalarType, Dim, true> {
    using type = Eigen::Matrix<ScalarType, Dim, 1>;
};
template <typename ScalarType, unsigned int Dim> struct value_type<ScalarType, Dim, false> {
    using type = ScalarType;
};

template <typename ScalarType, unsigned int Dim, bool IsVectorProblem> struct BoundaryPoint {
    Eigen::Matrix<ScalarType, Dim, 1> p;
    Eigen::Matrix<ScalarType, Dim, 1> n;
    ScalarType interior_angle; // out of 1
    typename value_type<ScalarType, Dim, IsVectorProblem>::type boundary_value;
    BoundaryType boundary_type;
    ScalarType robin_alpha;
};

template <typename ScalarType, unsigned int Dim> struct DomainPoint {
    Eigen::Matrix<ScalarType, Dim, 1> p;
};

template <typename ScalarType, unsigned int Dim, typename ValueType> struct DomainSamplePoint {
    Eigen::Matrix<ScalarType, Dim, 1> p;
    ValueType value;
};

} // namespace wob

#endif //__WOB_POINTS_HPP__