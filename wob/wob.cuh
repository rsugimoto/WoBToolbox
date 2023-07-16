/*
    Walk-on-Boundary Toolbox
    This file is a part of the public release of the Walk-on-Boundary (WoB) Toolbox.

    Author:         Ryusuke Sugimoto
    Affiliation:    University of Waterloo
    Date:           July 2023
    File Name:      wob.cuh
    Description:    Main header file for the WoB Toolbox. Include this file to use the toolbox.
*/

#ifndef __WOB_CUH__
#define __WOB_CUH__

#ifndef __CUDACC__
#define __device__
#define __host__
#endif

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include "wob/bie_models.cuh"
#include "wob/equation_kernel.cuh"
#include "wob/estimator.cuh"
#include "wob/points.hpp"
#include "wob/runtime_utils.cuh"
#include "wob/scene.cuh"
#include "wob/utils.cuh"

#endif // __WOB_CUH__