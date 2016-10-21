#ifndef _Cuda_h_
#define _Cuda_h_

#ifdef __CUDACC__
    #define CUDA_CALL __host__ __device__
    #define CUDA_CONSTANT __constant__

/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>
#define checkCudaErrors(err)  __checkCudaErrors( err, __FILE__, __LINE__ )

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors( cudaError_t err, const char *file, const int line )
{
    if( err == cudaSuccess )
        return;

    std::stringstream stream;
    stream << "Cuda operation failed: " << cudaGetErrorString( err ) << " "
           << file << " "
           << line;
    throw std::runtime_error( stream.str( ));

}

#else
    #define CUDA_CALL
    #define CUDA_CONSTANT
    #define checkCudaErrors(err)
#endif




#endif // _Cuda_h_

