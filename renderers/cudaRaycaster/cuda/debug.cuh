/* Copyright (c) 2011-2016  Ahmet Bilgili <ahmetbilgili@gmail.com>
 *
 * This file is part of Livre <https://github.com/bilgili/Libre>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _cuda_debug_cuh
#define _cuda_debug_cuh

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


#endif // _cuda_debug_cuh

