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

#ifndef _Cuda_Allocate_h
#define _Cuda_Allocate_h

#include "cuda.h"
#include "debug.cuh"

#include <cuda_runtime_api.h>
#include <utility>

namespace livre
{
namespace cuda
{

template< class T, class... Args >
__global__ void allocate( T* ptr, Args... args )
{
    new( ptr ) T( args... );
}

template< class T >
__global__ void deallocate( T* ptr )
{
   ptr->~T();
}

template< class T, class... Args >
CUDA_HOST_CALL T* allocateCudaObjectInDevice( Args&&... args )
{
    T* ptr;
    checkCudaErrors( cudaMalloc( &ptr, sizeof( T )));
    allocate< T ><<<1,1>>>( ptr, std::forward< Args >( args )... );
    return ptr;
}

template< class T >
CUDA_HOST_CALL void deallocateCudaObjectInDevice( T* ptr )
{
    deallocate< T ><<<1,1>>>( ptr );
    cudaFree( ptr );
}
}
}
#endif
