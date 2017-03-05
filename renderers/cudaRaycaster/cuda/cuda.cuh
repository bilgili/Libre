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

#ifndef _Cuda_h_
#define _Cuda_h_

#ifdef __CUDACC__
    #define CUDA_HOST_CALL __host__
    #define CUDA_DEVICE_CALL __device__
    #define CUDA_CALL __host__ __device__
    #define CUDA_CONSTANT __constant__
    #define CUDA_SHARED __shared__
#else
    #define CUDA_HOST_CALL
    #define CUDA_DEVICE_CALL
    #define CUDA_CALL
    #define CUDA_CONSTANT
    #define CUDA_SHARED
#endif

#endif // _Cuda_h_

