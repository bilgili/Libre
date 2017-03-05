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

#include "ColorMap.cuh"
#include "debug.cuh"

#include <cuda_runtime.h>
#include <cuda_texture_types.h>

namespace livre
{
namespace cuda
{
ColorMap::ColorMap()
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc< float4 >();
    checkCudaErrors( cudaMallocArray( &_array, &channelDesc, 256, 1));

    upload( lexis::render::ColorMap::getDefaultColorMap( 0.0f, 256.0f ));

    // create texture object
    cudaResourceDesc resDesc;
    ::memset( &resDesc, 0, sizeof( cudaResourceDesc ));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = _array;

    cudaTextureDesc texDesc;
    ::memset( &texDesc, 0, sizeof( cudaTextureDesc ));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.addressMode[ 0 ] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.normalizedCoords = 1;

    // create texture object: we only have to do this once!
    checkCudaErrors( cudaCreateTextureObject( &_texture, &resDesc, &texDesc, NULL ));
}

ColorMap::~ColorMap()
{}

/** Deletes the cuda objects */
void ColorMap::clear()
{
    cudaFreeArray( _array );
}

void ColorMap::upload( const lexis::render::ColorMap& colorMap )
{
    const auto& colors =
            colorMap.sampleColors< float >( 256, 0.0f, 256.0f, 0 );
    checkCudaErrors( cudaMemcpyToArray( _array, 0, 0,
                                        colors.data(),
                                        colors.size() * sizeof(float4),
                                        cudaMemcpyHostToDevice ));
}
}
}
