/* Copyright (c) 2011-2016  Ahmet Bilgili <ahmetbilgili@gmail.com>
 *
 * This file is part of Livre <https://github.com/bilgili/Livre>
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

namespace livre
{
namespace cuda
{
ColorMap::ColorMap()
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc< float4 >();
    cudaMallocArray( &_array, &channelDesc, 256, 1);

    upload( lexis::render::ColorMap::getDefaultColorMap( 0.0f, 256.0f ));
    _texture.filterMode = cudaFilterModeLinear;
    _texture.normalized = true;
    _texture.addressMode[0] = cudaAddressModeClamp;
    cudaBindTextureToArray( _texture, _array, channelDesc );
}

ColorMap::~ColorMap()
{
    cudaFreeArray( _array );
}

void ColorMap::upload( const lexis::render::ColorMap& colorMap )
{
    const auto& colors =
            colorMap.sampleColors< float >( 256, 0.0f, 256.0f, 0 );

    cudaMemcpyToArray( _array, 0, 0,
                       colors.data(),
                       colors.size() * sizeof(float4),
                       cudaMemcpyHostToDevice );
}
}
}
