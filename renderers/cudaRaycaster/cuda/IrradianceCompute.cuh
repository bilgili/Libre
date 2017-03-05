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

#ifndef _Cuda_IrradianceCompute_h
#define _Cuda_IrradianceCompute_h

#include "IrradianceTexture.cuh"
#include "common.cuh"
#include "ColorMap.cuh"
#include "ClipPlanes.cuh"
#include "types.cuh"

#include <livre/core/types.h>

namespace livre
{

class CudaLightSource;

namespace cuda
{

/** Holds the irradiance cache */
class IrradianceCompute
{
public:

    /**
     * Constructor
     * @param dataTypeSize size of the data type per voxel ( float, int, char etc )
     * @param isSigned true if the data is a signed data
     * @param isFloat true if the data is a floating poin type
     * @param nComponents is the number of components
     * @param overlap is the size of theoverlap in each block
     * @param coarseLevelSize is size of the volume in lowest resolution
     */
    IrradianceCompute( size_t dataTypeSize,
                       bool isSigned,
                       bool isFloat,
                       size_t nComponents,
                       const Vector3ui& overlap,
                       const Vector3ui& coarseLevelSize );
    ~IrradianceCompute();

    /**
     * Copies data in ptr to the 3d cuda array ( thread safe for slot retrieval )
     * @param ptr the data to be copied
     * @param pos is the position of the block in irradiance texture
     * @param size the size of the data in bytes
     */
    void upload( const unsigned char* ptr,
                 const Vector3ui& pos,
                 const Vector3ui& size );

    void update( const RenderData& renderData,
                 const ViewData& viewData,
                 const lexis::render::ColorMap& colorMap,
                 const ::livre::ConstLightSources& lightSources );

    IrradianceTexture compute();

private:

    IrradianceTexture _texture;
    ClipPlanes _clipPlanes;
    ColorMap _colorMap;
    RenderData _renderData;
    ViewData _viewData;
    LightSources _lightSources;
    const LightSource** _cudaLightSources;
};

}
}
#endif // _Cuda_IrradianceCompute_h
