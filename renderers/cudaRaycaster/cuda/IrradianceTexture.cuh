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

#ifndef _Cuda_IrradianceTexture_h
#define _Cuda_IrradianceTexture_h

#include "cuda.cuh"

#include <livre/core/mathTypes.h>
#include <livre/core/data/DataSource.h>

#include <cuda_runtime.h>
#include <cuda_texture_types.h>

namespace livre
{
namespace cuda
{

/** Holds the irradiance cache */
class IrradianceTexture
{
public:

    /** Empty constructor */
    IrradianceTexture() {}

    /**
     * Constructor
     * @param dataTypeSize size of the data type per voxel ( float, int, char etc )
     * @param isSigned true if the data is a signed data
     * @param isFloat true if the data is a floating poin type
     * @param nComponents is the number of components
     * @param overlap is the size of theoverlap in each block
     * @param coarseLevelSize is size of the volume in lowest resolution
     */
    IrradianceTexture( size_t dataTypeSize,
                       bool isSigned,
                       bool isFloat,
                       size_t nComponents,
                       const Vector3ui& overlap,
                       const Vector3ui& coarseLevelSize );
    ~IrradianceTexture();

    /** Deletes the cuda objects */
    void clear();

    /**
     * Copies data in ptr to the 3d cuda array ( thread safe for slot retrieval )
     * @param ptr the data to be copied
     * @param pos is the position of the block in irradiance texture
     * @param size the size of the data in bytes
     */
    void upload( const unsigned char* ptr,
                 const Vector3ui& pos,
                 const Vector3ui& size );

    /** @return the volume size */
    CUDA_CALL Vector3ui getVolumeSize() const { return _volumeSize; }

    /** @return the volume size */
    CUDA_CALL Vector3ui getIrradianceVolumeSize() const { return _irradianceVolumeSize; }

    /** @return the volume data */
    CUDA_CALL const cudaSurfaceObject_t* getIrradianceBuffer() const { return _irradianceSurf; }

    /** @return the volume data */
    CUDA_CALL cudaSurfaceObject_t* getIrradianceBuffer() { return _irradianceSurf; }

    /** @return the source buffer */
    CUDA_CALL const cudaTextureObject_t* getIrradianceTexture() const { return _irradianceTexture; }

    /** @return the destination buffer */
    CUDA_CALL cudaTextureObject_t getVolumeTexture() const { return _volumeTexture; }

private:

    cudaArray_t _irradianceArray[ 3 ]; // Quarter size volume, uint8
    cudaArray_t _volumeArray; // Full volume size, uint8
    cudaSurfaceObject_t _irradianceSurf[ 3 ];
    cudaTextureObject_t _irradianceTexture[ 3 ];
    cudaTextureObject_t _volumeTexture;
    uint32_t _dataTypeSize;
    bool _isSigned;
    bool _isFloat;
    uint32_t _nComponents;
    Vector3ui _overlap;
    Vector3ui _volumeSize;
    Vector3ui _irradianceVolumeSize;
};
}
}
#endif // _Cuda_IrradianceTexture_h
