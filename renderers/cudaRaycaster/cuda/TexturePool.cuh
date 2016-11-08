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

#ifndef _Cuda_TexturePool_h
#define _Cuda_TexturePool_h

#include <livre/core/mathTypes.h>
#include <livre/core/data/DataSource.h>

#include <cuda_runtime.h>
#include <cuda_texture_types.h>

#include <boost/thread/mutex.hpp>

#include <vector>
#include <algorithm>

namespace livre
{
namespace cuda
{

/**
 * Cuda representation of the texture pool. It creates a large 3D cuda array depending
 * on the size of size the GPU memory and the maximum GPU memory.
 */
class TexturePool
{
public:

    /**
     * Constructor
     * @param dataTypeSize size of the data type per voxel ( float, int, char etc )
     * @param isSigned true if the data is a signed data
     * @param isFloat true if the data is a floating poin type
     * @param nComponents is the number of components
     * @param maxBlockSize is the size of the maximum block
     * @param maxGpuMemory is the size of the max gpu cache memory
     */
    TexturePool( size_t dataTypeSize,
                 bool isSigned,
                 bool isFloat,
                 size_t nComponents,
                 const Vector3ui& maxBlockSize,
                 size_t maxGpuMemory );
    ~TexturePool();

    /**
     * Copies data in ptr to the 3d cuda array ( thread safe for slot retrieval )
     * @param ptr the data to be copied
     * @param size the size of the data in bytes
     * @return the normalized position of the 3d array. If no place is left in the
     * array returns Vector3f( -1.0f )
     */
    Vector3f copyToSlot( const unsigned char* ptr, const Vector3ui& size );

    /**
     * Releases the slot for further data
     * @param pos of the slot to be released
     */
    void releaseSlot( const Vector3f& pos );

    /** @return the cuda allocation size of a slot in bytes */
    size_t getSlotMemSize() const { return _cudaBlockSize; }

    /** @return the size of the cuda array in voxels */
    Vector3ui getTextureSize() const { return _cacheSlotsSize * _maxBlockSize; }

    /** @return the memory size of the cuda array in bytes */
    size_t getTextureMem() const { return ( _cacheSlotsSize ).product() * _cudaBlockSize; }

    /** @return the cuda array */
    cudaTextureObject_t getTexture() const { return _texture; }

private:

    Vector3ui _cacheSlotsSize;
    std::vector< Vector3f > _emptySlotsList;
    const uint32_t _dataTypeSize;
    const bool _isSigned;
    const bool _isFloat;
    const uint32_t _nComponents;
    const Vector3ui _maxBlockSize;
    boost::mutex _mutex;
    const size_t _cudaBlockSize;
    cudaArray_t _cudaArray;
    cudaTextureObject_t _texture;
};
}
}
#endif // _Cuda_TexturePool_h
