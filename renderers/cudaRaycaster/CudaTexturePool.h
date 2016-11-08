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

#ifndef _CudaTexturePool_h_
#define _CudaTexturePool_h_

#include "types.h"
#include <livre/core/types.h>
#include <livre/core/mathTypes.h>

namespace livre
{
namespace cuda
{
class TexturePool;
}

/** Manages the texture pool allocation, data copies and deallocations */
 class CudaTexturePool
{

public:

     /**
     * Constructor
     * @param dataSource the data source
     * @param textureMemory the maximum texture memory in bytes
     */
    CudaTexturePool( const DataSource& dataSource, const size_t textureMemory );
    ~CudaTexturePool();

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
    size_t getSlotMemSize() const;

    /** @return the size of the cuda array in voxels */
    Vector3ui getTextureSize() const;

    /** @return the memory size of the cuda array in bytes */
    size_t getTextureMem() const;

    /**
     * @return the cuda representation of the texture pool.
     * @note Should not be used directly
     */
    cuda::TexturePool& _getCudaTexturePool() const;

private:

    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}


#endif // _CudaTexturePool_h_

