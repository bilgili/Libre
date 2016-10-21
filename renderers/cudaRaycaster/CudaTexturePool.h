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

class CudaTexturePool
{

public:
    CudaTexturePool( const DataSource& dataSource, const size_t textureMemory );
    ~CudaTexturePool();

    Vector3f copyToSlot( const unsigned char* ptr, const Vector3ui& size );
    void releaseSlot( const Vector3f& pos );
    size_t getSlotMemSize() const;
    Vector3ui getTextureSize() const;
    size_t getTextureMem() const;
    cuda::TexturePool& getCudaTexturePool() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}


#endif // _CudaTexturePool_h_

