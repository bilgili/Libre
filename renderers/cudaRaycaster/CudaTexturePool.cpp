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

#include "CudaTexturePool.h"
#include "CudaTextureObject.h"
#include "cuda/TexturePool.cuh"

#include <livre/core/data/DataSource.h>

#include <lunchbox/debug.h>

namespace livre
{
struct CudaTexturePool::Impl
{
    Impl( const DataSource& dataSource, const size_t maxGpuMemory )
        : _volInfo( dataSource.getVolumeInfo( ))
    {
        bool isSigned = false;
        bool isFloat = false;

        switch( dataSource.getVolumeInfo().dataType )
        {
        case DT_UINT8:
        case DT_UINT16:
        case DT_UINT32:
        break;
        case DT_INT8:
        case DT_INT16:
        case DT_INT32:
            isSigned = true;
        break;
        case DT_FLOAT:
            isFloat = true;
        break;
        case DT_UNDEFINED:
        default:
           LBTHROW( std::runtime_error( "Undefined data type" ));
        break;
        }

        _texturePool.reset( new::livre::cuda::TexturePool( _volInfo.getBytesPerVoxel(),
                                                           isSigned,
                                                           isFloat,
                                                           _volInfo.compCount,
                                                           _volInfo.maximumBlockSize,
                                                           maxGpuMemory ));
    }

    Vector3f copyToSlot( const unsigned char* ptr, const Vector3ui& size )
    {
        return _texturePool->copyToSlot( ptr, size );
    }

    void releaseSlot( const Vector3f& slot  )
    {
        _texturePool->releaseSlot( slot );
    }

    size_t getSlotMemSize() const
    {
        return _texturePool->getSlotMemSize();
    }

    const VolumeInformation& _volInfo;
    std::unique_ptr< ::livre::cuda::TexturePool > _texturePool;
};

CudaTexturePool::CudaTexturePool( const DataSource& dataSource, const size_t textureMemory )
    : _impl( new CudaTexturePool::Impl( dataSource, textureMemory ))
{}

CudaTexturePool::~CudaTexturePool()
{}

Vector3f CudaTexturePool::copyToSlot( const unsigned char* ptr, const Vector3ui& size )
{
    return _impl->copyToSlot( ptr, size );
}

cuda::TexturePool& CudaTexturePool::_getCudaTexturePool() const
{
    return *_impl->_texturePool;
}

void CudaTexturePool::releaseSlot( const Vector3f& slot  )
{
    _impl->releaseSlot( slot );
}

size_t CudaTexturePool::getSlotMemSize() const
{
    return _impl->getSlotMemSize();
}

Vector3ui CudaTexturePool::getTextureSize() const
{
    return _impl->_texturePool->getTextureSize();
}

size_t CudaTexturePool::getTextureMem() const
{
    return _impl->_texturePool->getTextureMem();
}

}

