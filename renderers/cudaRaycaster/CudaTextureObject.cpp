/* Copyright (c) 2011-2016, EPFL/Blue Brain Project
 *                          Ahmet Bilgili <ahmet.bilgili@epfl.ch>
 *
 * This file is part of Livre <https://github.com/BlueBrain/Livre>
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

#include "CudaTextureObject.h"
#include "CudaTexturePool.h"

#include <livre/lib/cache/DataObject.h>

#include <livre/core/cache/Cache.h>
#include <livre/core/data/LODNode.h>
#include <livre/core/data/DataSource.h>
#include <livre/core/render/TexturePool.h>

#include <lunchbox/debug.h>

#include <cuda_runtime.h>

namespace livre
{
namespace
{
const Vector3f INVALID_SLOT_POSITION( -1.0f );
}

/**
 * The TextureObject class holds the informarmation for the data which is on the GPU.
 */
struct CudaTextureObject::Impl
{

    Impl( const CacheId& cacheId,
          const DataCache& dataCache,
          const DataSource& dataSource,
          CudaTexturePool& texturePool )
        : _size( 0 )
        , _texturePool( texturePool )
    {
        if( !load( cacheId, dataCache, dataSource ))
            LBTHROW( CacheLoadException( cacheId, "Unable to construct texture cache object" ));
    }

    ~Impl()
    {
        if( _slotPosition != INVALID_SLOT_POSITION )
            _texturePool.releaseSlot( _slotPosition );
    }

    bool load( const CacheId& cacheId, const DataCache& dataCache, const DataSource& dataSource )
    {
        ConstDataObjectPtr data = dataCache.get( cacheId );
        if( !data )
            return false;

        const VolumeInformation& volInfo = dataSource.getVolumeInfo();
        _size = _texturePool.getSlotMemSize();

        const LODNode& lodNode = dataSource.getNode( NodeId( cacheId ));
        _slotPosition = _texturePool.copyToSlot( (const uint8_t *)data->getDataPtr(),
                                                 lodNode.getBlockSize() + ( volInfo.overlap * 2 ));

        if( _slotPosition == INVALID_SLOT_POSITION )
            return false;

        const Vector3f cacheTextureSize = _texturePool.getTextureSize();
        const Vector3f overlap = volInfo.overlap;
        const Vector3f size = lodNode.getVoxelBox().getSize();
        const Vector3f overlapf = overlap / cacheTextureSize;
        _texturePos = _slotPosition + overlapf;
        _textureSize = size / cacheTextureSize;
        return true;
    }

    size_t _size;
    CudaTexturePool& _texturePool;
    Vector3f _slotPosition;
    Vector3f _texturePos;
    Vector3f _textureSize;
};

CudaTextureObject::CudaTextureObject( const CacheId& cacheId,
                                      const DataCache& dataCache,
                                      const DataSource& dataSource,
                                      CudaTexturePool& texturePool )
   : CacheObject( cacheId )
   , _impl( new Impl( cacheId, dataCache, dataSource, texturePool ))
{}

CudaTextureObject::~CudaTextureObject()
{}

Vector3f CudaTextureObject::getTexPosition() const
{
    return _impl->_texturePos;
}

Vector3f CudaTextureObject::getTexSize() const
{
    return _impl->_textureSize;

}

CudaTexturePool& CudaTextureObject::getTexturePool() const
{
    return _impl->_texturePool;
}

size_t CudaTextureObject::getSize() const
{
    return _impl->_size;
}

}
