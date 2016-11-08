/* Copyright (c) 2011-2015, EPFL/Blue Brain Project
 *                     Ahmet Bilgili <ahmet.bilgili@epfl.ch>
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

#include "CudaTextureUploadFilter.h"
#include "CudaTextureObject.h"
#include "cuda_runtime.h"

#include <livre/core/pipeline/Pipeline.h>
#include <livre/core/data/NodeId.h>
#include <livre/core/cache/Cache.h>

namespace livre
{

struct CudaTextureUploadFilter::Impl
{
public:

    Impl( const DataCache& dataCache,
          CudaTextureCache& cudaCache,
          CudaTexturePool& texturePool )
        : _dataCache( dataCache )
        , _cudaCache( cudaCache )
        , _texturePool( texturePool )
    {}

    void execute( const FutureMap& input, PromiseMap& output ) const
    {
        ConstCacheObjects cacheObjects;
        const RenderInputs renderInputs = input.get< RenderInputs >( "RenderInputs" )[ 0 ];
        for( const auto& dataCacheObjects: input.getFutures( "DataCacheObjects" ))
            for( const auto& dataCacheObject: dataCacheObjects.get< ConstCacheObjects >( ))
            {
                const auto& cacheObj = _cudaCache.load( dataCacheObject->getId(),
                                                        _dataCache,
                                                        renderInputs.dataSource,
                                                        _texturePool );
                if( cacheObj )
                    cacheObjects.push_back( cacheObj );


            }
        output.set( "CudaTextureCacheObjects", cacheObjects );
    }

    const DataCache& _dataCache;
    CudaTextureCache& _cudaCache;
    CudaTexturePool& _texturePool;
};

CudaTextureUploadFilter::CudaTextureUploadFilter( const DataCache& dataCache,
                                                  CudaTextureCache& cudaCache,
                                                  CudaTexturePool& texturePool )
    : _impl( new CudaTextureUploadFilter::Impl( dataCache, cudaCache, texturePool ))
{
}

CudaTextureUploadFilter::~CudaTextureUploadFilter()
{}

void CudaTextureUploadFilter::execute( const FutureMap& input, PromiseMap& output ) const
{
    _impl->execute( input, output );
}
}

