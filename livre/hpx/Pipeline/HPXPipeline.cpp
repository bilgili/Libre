/* Copyright (c) 2011-2014, EPFL/Blue Brain Project
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

#include <livre/core/Data/NodeId.h>

#include <livre/Lib/Cache/TextureDataCache.h>
#include <livre/Lib/Cache/TextureCache.h>

#include <livre/Lib/Cache/TextureDataObject.h>
#include <livre/Lib/Cache/TextureObject.h>

#include <livre/hpx/Pipeline/HPXPipeline.h>
#include <hpx/include/thread_executors.hpp>

#include <hpx/hpx.hpp>
#include <hpx/include/lcos.hpp>

namespace livre
{

typedef hpx::lcos::queue< NodeId > NodeIdQueue;
typedef hpx::lcos::queue< ConstTextureDataObjectPtr > TextureDataObjectQueue;

class HPXPipeline::HPXPipelineImpl :
        hpx::components::simple_component_base< HPXPipeline::HPXPipelineImpl >
{
    ConstTextureDataObjectPtr uploadTextureData( const NodeId& nodeId )
    {
        ConstTextureDataObjectPtr cacheObject =
                _dataCache.getNodeTextureData( nodeId.getId( ));
        cacheObject->cacheLoad( );
        return cacheObject;
    }

    HPX_DEFINE_COMPONENT_ACTION( HPXPipeline::HPXPipelineImpl,
                                 uploadTextureData,
                                 UploadTextureData );

    ConstTextureObjectPtr uploadTexture( ConstTextureDataObjectPtr textureDataObject )
    {
        ConstTextureObjectPtr cacheObject =
                _textureCache.getNodeTexture( textureDataObject->getCacheID( ));
        cacheObject->setTextureDataObject( textureDataObject );
        cacheObject->cacheLoad();
        return cacheObject;
    }

    HPX_DEFINE_COMPONENT_ACTION( HPXPipeline::HPXPipelineImpl,
                                 uploadTexture,
                                 UploadTexture );

    HPXPipelineImpl( TextureDataCache& dataCache,
                      TextureCache& textureCache,
                      const size_t nDataUploaders,
                      const size_t nTextureUploaders,
                      GLContextPtr glContext )
        : _dataCache( dataCache ),
          _textureCache( textureCache ),
          _dataUploaderPool( nDataUploaders ),
          _textureUploaderPool( nTextureUploaders )
    {}

     void getCacheObjects( const NodeIds& nodeIds,
                           ConstCacheObjects& cacheObjects,
                           bool wait = /* false */)
     {
        cacheObjects.reserve( nodeIds.size( ));

        std::vector< hpx::future< ConstTextureObjectPtr >> resultFutures;
        resultFutures.reserve( nodeIds.size( ));

        UploadTextureData uploadTextureData;
        UploadTexture uploadTexture;
        BOOST_FOREACH( const NodeId& nodeId, nodeIds )
        {
            ConstTextureObjectPtr cacheObject =
                    _textureCache.getNodeTexture( textureDataObject->getCacheID( ));

            if( cacheObject->isLoaded())
                cacheObjects.push_back( cacheObject );

            hpx::future< ConstTextureDataObjectPtr > textureDataFuture =
                hpx::async( _dataUploaderPool, uploadTextureData, nodeId );

            resultFutures.push_back(
                        textureDataFuture.then( _textureUploaderPool,
                                                uploadTexture ));
        }

        if( wait )
            hpx::wait_all();

        BOOST_FOREACH( hpx::future< ConstTextureObjectPtr >& future, resultFutures )
        {
            if( future.has_value())
                cacheObjects.push_back( future.get( ));
        }
     }


    TextureDataObjectPtr _uploadRawData( const NodeId& nodeId );
    TextureObjectPtr _uploadTexture( const TextureDataObjectPtr cacheObject );

    TextureDataCache& _dataCache;
    TextureCache& _textureCache;
    GLContextPtr _glContext;

    hpx::threads::executors::local_priority_queue_executor _dataUploaderPool;
    hpx::threads::executors::local_priority_queue_executor _textureUploaderPool;
};

void HPXPipeline::getCacheObjects( const NodeIds& nodeIds,
                                   ConstCacheObjects& cacheObjects,
                                   bool wait = /* false */)
{
     _impl->getCacheObjects( cacheObjects,
                             nodeIds,
                             wait );
}

HPXPipeline::HPXPipeline( TextureDataCache& dataCache,
                          TextureCache& textureCache,
                          const size_t nDataUploaders,
                          const size_t nTextureUploaders,
                          GLContextPtr glContext )
    : _impl( new HPXPipelineImpl( dataCache,
                                  textureCache,
                                  nDataUploaders,
                                  nTextureUploaders,
                                  glContext ))
{

}





}
