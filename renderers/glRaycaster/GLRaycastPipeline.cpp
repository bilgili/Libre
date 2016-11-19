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

#include "GLRaycastPipeline.h"
#include "GLRenderUploadFilter.h"

#include <livre/lib/pipeline/RenderingSetGeneratorFilter.h>
#include <livre/lib/pipeline/VisibleSetGeneratorFilter.h>
#include <livre/lib/pipeline/DataUploadFilter.h>
#include <livre/lib/pipeline/RenderFilter.h>
#include <livre/lib/pipeline/HistogramFilter.h>
#include <livre/lib/cache/TextureObject.h>

#include <livre/core/cache/Cache.h>
#include <livre/core/pipeline/SimpleExecutor.h>
#include <livre/core/pipeline/Pipeline.h>
#include <livre/core/data/DataSource.h>
#include <livre/core/render/RenderInputs.h>
#include <livre/core/render/TexturePool.h>
#include <livre/core/render/Renderer.h>
#include <livre/core/render/GLContext.h>
#include <livre/core/settings/RenderSettings.h>

#include <livre/core/util/PluginRegisterer.h>
#include <lunchbox/debug.h>

#include <boost/progress.hpp>
#include <boost/thread/tss.hpp>

#include <livre/core/version.h>

extern "C"
int LunchboxPluginGetVersion() { return LIVRECORE_VERSION_ABI; }

extern "C"
bool LunchboxPluginRegister() { return true; }

namespace livre
{
namespace
{
const size_t nRenderThreads = 1;
const size_t nUploadThreads = 4;
const size_t nComputeThreads = 2;
const size_t nAsyncUploadThreads = 1;
PluginRegisterer< GLRaycastPipeline, const std::string& > registerer;

boost::thread_specific_ptr< TextureCache > textureCache;
boost::thread_specific_ptr< TexturePool > texturePool;
std::unique_ptr< DataCache > dataCache;
std::unique_ptr< HistogramCache > histogramCache;
}

struct GLRaycastPipeline::Impl
{
    Impl()
        : _renderExecutor( nRenderThreads, "Render Executor", GLContext::getCurrent()->clone( ))
        , _computeExecutor( nComputeThreads, "Compute Executor", GLContext::getCurrent()->clone( ))
        , _uploadExecutor( nUploadThreads, "Upload Executor", GLContext::getCurrent()->clone( ))
        , _asyncUploadExecutor( nAsyncUploadThreads, "Async Upload Executor", GLContext::getCurrent()->clone( ))
    {
    }

    void setupVisibleGeneratorFilter( PipeFilter& visibleSetGenerator,
                                      const RenderInputs& renderInputs ) const
    {
        visibleSetGenerator.getPromise( "Frustum" ).set( renderInputs.frameInfo.frustum );
        visibleSetGenerator.getPromise( "Frame" ).set( renderInputs.frameInfo.timeStep );
        visibleSetGenerator.getPromise( "DataRange" ).set( renderInputs.renderDataRange );
        visibleSetGenerator.getPromise( "Params" ).set( renderInputs.vrParameters );
        visibleSetGenerator.getPromise( "Viewport" ).set( renderInputs.pixelViewPort );
        visibleSetGenerator.getPromise( "ClipPlanes" ).set(
                    renderInputs.renderSettings.getClipPlanes( ));
    }

    // Sort helper function for sorting the textures with their distances to viewpoint
    struct DistanceOperator
    {
        explicit DistanceOperator( const DataSource& dataSource, const Frustum& frustum )
            : _frustum( frustum )
            , _dataSource( dataSource )
        { }

        bool operator()( const NodeId& rb1, const NodeId& rb2 )
        {
            const LODNode& lodNode1 = _dataSource.getNode( rb1 );
            const LODNode& lodNode2 = _dataSource.getNode( rb2 );

            const float distance1 = ( _frustum.getMVMatrix() *
                                      lodNode1.getWorldBox().getCenter() ).length();
            const float distance2 = ( _frustum.getMVMatrix() *
                                      lodNode2.getWorldBox().getCenter() ).length();
            return  distance1 < distance2;
        }
        const Frustum& _frustum;
        const DataSource& _dataSource;
    };

    void renderSync( RenderStatistics& statistics,
                     Renderer& renderer,
                     const RenderInputs& renderInputs )
    {

        PipeFilter sendHistogramFilter = renderInputs.filters.find( "SendHistogramFilter" )->second;
        PipeFilter preRenderFilter = renderInputs.filters.find( "PreRenderFilter" )->second;
        PipeFilterT< VisibleSetGeneratorFilter > visibleSetGenerator( "VisibleSetGenerator",
                                                                      renderInputs.dataSource );
        setupVisibleGeneratorFilter( visibleSetGenerator, renderInputs );
        visibleSetGenerator.connect( "VisibleNodes", preRenderFilter, "VisibleNodes" );
        preRenderFilter.getPromise( "Frustum" ).set( renderInputs.frameInfo.frustum );
        visibleSetGenerator.execute();
        preRenderFilter.execute();

        const livre::UniqueFutureMap portFutures( visibleSetGenerator.getPostconditions( ));
        NodeIds nodeIdsCopy = portFutures.get< NodeIds >( "VisibleNodes" );
        DistanceOperator distanceOp( renderInputs.dataSource, renderInputs.frameInfo.frustum );
        std::sort( nodeIdsCopy.begin(), nodeIdsCopy.end(), distanceOp );

        const VolumeInformation& volInfo = renderInputs.dataSource.getVolumeInfo();
        const size_t blockMemSize = volInfo.maximumBlockSize.product() *
                                    volInfo.getBytesPerVoxel() *
                                    volInfo.compCount;

        const uint32_t maxNodesPerPass =
                renderInputs.vrParameters.getMaxGPUCacheMemoryMB() * LB_1MB / blockMemSize;

        const uint32_t numberOfPasses = std::ceil( (float)nodeIdsCopy.size() / (float)maxNodesPerPass );

        std::unique_ptr< boost::progress_display > showProgress;
        if( numberOfPasses > 1 )
        {
            LBINFO << "Multipass rendering. Number of passes: " << numberOfPasses << std::endl;
            showProgress.reset( new boost::progress_display( numberOfPasses ));
        }

        for( uint32_t i = 0; i < numberOfPasses; ++i )
        {
            uint32_t renderStages = RENDER_FRAME;
            if( i == 0 )
                renderStages |= RENDER_BEGIN;

            if( i == numberOfPasses - 1u )
                renderStages |= RENDER_END;

            const uint32_t startIndex = i * maxNodesPerPass;
            const uint32_t endIndex = ( i + 1 ) * maxNodesPerPass;
            const NodeIds nodesPerPass( nodeIdsCopy.begin() + startIndex,
                                        endIndex > nodeIdsCopy.size() ? nodeIdsCopy.end() :
                                        nodeIdsCopy.begin() + endIndex );

            createAndExecuteSyncPass( nodesPerPass,
                                      renderInputs,
                                      renderer,
                                      renderStages );
            if( numberOfPasses > 1 )
                ++(*showProgress);
        }

        PipeFilterT< HistogramFilter > histogramFilter( "HistogramFilter",
                                                        *histogramCache,
                                                        *dataCache,
                                                        renderInputs.dataSource );
        histogramFilter.getPromise( "Frustum" ).set( renderInputs.frameInfo.frustum );
        histogramFilter.connect( "Histogram", sendHistogramFilter, "Histogram" );
        histogramFilter.getPromise( "RelativeViewport" ).set( renderInputs.viewport );
        histogramFilter.getPromise( "DataSourceRange" ).set( renderInputs.dataSourceRange );
        histogramFilter.getPromise( "NodeIds" ).set( nodeIdsCopy );

        sendHistogramFilter.getPromise( "RelativeViewport" ).set( renderInputs.viewport );
        sendHistogramFilter.getPromise( "Id" ).set( renderInputs.frameInfo.frameId );

        histogramFilter.schedule( _computeExecutor );
        sendHistogramFilter.schedule( _computeExecutor );

        const UniqueFutureMap futures( visibleSetGenerator.getPostconditions( ));
        statistics.nAvailable = futures.get< NodeIds >( "VisibleNodes" ).size();
        statistics.nNotAvailable = 0;
        statistics.nRenderAvailable = statistics.nAvailable;
    }

    void createAndExecuteSyncPass( NodeIds nodeIds,
                                   const RenderInputs& renderInputs,
                                   Renderer& renderer,
                                   const uint32_t renderStages )
    {

        Pipeline renderPipeline;

        PipeFilterT< RenderFilter > renderFilter( "RenderFilter",
                                                  renderInputs.dataSource,
                                                  renderer );
        renderFilter.getPromise( "RenderInputs" ).set( renderInputs );
        renderFilter.getPromise( "RenderStages" ).set( renderStages );

        PipeFilterT< GLRenderUploadFilter > renderUploader( "RenderUploader",
                                                            *dataCache,
                                                            *textureCache,
                                                            *texturePool,
                                                            nUploadThreads,
                                                            _uploadExecutor );

        renderUploader.getPromise( "RenderInputs" ).set( renderInputs );
        renderUploader.getPromise( "NodeIds" ).set( nodeIds );
        renderUploader.connect( "TextureCacheObjects", renderFilter, "CacheObjects" );

        renderPipeline.schedule( _renderExecutor );
        renderUploader.schedule( _uploadExecutor );
        renderFilter.execute();
    }


    void renderAsync( RenderStatistics& statistics,
                      Renderer& renderer,
                      const RenderInputs& renderInputs )
    {
        PipeFilter sendHistogramFilter = renderInputs.filters.find( "SendHistogramFilter" )->second;
        PipeFilter preRenderFilter = renderInputs.filters.find( "PreRenderFilter" )->second;
        PipeFilter redrawFilter = renderInputs.filters.find( "RedrawFilter" )->second;
        PipeFilterT< HistogramFilter > histogramFilter( "HistogramFilter",
                                                        *histogramCache,
                                                        *dataCache,
                                                        renderInputs.dataSource );
        histogramFilter.getPromise( "Frustum" ).set( renderInputs.frameInfo.frustum );
        histogramFilter.connect( "Histogram", sendHistogramFilter, "Histogram" );
        histogramFilter.getPromise( "RelativeViewport" ).set( renderInputs.viewport );
        histogramFilter.getPromise( "DataSourceRange" ).set( renderInputs.dataSourceRange );
        sendHistogramFilter.getPromise( "RelativeViewport" ).set( renderInputs.viewport );
        sendHistogramFilter.getPromise( "Id" ).set( renderInputs.frameInfo.frameId );
        preRenderFilter.getPromise( "Frustum" ).set( renderInputs.frameInfo.frustum );

        Pipeline renderPipeline;
        Pipeline uploadPipeline;

        PipeFilterT< RenderFilter > renderFilter( "RenderFilter",
                                                  renderInputs.dataSource,
                                                  renderer );

        PipeFilter visibleSetGenerator =
                renderPipeline.add< VisibleSetGeneratorFilter >(
                    "VisibleSetGenerator", renderInputs.dataSource );
        setupVisibleGeneratorFilter( visibleSetGenerator, renderInputs );

        PipeFilter renderingSetGenerator =
                renderPipeline.add< RenderingSetGeneratorFilter< TextureObject >>(
                    "RenderingSetGenerator", *textureCache );

        visibleSetGenerator.connect( "VisibleNodes", renderingSetGenerator, "VisibleNodes" );
        renderingSetGenerator.connect( "CacheObjects", renderFilter, "CacheObjects" );
        renderingSetGenerator.connect( "NodeIds", histogramFilter, "NodeIds" );
        renderingSetGenerator.connect( "RenderingDone", redrawFilter, "RenderingDone" );
        visibleSetGenerator.connect( "VisibleNodes", preRenderFilter, "VisibleNodes" );

        PipeFilterT< GLRenderUploadFilter > renderUploader( "RenderUploader",
                                                           *dataCache,
                                                           *textureCache,
                                                           *texturePool,
                                                           nUploadThreads,
                                                           _uploadExecutor );

        renderUploader.getPromise( "RenderInputs" ).set( renderInputs );
        visibleSetGenerator.connect( "VisibleNodes", renderUploader, "NodeIds" );

        renderFilter.getPromise( "RenderInputs" ).set( renderInputs );
        renderFilter.getPromise( "RenderStages" ).set( RENDER_ALL );

        redrawFilter.schedule( _renderExecutor );
        renderPipeline.schedule( _renderExecutor );
        renderUploader.schedule( _asyncUploadExecutor );
        sendHistogramFilter.schedule( _computeExecutor );
        histogramFilter.schedule( _computeExecutor );
        preRenderFilter.execute();
        renderFilter.execute();

        const UniqueFutureMap futures( renderingSetGenerator.getPostconditions( ));
        statistics = futures.get< RenderStatistics >( "RenderStatistics" );
    }

    void initTextureCache( const RenderInputs& renderInputs )
    {
        if( textureCache.get( ))
            return;

        ScopedLock lock( _initMutex );
        if( textureCache.get( ))
            return;

        const RendererParameters& vrParams = renderInputs.vrParameters;
        const size_t gpuMem = vrParams.getMaxGPUCacheMemoryMB() * LB_1MB;
        textureCache.reset( new TextureCache( "TextureCache", gpuMem ));
        texturePool.reset( new TexturePool( renderInputs.dataSource ));

        if( !dataCache )
            dataCache.reset( new DataCache( "Data Cache",
                                            vrParams.getMaxCPUCacheMemoryMB() * LB_1MB ));

        if( !histogramCache )
            histogramCache.reset( new HistogramCache( "Histogram Cache",
                                                      32 * LB_1MB )); // 32 MB
    }

    void render( RenderStatistics& statistics,
                 Renderer& renderer,
                 const RenderInputs& renderInputs )
    {
        initTextureCache( renderInputs );
        if( renderInputs.vrParameters.getSynchronousMode( ))
            renderSync( statistics, renderer, renderInputs );
        else
            renderAsync( statistics, renderer, renderInputs );
    }

    SimpleExecutor _renderExecutor;
    SimpleExecutor _computeExecutor;
    SimpleExecutor _uploadExecutor;
    SimpleExecutor _asyncUploadExecutor;
    boost::mutex _initMutex;
};

GLRaycastPipeline::GLRaycastPipeline( const std::string& name )
    : RenderPipelinePlugin( name )
    , _impl( new GLRaycastPipeline::Impl())
{}

GLRaycastPipeline::~GLRaycastPipeline()
{}

RenderStatistics GLRaycastPipeline::render( Renderer& renderer, const RenderInputs& renderInputs )
{
    RenderStatistics statistics;
    _impl->render( statistics, renderer, renderInputs );
    return statistics;
}
}
