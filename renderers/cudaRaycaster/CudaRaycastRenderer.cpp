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

#include "CudaRaycastRenderer.h"
#include "CudaTextureObject.h"
#include "cuda/Renderer.cuh"

#include <livre/core/configuration/RendererParameters.h>
#include <livre/lib/cache/DataObject.h>

#include <livre/core/cache/Cache.h>
#include <livre/core/data/DataSource.h>
#include <livre/core/data/VolumeInformation.h>
#include <livre/core/render/Frustum.h>
#include <livre/core/data/LODNode.h>
#include <livre/core/settings/RenderSettings.h>
#include <livre/core/render/RenderInputs.h>

#include <lunchbox/debug.h>
#include <livre/core/util/PluginRegisterer.h>
#include <GL/glew.h>

namespace livre
{
// Sort helper function for sorting the textures with their distances to viewpoint
struct DistanceOperator
{
    explicit DistanceOperator( const DataSource& dataSource, const Frustum& frustum )
        : _frustum( frustum )
        , _dataSource( dataSource )
    { }

    bool operator()( const ConstCacheObjectPtr& rb1, const ConstCacheObjectPtr& rb2 )
    {
        const LODNode& lodNode1 = _dataSource.getNode( NodeId( rb1->getId( )));
        const LODNode& lodNode2 = _dataSource.getNode( NodeId( rb2->getId( )));

        const float distance1 = ( _frustum.getMVMatrix() *
                                  lodNode1.getWorldBox().getCenter() ).length();
        const float distance2 = ( _frustum.getMVMatrix() *
                                  lodNode2.getWorldBox().getCenter() ).length();
        return  distance1 < distance2;
    }
    const Frustum& _frustum;
    const DataSource& _dataSource;
};

namespace
{
const uint32_t maxSamplesPerRay = 32;
const uint32_t minSamplesPerRay = 512;
const uint32_t SH_UINT = 0u;
const uint32_t SH_INT = 1u;
const uint32_t SH_FLOAT = 2u;
PluginRegisterer< CudaRaycastRenderer, const std::string& > registerer;
}

struct CudaRaycastRenderer::Impl
{
    Impl()
    {}

    ~Impl()
    {}

    void update( const lexis::render::ClipPlanes& clipPlanes,
                 const lexis::render::ColorMap& colorMap )
    {
        _cudaRenderer.update( clipPlanes, colorMap );
    }

    uint32_t getShaderDataType( const VolumeInformation& volInfo ) const
    {
        switch( volInfo.dataType )
        {
            case DT_UINT8:
            case DT_UINT16:
            case DT_UINT32:
                return SH_UINT;
            case DT_FLOAT:
                return SH_FLOAT;
            case DT_INT8:
            case DT_INT16:
            case DT_INT32:
                return SH_INT;
            case DT_UNDEFINED:
            default:
                LBTHROW( std::runtime_error( "Unsupported type in the shader." ));
        }
    }

    void preRender( const RenderInputs& renderInputs, const ConstCacheObjects& renderData )
    {
        update( renderInputs.renderSettings.getClipPlanes(),
                renderInputs.renderSettings.getColorMap( ));

        const auto& volInfo = renderInputs.dataSource.getVolumeInfo();
        if( renderInputs.vrParameters.getSamplesPerRay() == 0 ) // Find sampling rate
        {
            uint32_t maxLOD = 0;
            for( const auto& rb : renderData )
            {
                const LODNode& lodNode = renderInputs.dataSource.getNode( NodeId( rb->getId()));
                const uint32_t level = lodNode.getRefLevel();
                if( level > maxLOD )
                    maxLOD = level;
            }

            const float maxVoxelDim = volInfo.voxels.find_max();
            const float maxVoxelsAtLOD = maxVoxelDim /
                    (float)( 1u << ( volInfo.rootNode.getDepth() - maxLOD - 1 ));
            // Nyquist limited nb of samples according to voxel size
            _computedSamplesPerRay = std::max( maxVoxelsAtLOD, (float)minSamplesPerRay );
        }

        glDisable( GL_LIGHTING );
        glEnable( GL_CULL_FACE );
        glDisable( GL_DEPTH_TEST );
        glDisable( GL_BLEND );

        const Vector3f halfWorldSize = volInfo.worldSize / 2.0f;
        Vector4i glViewPort;
        glGetIntegerv( GL_VIEWPORT, glViewPort.array );

        const Frustum& frustum = renderInputs.frameInfo.frustum;
        const cuda::ViewData viewData = {
                                          frustum.getEyePos(),
                                          glViewPort,
                                          frustum.getInvProjMatrix(),
                                          frustum.getMVMatrix(),
                                          frustum.getInvMVMatrix(),
                                          -halfWorldSize,
                                          halfWorldSize,
                                          frustum.nearPlane()
                                        };

        _cudaRenderer.preRender( viewData );
    }

    void render( const RenderInputs& renderInputs, const ConstCacheObjects& renderData )
    {
        if( renderData.empty( ))
            return;

        auto renderDataCopy = renderData;
        const DistanceOperator distanceOp( renderInputs.dataSource,
                                           renderInputs.frameInfo.frustum );
        std::sort( renderDataCopy.begin(), renderDataCopy.end(), distanceOp );

        const VolumeInformation& volInfo = renderInputs.dataSource.getVolumeInfo();
        cuda::NodeDatas nodeDatas;
        nodeDatas.reserve( renderData.size( ));
        cuda::TexturePool* pool = 0;
        for( const auto& cacheObject: renderDataCopy )
        {
            const ConstCudaTextureObjectPtr cudaObject =
                    std::static_pointer_cast< const CudaTextureObject >( cacheObject );
            const LODNode& lodNode =
                    renderInputs.dataSource.getNode( NodeId( cudaObject->getId( )));
            const Boxf& aabb = lodNode.getWorldBox();
            nodeDatas.push_back( { cudaObject->getTexPosition(),
                                   cudaObject->getTexSize(),
                                   aabb.getMin(),
                                   aabb.getSize( )});

            if( !pool )
                pool = &cudaObject->getTexturePool()._getCudaTexturePool();
        }

        const Vector3f halfWorldSize = volInfo.worldSize / 2.0;
        Vector4i glViewPort;
        glGetIntegerv( GL_VIEWPORT, glViewPort.array );
        const Frustum& frustum = renderInputs.frameInfo.frustum;

        const cuda::ViewData viewData = {
                                          frustum.getEyePos(),
                                          glViewPort,
                                          frustum.getInvProjMatrix(),
                                          frustum.getMVMatrix(),
                                          frustum.getInvMVMatrix(),
                                          -halfWorldSize,
                                          halfWorldSize,
                                          frustum.nearPlane()
                                        };
        const cuda::RenderData rData = {
                                          _computedSamplesPerRay,
                                          renderInputs.vrParameters.getSamplesPerPixel(),
                                          maxSamplesPerRay,
                                          getShaderDataType( volInfo ),
                                          Vector2f( 0.0f, 255.0f )
                                        };

        _cudaRenderer.render( viewData,
                              nodeDatas,
                              rData,
                              *pool );
    }

    void postRender()
    {
        _cudaRenderer.postRender();
    }

    ::livre::cuda::Renderer _cudaRenderer;
    uint32_t _computedSamplesPerRay;
    CudaTexturePool* _texturePool;
};

CudaRaycastRenderer::CudaRaycastRenderer( const std::string& )
    : RendererPlugin( "cuda" )
    , _impl( new CudaRaycastRenderer::Impl( ))
{}

CudaRaycastRenderer::~CudaRaycastRenderer()
{}

void CudaRaycastRenderer::preRender( const RenderInputs& renderInputs,
                                     const ConstCacheObjects& renderData )
{
    _impl->preRender( renderInputs, renderData );
}

void CudaRaycastRenderer::render( const RenderInputs& renderInputs,
                                  const ConstCacheObjects& renderData )
{
    _impl->render( renderInputs, renderData );
}

void CudaRaycastRenderer::postRender( const RenderInputs&,
                                      const ConstCacheObjects& )
{
    _impl->postRender();
}

}
