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

#include <livre/lib/render/CUDARenderer.h>

#include <livre/lib/render/cuda/Renderer.h>
#include <livre/lib/configuration/VolumeRendererParameters.h>
#include <livre/lib/cache/DataObject.h>

#include <livre/core/cache/Cache.h>
#include <livre/core/data/DataSource.h>
#include <livre/core/data/VolumeInformation.h>
#include <livre/core/render/Frustum.h>
#include <livre/core/data/LODNode.h>
#include <livre/core/settings/RenderSettings.h>

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

namespace
{
const uint32_t maxSamplesPerRay = 32;
const uint32_t minSamplesPerRay = 512;
const uint32_t SH_UINT = 0u;
const uint32_t SH_INT = 1u;
const uint32_t SH_FLOAT = 2u;
}

struct CUDARenderer::Impl
{
    Impl( const DataSource& dataSource,
          const Cache& dataCache,
          const uint32_t samplesPerRay,
          const uint32_t samplesPerPixel )
        : _nSamplesPerRay( samplesPerRay )
        , _nSamplesPerPixel( samplesPerPixel )
        , _computedSamplesPerRay( samplesPerRay )
        , _dataCache( dataCache )
        , _dataSource( dataSource )
        , _volInfo( _dataSource.getVolumeInfo( ))
        , _cudaClipPlanes( 0 )
        , _cudaRenderer( _volInfo.dataType, _volInfo.overlap )
    {}

    ~Impl()
    {}

    NodeIds order( const NodeIds& bricks, const Frustum& frustum ) const
    {
        NodeIds rbs = bricks;
        DistanceOperator distanceOp( _dataSource, frustum );
        std::sort( rbs.begin(), rbs.end(), distanceOp );
        return rbs;
    }

    void update( const RenderSettings& renderSettings,
                 const VolumeRendererParameters& renderParams )
    {
        _cudaRenderer.update( renderSettings.getColorMap( ));
        _nSamplesPerRay = renderParams.getSamplesPerRay();
        _computedSamplesPerRay = _nSamplesPerRay;
        _nSamplesPerPixel = renderParams.getSamplesPerPixel();
    }

    uint32_t getShaderDataType() const
    {
        switch( _volInfo.dataType )
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

    void preRender( const Frustum& frustum,
                    const NodeIds& renderBricks )
    {
        if( _nSamplesPerRay == 0 ) // Find sampling rate
        {
            uint32_t maxLOD = 0;
            for( const NodeId& rb : renderBricks )
            {
                const LODNode& lodNode = _dataSource.getNode( rb );
                const uint32_t level = lodNode.getRefLevel();
                if( level > maxLOD )
                    maxLOD = level;
            }

            const float maxVoxelDim = _volInfo.voxels.find_max();
            const float maxVoxelsAtLOD = maxVoxelDim /
                    (float)( 1u << ( _volInfo.rootNode.getDepth() - maxLOD - 1 ));
            // Nyquist limited nb of samples according to voxel size
            _computedSamplesPerRay = std::max( maxVoxelsAtLOD, (float)minSamplesPerRay );
        }

        glDisable( GL_LIGHTING );
        glEnable( GL_CULL_FACE );
        glDisable( GL_DEPTH_TEST );
        glDisable( GL_BLEND );
        glGetIntegerv( GL_DRAW_BUFFER, &_drawBuffer );
        glDrawBuffer( GL_NONE );

        const Vector3f halfWorldSize = _volInfo.worldSize / 2.0;
        Vector4ui glViewPort;
        glGetIntegerv( GL_VIEWPORT, glViewPort.array );
        const cuda::ViewData viewData = { frustum.getEyePos(),
                                          glViewPort,
                                          frustum.getInvProjMatrix(),
                                          frustum.getInvMVMatrix(),
                                          { -halfWorldSize, halfWorldSize },
                                          frustum.getNearPlane()
                                        };

        _cudaRenderer.preRender( viewData );
    }

    void render( const Frustum& frustum,
                 const ClipPlanes& planes,
                 const NodeIds& renderBricks )
    {
        cuda::NodeDatas nodeDatas;
        nodeDatas.reserve( bricks.size( ));
        for( const NodeId& nodeId: renderBricks )
        {
            const ConstDataObjectPtr dataObj =
                    std::static_pointer_cast< const DataObject >( _dataCache.get( nodeId.getId( )));
            const LODNode& lodNode = _dataSource.getNode( nodeId );
            const Boxf& aabb = lodNode.getWorldBox();
            nodeDatas.emplace_back( dataObj->getDataPtr(),
                                    dataObj->getSize(),
                                    lodNode.getBlockBox(),
                                    aabb.getMin(),
                                    aabb.getMax( ));
        }

        const Vector3f halfWorldSize = _volInfo.worldSize / 2.0;
        Vector4ui glViewPort;
        glGetIntegerv( GL_VIEWPORT, glViewPort.array );
        const cuda::ViewData viewData = { frustum.getEyePos(),
                                          glViewPort,
                                          frustum.getInvProjMatrix(),
                                          frustum.getInvMVMatrix(),
                                          -halfWorldSize,
                                          halfWorldSize,
                                          frustum.getNearPlane()
                                        };
        const cuda::RenderData renderData = { _nSamplesPerRay,
                                              _nSamplesPerPixel,
                                              _maxSamplesPerRay,
                                              getShaderDataType(),
                                              Vector2f( 0.0f, 255.0f ),
                                              _volInfo.overlap
                                            };

        _cudaRenderer.render( planes, viewData, nodeDatas, renderData );
    }

    void postRender()
    {
        _cudaRenderer.postRender();

        glDrawBuffer( _drawBuffer );
        glDisable(GL_DEPTH_TEST);
        glRasterPos2i(0, 0);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, _cudaRenderPBO.getId( ));
        glDrawPixels( _cudaRenderPBO.getWidth(),
                      _cudaRenderPBO.getHeight(),
                      GL_RGBA, GL_FLOAT, 0);
        glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
    }

    uint32_t _nSamplesPerRay;
    uint32_t _nSamplesPerPixel;
    uint32_t _computedSamplesPerRay;
    const Cache& _dataCache;
    const DataSource& _dataSource;
    const VolumeInformation& _volInfo;
    GLint _drawBuffer;

    ::livre::cuda::Renderer _cudaRenderer;
};

CUDARenderer::CUDARenderer( const DataSource& dataSource,
                            const Cache& dataCache,
                            const uint32_t samplesPerRay,
                            const uint32_t samplesPerPixel )
    : _impl( new CUDARenderer::Impl( dataSource,
                                     dataCache,
                                     samplesPerRay,
                                     samplesPerPixel ))
{}

CUDARenderer::~CUDARenderer()
{}

void CUDARenderer::update( const RenderSettings& renderSettings,
                           const VolumeRendererParameters& renderParams )
{
    _impl->update( renderSettings, renderParams );
}


NodeIds CUDARenderer::order( const NodeIds& bricks,
                             const Frustum& frustum ) const
{
    return _impl->order( bricks, frustum );
}

void CUDARenderer::_preRender( const Frustum& frustum,
                               const ClipPlanes&,
                               const PixelViewport&,
                               const NodeIds& renderBricks )
{
    _impl->preRender( frustum, renderBricks );
}

void CUDARenderer::_render( const Frustum& frustum,
                            const ClipPlanes& planes,
                            const PixelViewport& viewport,
                            const NodeIds& orderedBricks )
{
    _impl->render( frustum, planes, orderedBricks );
}

void CUDARenderer::_postRender( const Frustum&,
                                const ClipPlanes&,
                                const PixelViewport&,
                                const NodeIds& bricks )
{
    _impl->postRender();
}

}
