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

#include "CudaIrradianceCompute.h"
#include "cuda/IrradianceCompute.cuh"
#include "cuda/types.cuh"

#include <livre/lib/cache/DataObject.h>
#include <livre/core/cache/Cache.h>
#include <livre/core/render/RenderInputs.h>
#include <livre/core/data/DataSource.h>

namespace livre
{
namespace
{

Vector3ui getCoarseLevelSize( const DataSource& dataSource )
{
    const VolumeInformation& volInfo = dataSource.getVolumeInfo();
    const Vector3ui& lodZeroBlocks = volInfo.rootNode.getBlockSize();
    uint32_t iPos = 0, jPos = 0, kPos = 0;
    for( uint32_t k = 0; k < lodZeroBlocks[ 2 ]; ++k )
    {
        const NodeId nodeId( 0, { 0, 0, k }, 0 );
        const LODNode& lodNode = dataSource.getNode( nodeId );
        const Vector3ui& blockSize = lodNode.getBlockSize();
        kPos += blockSize[ 2 ];
    }

    for( uint32_t j = 0; j < lodZeroBlocks[ 1 ]; ++j )
    {
        const NodeId nodeId( 0, { 0, j, 0 }, 0 );
        const LODNode& lodNode = dataSource.getNode( nodeId );
        const Vector3ui& blockSize = lodNode.getBlockSize();
        jPos += blockSize[ 1 ];
    }

    for( uint32_t i = 0; i < lodZeroBlocks[ 0 ]; ++i )
    {
        const NodeId nodeId( 0, { i, 0, 0 }, 0 );
        const LODNode& lodNode = dataSource.getNode( nodeId );
        const Vector3ui& blockSize = lodNode.getBlockSize();
        iPos += blockSize[ 0 ];
    }
    std::cout << "volume size:" << Vector3ui( iPos, jPos, kPos ) << std::endl;
    return { iPos, jPos, kPos };
}

const uint32_t maxSamplesPerRay = 32;
const uint32_t minSamplesPerRay = 512;
const uint32_t SH_UINT = 0u;
const uint32_t SH_INT = 1u;
const uint32_t SH_FLOAT = 2u;

uint32_t getShaderDataType( const VolumeInformation& volInfo )
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
}
struct CudaIrradianceCompute::Impl
{
    Impl( DataSource& dataSource, DataCache& dataCache )
        : _dataSource( dataSource )
        , _dataCache( dataCache )
        , _currentTimeStep( -1u )
    {
        const VolumeInformation& volInfo = _dataSource.getVolumeInfo();
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
        _cudaCompute.reset( new ::livre::cuda::IrradianceCompute( volInfo.getBytesPerVoxel(),
                                                                  isSigned,
                                                                  isFloat,
                                                                  volInfo.compCount,
                                                                  volInfo.overlap,
                                                                  getCoarseLevelSize( dataSource )));
    }

    bool checkChanged( const lexis::render::ColorMap& colorMap,
                       const ConstLightSources& lightSources,
                       const uint32_t& timeStep )
    {

        if( timeStep != _currentTimeStep )
            return true;

        if( lightSources.size() != _currentLightSources.size( ))
            return true;

        for( size_t i = 0; i < lightSources.size(); ++i )
        {
            if( *lightSources[ i ] != *_currentLightSources[ i ] )
                return true;
        }

        if( colorMap != _currentColorMap )
            return true;

        return false;
    }

    bool update( const RenderInputs& renderInputs )
    {

        const auto& colorMap = renderInputs.renderSettings.getColorMap();
        const auto& timeStep = renderInputs.frameInfo.timeStep;
        const auto& lightSources = renderInputs.lightSources;

        if( !checkChanged( colorMap, lightSources, timeStep ) )
            return false;

        const VolumeInformation& volInfo = _dataSource.getVolumeInfo();
        if( timeStep != _currentTimeStep )
        {
            const Vector3ui& lodZeroBlocks = volInfo.rootNode.getBlockSize();

            uint32_t iPos = 0, jPos = 0, kPos = 0;
            for( uint32_t i = 0; i < lodZeroBlocks[ 0 ]; ++i )
            {
                for( uint32_t j = 0; j < lodZeroBlocks[ 1 ]; ++j )
                {
                    for( uint32_t k = 0; k < lodZeroBlocks[ 2 ]; ++k )
                    {
                        const NodeId nodeId( 0, { i, j, k }, timeStep );
                        const LODNode& lodNode = _dataSource.getNode( nodeId );
                        const Vector3ui& blockSize = lodNode.getBlockSize();

                        ConstDataObjectPtr data = _dataCache.load( nodeId.getId(), _dataSource );
                        if( !data )
                            continue;

                        _cudaCompute->upload( (const uint8_t *)data->getDataPtr(),
                                              { iPos, jPos, kPos },
                                              lodNode.getBlockSize() + ( volInfo.overlap * 2 ));
                        std::cout << "i :" << i << " j: " << j << " k: " << k << " "
                                  << "Position: " << Vector3ui( iPos, jPos, kPos )
                                  << "Blocks size: " << blockSize << std::endl;

                        kPos += blockSize[ 2 ];
                    }
                    kPos = 0;
                    const NodeId nodeId( 0, { i, j, 0 }, 0 );
                    const LODNode& lodNode = _dataSource.getNode( nodeId );
                    const Vector3ui& blockSize = lodNode.getBlockSize();
                    jPos += blockSize[ 1 ];
                }
                jPos = 0;
                const NodeId nodeId( 0, { i, 0, 0 }, 0 );
                const LODNode& lodNode = _dataSource.getNode( nodeId );
                const Vector3ui& blockSize = lodNode.getBlockSize();
                iPos += blockSize[ 0 ];
            }
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
        const cuda::RenderData renderData = {
                                              16,
                                              renderInputs.vrParameters.getSamplesPerPixel(),
                                              16,
                                              getShaderDataType( volInfo ),
                                              Vector2f( 0.0f, 255.0f )
                                            };

        _cudaCompute->update( renderData,
                              viewData,
                              colorMap,
                              lightSources );

        _currentTimeStep = timeStep;
        _currentLightSources = lightSources;
        _currentColorMap = colorMap;
        return true;
    }

    DataSource& _dataSource;
    DataCache& _dataCache;
    std::unique_ptr< ::livre::cuda::IrradianceCompute > _cudaCompute;
    uint32_t _currentTimeStep;
    ConstLightSources _currentLightSources;
    lexis::render::ColorMap _currentColorMap;
};

CudaIrradianceCompute::CudaIrradianceCompute( DataSource& dataSource, DataCache& dataCache )
    : _impl( new CudaIrradianceCompute::Impl( dataSource, dataCache ))
{}

bool CudaIrradianceCompute::update( const RenderInputs& renderInputs )
{
    return _impl->update( renderInputs );
}

::livre::cuda::IrradianceCompute& CudaIrradianceCompute::getCudaCompute() const
{
    return *_impl->_cudaCompute;
}

CudaIrradianceCompute::~CudaIrradianceCompute()
{}
}


