/* Copyright (c) 2011-2016, EPFL/Blue Brain Project
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

#include <livre/eq/render/ospray/OSPrayVolume.h>

#include <livre/core/data/DataSource.h>
#include <livre/lib/cache/TextureDataCache.h>

#include <ospray/common/Data.h>
#include <ospray/common/Core.h>
#include <ospray/common/Library.h>

#include <OSPrayVolume_ispc.h>

namespace livre
{

OSPrayVolume::OSPrayVolume()
    : _dataCache( 0 )
    , _dataSource( 0 )
    , _volInfo( 0 )
{
}

OSPrayVolume::~OSPrayVolume() {}

std::string OSPrayVolume::toString() const
{
    return( "livre::OSPrayVolume" );
}

int OSPrayVolume::setRegion( const void*,
                             const ospray::vec3i&,
                             const ospray::vec3i& )
{
    LBTHROW( std::runtime_error( "OSPrayVolume does not support setting regions"));
}

void OSPrayVolume::createEquivalentISPC()
{
    if( ispcEquivalent != 0 )
        return;

    ispcEquivalent = ispc::constructISPCVolume( this /*, ospBBox */ );
}

void OSPrayVolume::commit()
{
    if( !ispcEquivalent )
        return;

    ospray::Data* nodeIdsData = getParamData( "nodeIds",  0 );
    uint64_t* nodeIds = (uint64_t*)nodeIdsData->data;
    const size_t count = getParam1i( "count",  0 );
    ospray::Data* cachePtr = getParamData( "dataCache",  0 );
    _dataCache = (const TextureDataCache *)cachePtr->data;
    _dataSource = &_dataCache->getDataSource();
    _volInfo = &_dataSource->getVolumeInfo();
    ispc::setListOfNodes( ispcEquivalent, nodeIds, count );
}

void OSPrayVolume::computeSamples( float** results,
                                   const ospray::vec3f *worldCoordinates,
                                   const size_t& count )
{
    *results = new float[ count ];
    float* ispcResults = new float[ count ];
    ispc::computeSamples( ispcEquivalent,
                          ispcResults,
                          (const ispc::vec3f *)worldCoordinates,
                          count );

    // Copy samples and free ISPC results memory
    memcpy( *results, ispcResults, count * sizeof( float ));
    delete [] ispcResults;
}

float OSPrayVolume::sample( const uint64_t* nodeIds,
                            const size_t count,
                            const float x,
                            const float y,
                            const float z ) const
{
    NodeId intersectionId( INVALID_NODE_ID );
    for( size_t i = 0; i < count; ++i )
    {
        const LODNode& lodNode = _dataSource->getNode( NodeId( nodeIds[ i ]));
        if( lodNode.getWorldBox().isIn( Vector3f( x, y, z )))
        {
            intersectionId = NodeId( nodeIds[ i ]);
            break;
        }
    }

    if( intersectionId.getId() == INVALID_NODE_ID )
        return 0.0f;

    // Compute sample
    return 0.1f;
}

}

namespace ospray
{
// A volume type with XYZ storage order. The voxel data is provided by the
// application via a shared data buffer.
OSP_REGISTER_VOLUME(livre::OSPrayVolume, livre_ospray_volume)
}

namespace
{
float sampleLivreData( void* osprayVolume,
                       uint64_t* nodeIds,
                       uint64_t count,
                       float x,
                       float y,
                       float z )
{
    const ::livre::OSPrayVolume* volume = static_cast< ::livre::OSPrayVolume* >( osprayVolume );
    return volume->sample( nodeIds, count, x, y, z );
}
}
