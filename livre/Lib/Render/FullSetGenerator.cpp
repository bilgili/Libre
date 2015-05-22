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

#include <livre/Lib/Render/FullSetGenerator.h>
#include <livre/Lib/Visitor/CollectionTraversal.h>
#include <livre/Lib/Visitor/DFSTraversal.h>
#include <livre/Lib/Cache/TextureObject.h>
#include <livre/Lib/Cache/TextureCache.h>

#include <livre/core/Data/VolumeDataSource.h>
#include <livre/core/Data/NodeId.h>
#include <livre/core/Maths/Maths.h>
#include <livre/core/Render/GLWidget.h>
#include <livre/core/Render/Frustum.h>

#include <boost/bind.hpp>

namespace livre
{

class ComputeLODTreeVisitor : public Das
{
public:

    ComputeLODTreeVisitor( LODTree& lodTree,
                           const Frustum& frustum,
                           uint32_t windowHeight,
                           float screenSpaceError,
                           NodeIdSet& allNodesSet )
        : _lodTree( lodTree ),
          _frustum( frustum ),
          _windowHeight( windowHeight ),
          _screenSpaceError( screenSpaceError ),
          _maxRecursionLevel( lodTree.getDataSource()->getDepth() ),
          _allNodesSet( allNodesSet )
    {}

    void visit( const NodeId& nodeId, VisitState& state ) override
    {
        const LODNode& node = _lodTree.getLODNode( nodeId );
        const Boxf& worldBox = node.getWorldBox();
        if( !_frustum.boxInFrustum( worldBox ) )
        {
            state.setVisitChild( false );
            return;
        }

        const Plane& nearPlane = _frustum.getWPlane( PL_NEAR );
        Vector3f vmin, vmax;
        nearPlane.getNearFarPoints( worldBox, vmin, vmax );

        const uint32_t lodHigh = std::min(
                    maths::getLODForPoint( vmin,
                                           _frustum,
                                           _windowHeight,
                                           _lodTree.getDataSource()->getWorldSpacePerVoxel(),
                                           _screenSpaceError ),
                                           _lodTree.getDataSource()->getDepth() );

        // If node at the right level add the node
        if( nodeId.getLevel() == lodHigh )
        {
            _allNodesSet.insert( nodeId );
            state.setVisitChild( false );
        }
        else
        {
            const uint32_t lodLow = std::max(
                        maths::getLODForPoint( vmin,
                                               _frustum,
                                               _windowHeight,
                                               _lodTree.getDataSource()->getWorldSpacePerVoxel(),
                                               _screenSpaceError ), 0u );

            BOOST_FOREACH( const NodeId& n, nodeId.getChildrenAtLevel( lodLow ) )
                state.addNextVisit( n );

        }
    }

    LODTree& _lodTree;
    const Frustum& _frustum;
    const uint32_t _windowHeight;
    const float _screenSpaceError;
    uint32_t _maxRecursionLevel;
    NodeIdSet& _allNodesSet;
};


ComputeFullLODSet::ComputeFullLODSet( const TextureCachePtr textureCachePtr,
                                      LODTreePtr lodTreePtr,
                                      float screenSpaceError )
    : ComputeRenderingLODSet( lodTreePtr ),
      _textureCachePtr( textureCachePtr ),
      _screenSpaceError( screenSpaceError )
{
}

void ComputeFullLODSet::generateRenderingSet( const GLWidget& widget,
                                              const Frustum& viewFrustum,
                                              NodeIdSet& requestedNodeSet,
                                              NodeIdSet& renderNodeSet )
{
     _generateRenderingSet( widget, viewFrustum, requestedNodeSet );

    BOOST_FOREACH( const NodeId& nodeId, requestedNodeSet )
    {
        if( _textureCachePtr->getObjectFromCache( nodeId.getId() )->isLoaded() )
            renderNodeSet.insert( nodeId );
        else
        {
            renderNodeSet.clear();
            return;
        }
    }

}


void ComputeFullLODSet::_generateRenderingSet( const GLWidget& widget,
                                               const Frustum& viewFrustum,
                                               NodeIdSet& allNodesSet )
{
    const NodeIds& rootNodeIds = _lodTreePtr->getRootNodes();

    std::vector< NodeIdSet > allNodesSets( rootNodeIds.size() );

    // OpenMP me !
    for( size_t i = 0; i < rootNodeIds.size(); ++i )
    {
        DFSTraversal dfsTraverser;
        ComputeLODTreeVisitor visibleSelector( *_lodTreePtr,
                                               viewFrustum,
                                               widget.getHeight(),
                                               _screenSpaceError,
                                               allNodesSets[ i ] );

        dfsTraverser.traverse( *_lodTreePtr, rootNodeIds[ i], visibleSelector );
    }

    // Add nodes to the list
    for( size_t i = 0; i < rootNodeIds.size(); ++i )
    {
        BOOST_FOREACH( const NodeId& nodeId, allNodesSets[ i ] )
        {
            if( nodeId.isValid() )
                allNodesSet.insert( nodeId );
        }
    }
}



}
