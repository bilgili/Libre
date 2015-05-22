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

#include <livre/Lib/Render/AvailableSetGenerator.h>
#include <livre/Lib/Visitor/CollectionTraversal.h>
#include <livre/Lib/Visitor/DFSTraversal.h>
#include <livre/Lib/Cache/TextureObject.h>
#include <livre/Lib/Cache/TextureCache.h>

#include <livre/core/Data/VolumeDataSource.h>
#include <livre/core/Data/NodeId.h>
#include <livre/core/Dash/DashRenderNode.h>
#include <livre/core/Dash/DashTree.h>
#include <livre/core/Maths/Maths.h>
#include <livre/core/Render/GLWidget.h>

#include <boost/bind.hpp>

namespace livre
{
void removeChildren( const NodeId& nodeId,
                     uint32_t maxLevel,
                     NodeIdSet& renderNodeSet );

void removeSiblings( const NodeId& nodeId,
                     uint32_t maxLevel,
                     NodeIdSet& renderNodeSet )
{
    const NodeIds& siblings = nodeId.getSiblings();
    BOOST_FOREACH( const NodeId& s, siblings)
    {
        renderNodeSet.erase( s );
        removeChildren( s, maxLevel, renderNodeSet );
    }
}

void removeChildren( const NodeId& nodeId,
                     uint32_t maxLevel,
                     NodeIdSet& renderNodeSet )
{
    if( nodeId.getLevel() == maxLevel - 1 )
        return;

    const NodeIds& children = nodeId.getChildren();
    BOOST_FOREACH( const NodeId& c, children)
    {
        renderNodeSet.erase( c );
        removeSiblings( c, maxLevel, renderNodeSet );
    }

}

bool areSiblingsLoaded( const LODTree& lodTree,
                        const Frustum& frustum,
                        const NodeId& nodeId,
                        const TextureCache& textureCache )
{
    const NodeIds& siblings = nodeId.getSiblings();
    BOOST_FOREACH( const NodeId& s, siblings)
    {
       if( frustum.boxInFrustum( lodTree.getLODNode( nodeId ).getWorldBox()) &&
               !textureCache.getObjectFromCache( s.getId() )->isLoaded() )
           return false;
    }

    return true;
}

bool isParentLoaded( const NodeId& nodeId, const TextureCache& textureCache )
{
    return textureCache.getObjectFromCache( nodeId.getId() )->isLoaded();
}

AvailableSetGenerator::AvailableSetGenerator( DashTreePtr tree,
                                              const uint32_t windowHeight,
                                              const float screenSpaceError )
    : ComputeFullLODSet( textureCachePtr, lodTreePtr, screenSpaceError )
{
}

void AvailableSetGenerator::generateRenderingSet(const GLWidget& widget,
                                                  const Frustum& viewFrustum,
                                                  NodeIdSet& requestedNodeSet,
                                                  NodeIdSet& renderNodeSet )
{
    NodeIdSet renderSet;
    _generateRenderingSet( widget, viewFrustum, renderSet );

    // NodeIds (std::set<NodeId>) keeps the list of NodeIds in order of their level of detail order.
    // The below algorithm uses this information to traverse the set.
    while( !renderSet.empty() )
    {
        const NodeId& nodeId = *renderSet.end() ;
        if( nodeId.isRoot() )
        {
            renderNodeSet.insert( nodeId );
            renderSet.erase( nodeId );
            continue;
        }

        if( areSiblingsLoaded( *_lodTreePtr, viewFrustum, nodeId, *_textureCachePtr ) )
        {
            const NodeIds& siblings = nodeId.getSiblings();
            renderNodeSet.insert( siblings.begin(), siblings.end() );
            removeChildren( nodeId, _lodTreePtr->getDataSource()->getDepth(), renderNodeSet );
            removeChildren( nodeId, _lodTreePtr->getDataSource()->getDepth(), renderSet );
            removeSiblings( nodeId, _lodTreePtr->getDataSource()->getDepth(), renderSet );
        }
        else
        {
            removeSiblings( nodeId, _lodTreePtr->getDataSource()->getDepth(), renderNodeSet );
            const NodeIds& parents = nodeId.getParents();
            BOOST_FOREACH( const NodeId& p, parents )
            {
                if( isParentLoaded( p, *_textureCachePtr ) )
                {
                    removeChildren( p, _lodTreePtr->getDataSource()->getDepth(), renderNodeSet );
                    removeChildren( p, _lodTreePtr->getDataSource()->getDepth(), renderSet );
                    renderSet.insert( p );
                    break;
                }
            }
        }
    }

    requestedNodeSet = renderNodeSet;
}

}
