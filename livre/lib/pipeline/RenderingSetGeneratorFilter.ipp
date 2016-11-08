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

template< class CacheObjectT >
struct RenderingSetGenerator
{
    explicit RenderingSetGenerator( const Cache< CacheObjectT >& cache )
        : _cache( cache )
    {}

    bool hasParentInMap( const NodeId& childRenderNode,
                         const ConstCacheMap& cacheMap ) const
    {
        const NodeIds& parentNodeIds = childRenderNode.getParents();

        for( const NodeId& parentId : parentNodeIds )
            if( cacheMap.find( parentId.getId( )) != cacheMap.end() )
                return true;

        return false;
    }

    void collectLoadedData( const NodeId& nodeId, ConstCacheMap& cacheMap ) const
    {
        NodeId current = nodeId;
        while( current.isValid( ))
        {
            const NodeId& currentNodeId = current;
            const ConstCacheObjectPtr data = _cache.get( currentNodeId.getId( ));
            if( data )
            {
                cacheMap[ currentNodeId.getId() ] = data;
                break;
            }

            current = currentNodeId.isRoot() ? NodeId() :
                                               currentNodeId.getParent();
        }
    }

    ConstCacheObjects generateRenderingSet( const NodeIds& visibles,
                                            RenderStatistics& availability ) const
    {
        ConstCacheMap cacheMap;
        for( const NodeId& nodeId : visibles )
        {
            collectLoadedData( nodeId, cacheMap );
            cacheMap.count( nodeId.getId( )) > 0 ?
                        ++availability.nAvailable : ++availability.nNotAvailable;
        }

        if( visibles.size() != cacheMap.size( ))
        {
            ConstCacheMap::const_iterator it = cacheMap.begin();
            size_t previousSize = 0;
            do
            {
                previousSize = cacheMap.size();
                while( it != cacheMap.end( ))
                {
                    if( hasParentInMap( NodeId( it->first ), cacheMap ))
                        it = cacheMap.erase( it );
                    else
                        ++it;
                }
            }
            while( previousSize != cacheMap.size( ));
        }

        ConstCacheObjects cacheObjects;
        cacheObjects.reserve( cacheMap.size( ));
        for( ConstCacheMap::const_iterator it = cacheMap.begin();
             it != cacheMap.end(); ++it )
        {
            cacheObjects.push_back( it->second );
        }
        availability.nRenderAvailable = cacheObjects.size();
        return cacheObjects;
    }

    const Cache< CacheObjectT >& _cache;
};

template< class CacheObjectT >
struct RenderingSetGeneratorFilter< CacheObjectT >::Impl
{
    explicit Impl( const Cache< CacheObjectT >& cache )
        : _cache( cache )
    {}

    void execute( const FutureMap& input, PromiseMap& output ) const
    {
        RenderingSetGenerator< CacheObjectT > renderSetGenerator( _cache );

        ConstCacheObjects cacheObjects;
        size_t nVisible = 0;
        RenderStatistics cumulativeAvailability;
        for( const auto& visibles: input.get< NodeIds >( "VisibleNodes" ))
        {
            RenderStatistics avaliability;
            const ConstCacheObjects& objs = renderSetGenerator.generateRenderingSet( visibles,
                                                                                     avaliability );
            cacheObjects.insert( cacheObjects.end(), objs.begin(), objs.end( ));
            nVisible += visibles.size();
            cumulativeAvailability += avaliability;
        }

        output.set( "CacheObjects", cacheObjects );
        output.set( "RenderingDone", cacheObjects.size() == nVisible );
        output.set( "RenderStatistics", cumulativeAvailability );

        NodeIds ids;
        ids.reserve( cacheObjects.size( ));
        for( const auto& cacheObject: cacheObjects )
            ids.emplace_back( cacheObject->getId( ));

        output.set( "NodeIds", ids );
    }

    const Cache< CacheObjectT >& _cache;
};

template< class CacheObjectT >
RenderingSetGeneratorFilter< CacheObjectT >::RenderingSetGeneratorFilter(
        const Cache< CacheObjectT >& cache )
    : _impl( new RenderingSetGeneratorFilter::Impl( cache ))
{
}

template< class CacheObjectT >
RenderingSetGeneratorFilter< CacheObjectT >::~RenderingSetGeneratorFilter()
{
}

template< class CacheObjectT >
void RenderingSetGeneratorFilter< CacheObjectT >::execute( const FutureMap& input,
                                           PromiseMap& output ) const
{
    _impl->execute( input, output );
}

