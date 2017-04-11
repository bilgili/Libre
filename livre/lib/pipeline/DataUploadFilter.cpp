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

#include <livre/lib/cache/DataObject.h>
#include <livre/lib/pipeline/DataUploadFilter.h>

#include <livre/core/data/NodeId.h>
#include <livre/core/cache/Cache.h>

namespace livre
{

struct DataUploadFilter::Impl
{
    Impl( DataCache& dataCache )
        : _dataCache( dataCache )
    {}

    void execute( const tuyau::FutureMap& input, tuyau::PromiseMap& output ) const
    {
        ConstCacheObjects cacheObjects;
        const RenderInputs renderInputs = input.get< RenderInputs >( "RenderInputs" )[ 0 ];
        for( const auto& nodeIds: input.getFutures( "NodeIds" ))
            for( const auto& nodeId: nodeIds.get< NodeIds >( ))
            {
                const auto& cacheObj = _dataCache.load( nodeId.getId(), renderInputs.dataSource );
                if( cacheObj )
                    cacheObjects.push_back( cacheObj );


            }
        output.set( "DataCacheObjects", cacheObjects );
    }

    DataCache& _dataCache;
};

DataUploadFilter::DataUploadFilter( DataCache& dataCache )
    : _impl( new DataUploadFilter::Impl( dataCache ))
{
}

DataUploadFilter::~DataUploadFilter()
{}

void DataUploadFilter::execute( const tuyau::FutureMap& input,
                                tuyau::PromiseMap& output ) const
{
    _impl->execute( input, output );
}
}
