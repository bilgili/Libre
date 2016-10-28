/* Copyright (c) 2011-2015, EPFL/Blue Brain Project
 *                     Ahmet Bilgili <ahmet.bilgili@epfl.ch>
 *                     Daniel Nachbaur <daniel.nachbaur@epfl.ch>
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

template< class Allocator >
DataObject< Allocator >::DataObject( const CacheId& cacheId,
                                     Allocator& allocator,
                                     DataSource& dataSource )
    : _data( allocator )
{
    const DataType dataType = dataSource.getVolumeInfo().dataType;
    if( dataType == DT_UNDEFINED )
        LBTHROW( std::runtime_error( "Undefined data type" ));

    const NodeId nodeId( cacheId );
    _dataSource.getData( nodeId, _data );
    if( !_data )
        return false;
    return true;
}

template< class Allocator >
const void* DataObject< Allocator >::getDataPtr() const
{
    return _data->getData< void >();
}

size_t DataObject< Allocator >::getSize() const
{
    return _impl->_data->getAllocSize();
}

const void* DataObject< Allocator >::getDataPtr() const
{
    return _impl->getDataPtr();
}

}
