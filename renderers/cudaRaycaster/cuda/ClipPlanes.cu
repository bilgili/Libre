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

#include "ClipPlanes.cuh"

namespace livre
{
namespace cuda
{
ClipPlanes::ClipPlanes()
    : _nPlanes( 0 )
{}

ClipPlanes::~ClipPlanes()
{}

void ClipPlanes::upload( const lexis::render::ClipPlanes& clipPlanes )
{
    const unsigned int nPlanes = clipPlanes.getPlanes().size();

    if( nPlanes == 0 )
        return;

    for( size_t i = 0; i < nPlanes; ++i )
    {
        const ::lexis::render::Plane& plane = clipPlanes.getPlanes()[ i ];
        const float* normal = plane.getNormal();
        _clipPlanes[ i ] = make_float4( normal[ 0 ], normal[ 1 ], normal[ 2 ], plane.getD( ));
    }
    _nPlanes = nPlanes;
}
}
}
