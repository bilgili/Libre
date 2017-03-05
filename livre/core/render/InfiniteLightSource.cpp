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
#include "InfiniteLightSource.h"

namespace livre
{
namespace
{
const float PI = 3.141592654f;
const float INV_TWOPI = 1.0f / ( 3.141592654f * 2.0f );
const float INV_PI = 1.0f / 3.141592654f;

inline float getTheta( const Vector3f& v )
{
    return std::acos( std::min( std::max( v[ 2 ], -1.f ), 1.f ));
}
inline float getPhi( const Vector3f& v )
{
    const float p = std::atan2( v[ 1 ], v[ 0 ]);
    return ( p < 0.0f ) ? p + 2.0f * PI : p;
}
}
struct InfiniteLightSource::Impl
{
    Impl( const Image& image )
        : _image( image )
    {}

    Vector4f InfiniteLightSource::getSample( const Vector3f& dir )
    {
        const Vector2ui& size = _image.getSize();
        const size_t x = getPhi( dir ) * INV_TWOPI * size[ 0 ];
        const size_t y = getTheta( dir ) * INV_PI * size[ 1 ];
        return _image.getPixel( x, y );
    }

    Image _image;
};

bool InfiniteLightSource::operator==( const LightSource& ls ) const
{
    const InfiniteLightSource* ils = dynamic_cast< const InfiniteLightSource* >( &ls );
    return ils && *_ils->_impl->_image == _impl->_image;
}

Vector4f InfiniteLightSource::getSample( const Vector3f& dir )
{
    return _impl->getSample( dir );
}

InfiniteLightSource::InfiniteLightSource( const Image& image )
    : LightSource( INFINITE )
    , _impl( new Impl( image ))
{}

}

