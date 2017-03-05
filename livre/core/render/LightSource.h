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

#ifndef _LightSource_h_
#define _LightSource_h_

#include <livre/core/mathTypes.h>

namespace livre
{

enum LightSourceType
{
    DIRECT,
    POINT,
    INFINITE
};

class LightSource
{

public:
    virtual ~LightSource() {}
    virtual bool operator==( const LightSource& ls ) const = 0;
    bool operator!=( const LightSource& ls ) const { return !( *this == ls ); }
    LightSourceType getType() const { return _type; }

protected:
    LightSource( LightSourceType type )
        : _type( type )
    {}
private:
    LightSourceType _type;
};

class PointLightSource : public LightSource
{
public:
    PointLightSource( const Vector3f& position_, const Vector3f& color_ )
        : LightSource( POINT )
        , position( position_ )
        , color( color_ )
    {}
    bool operator==( const LightSource& ls ) const final;
    const Vector3f position;
    const Vector3f color;
};

class DirectLightSource : public LightSource
{
public:
    DirectLightSource( const Vector3f& direction_, const Vector3f& color_ )
        : LightSource( DIRECT )
        , direction( direction_ )
        , color( color_ )
    {}
    bool operator==( const LightSource& ls ) const final;
    const Vector3f direction;
    const Vector3f color;
};

}
#endif // _LightSource_h_

