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

#ifndef _InfiniteLightSource_h_
#define _InfiniteLightSource_h_

#include "LightSource.h"
#include "Image.h"

namespace livre
{

class InfiniteLightSource : public LightSource
{
public:
    InfiniteLightSource( const Image& image );
    bool operator==( const LightSource& ls ) const final;
    Vector4f getSample( const Vector3f& dir );
private:
    struct Impl;
    std::unique_ptr< Impl >  _impl;
};

}
#endif // _InfiniteLightSource_h_

