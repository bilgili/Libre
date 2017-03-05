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

#ifndef _Image_h_
#define _Image_h_

#include <livre/core/types.h>
#include <livre/core/mathTypes.h>

namespace livre
{

class Image
{
public:

    Image( const Strings& resourceFolders, const std::string& imageFileName );
    Image getTonedImage( double gamma, double exposure ) const;
    bool isHDR() const;
    Vector4f getPixel( size_t x, size_t y ) const;
    Vector2ui getSize() const;

private:

    Image( const Image& image, double gamma, double exposure );

    struct Impl;
    std::shared_ptr< Impl >  _impl;
};

}
#endif // _Image_h_

