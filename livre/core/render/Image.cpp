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
#include "Image.h"

#include <lunchbox/debug.h>
#include <FreeImage.h>

namespace livre
{

namespace
{
FIBITMAP* convertToRGBAf( const FIBITMAP *src, bool& isHdr )
{
    const FREE_IMAGE_TYPE imageType = FreeImage_GetImageType(src);
    isHdr = imageType == FIT_RGBF || imageType == FIT_RGBAF;
    switch( imageType )
    {
    case FIT_BITMAP:
    case FIT_UINT16:
    case FIT_FLOAT:
    case FIT_RGBF:
         return FreeImage_ConvertToRGBAF( src );
    case FIT_RGBAF:
        return src;
    default:
        LBTHROW( std::runtime_error( "Unsupported data format" ));
    }
    return 0;
}

FIBITMAP* openImage( const char* filename )
{
    FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
    // check the file signature and deduce its format
    // (the second argument is currently not used by FreeImage)
    fif = FreeImage_GetFileType( filename, 0 );
    if(fif == FIF_UNKNOWN)
    {
        // no signature ?
        // try to guess the file format from the file extension
        fif = FreeImage_GetFIFFromFilename(lpszPathName);
    }

    // check that the plugin has reading capabilities ...
    if((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading( fif ))
    {
        // ok, let's load the file
        FIBITMAP *dib = FreeImage_Load(fif, lpszPathName, flag);
        // unless a bad file format, we are done !
        return dib;
    }

    return 0;
}
}

struct Image::Impl
{
    Impl( const FIBITMAP* image, double gamma, double exposure )
       : _isHdr( false )
       , _image( FreeImage_TmoDrago03( image, gamma, exposure ))
    {}

    Impl( const Strings& resourceFolders, const std::string& imageFileName )
        : _isHdr( false )
        , _image( 0 )
    {
        for( const auto& resourceFolder: resourceFolders )
        {
            const std::string filePath = resourceFolder + "/" + file;
            _image = openImage( filePath.c_str( ));
            if( !_image )
                continue;

            FIBITMAP* image = convertToRGBAf( _image );
            if( image != _image )
            {
                FreeImage_Unload( _image );
                _image = image;
            }

            return;
        }

        LBTHROW( std::runtime_error( "Image cannot be loaded" ));
    }

    Vector4f getPixel( const size_t x, const size_t y ) const
    {
        const size_t pitch = FreeImage_GetPitch( _image );
        const FIRGBAF* data = (FIRGBAF*)FreeImage_GetBits( _image );
        const FIRGBAF* pixel = data + y * pitch +  x;
        return { pixel->red, pixel->green, pixel->blue, pixel->alpha };
    }

    Vector2ui getSize() const
    {
        return { FreeImage_GetWidth( _image ), FreeImage_GetHeight( _image ) };
    }

    ~Impl()
    {
        if( _image )
            FreeImage_Unload( _image );
    }

    bool _isHdr;
    FIBITMAP* _image;
};

bool Image::operator==( const Image& image ) const
{
    return _impl.get() == image._impl.get();
}

Image::Image( const Strings& resourceFolders, const std::string& imageFileName )
    : _impl( new Impl( resourceFolders, imageFileName ))
{}

Image::Image( const Image& image, double gamma, double exposure )
    : _impl( new Impl( image._impl->_image, gamma, exposure ))
{}

Image Image::getTonedImage( const double gamma, const double exposure ) const
{
    return Image( *this, gamma, exposure );
}

bool Image::isHDR() const
{
    return _impl->_isHdr;
}

Vector4f Image::getPixel( const size_t x, const size_t y ) const
{
    return _impl->getPixel( x, y );
}

Vector2ui Image::getSize() const
{
    return _impl->getSize();
}

}

