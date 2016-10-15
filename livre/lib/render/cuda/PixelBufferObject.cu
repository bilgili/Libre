/* Copyright (c) 2011-2016  Ahmet Bilgili <ahmetbilgili@gmail.com>
 *
 * This file is part of Livre <https://github.com/bilgili/Livre>
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

#include "PixelBufferObject.cuh"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <lunchbox/debug.h>

namespace livre
{
namespace cuda
{

PixelBufferObject::PixelBufferObject()
    : _pbo( -1u )
    , _width( -1u )
    , _height( -1u )
    , _nComponent( 4 )
{}

PixelBufferObject::~PixelBufferObject()
{}

GLuint PixelBufferObject::getId() const { return _pbo; }

void PixelBufferObject::resize( const unsigned int width, const unsigned int height )
{
    if( width == _width && height == _height )
        return;

    _width = width;
    _height = height;

    if( _pbo != -1u )
    {
        cudaGLUnregisterBufferObject( _pbo );
        glDeleteBuffersARB( 1, &_pbo );
    }

    glGenBuffers( 1, &_pbo );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, _pbo );
    glBufferData( GL_PIXEL_UNPACK_BUFFER, _width *
                                          _height *
                                          sizeof( float ) *
                                          _nComponent,
                                          0, GL_STREAM_DRAW );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
    cudaGLRegisterBufferObject( _pbo );

    const int ret = glGetError();
    if( ret != GL_NO_ERROR )
        LBTHROW( std::runtime_error( "Error resizing render pbo" ));
}

void PixelBufferObject::mapBuffer()
{
    cudaGLMapBufferObject( (void**)&_buffer, _pbo );
    cudaMemset( _buffer, 0, _width * _height * _nComponent );
}

void PixelBufferObject::unmapBuffer()
{
    cudaGLUnmapBufferObject( _pbo );
}

size_t PixelBufferObject::getBufferSize() const
{
    return _width * _height * _nComponent * sizeof( float );
}
}
}
