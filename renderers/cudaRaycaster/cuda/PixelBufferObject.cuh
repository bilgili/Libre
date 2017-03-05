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

#ifndef _Cuda_PixelBufferObject_h_
#define _Cuda_PixelBufferObject_h_

#include "cuda.cuh"
#include <cuda_runtime.h>
#include <memory>
#include <GL/glew.h>

namespace livre
{
namespace cuda
{

/** Cuda representation of the pixel buffer */
class PixelBufferObject
{
public:

    /** Constructor */
    PixelBufferObject();
    ~PixelBufferObject();

    /** @return id of the pixel buffer. If none is initialized returns -1u */
    GLuint getId() const { return _pbo; }

    /**
     * Resizes the color buffer
     * @param width of the color buffer
     * @param height of the color buffer
     */
    void resize( unsigned int width, unsigned int height );

    /** Maps the color buffer to cuda kernel for rendering */
    void mapBuffer();

    /** Unmaps the color buffer from the cuda kernel */
    void unmapBuffer();

    /** @return the size of the PBO in bytes */
    size_t getBufferSize() const;

    /** @return the RGBAf pixel buffer pointer ( used in kernel ) */
    float4* getBuffer() const { return _buffer; }

    /** @return the width of the pixel buffer */
    unsigned int getWidth() const { return _width; }

    /** @return the height of the pixel buffer */
    unsigned int getHeight() const { return _height; }

private:

    float4* _buffer;
    cudaGraphicsResource_t _pboResource;
    GLuint _pbo;
    unsigned int _width;
    unsigned int _height;
};
}
}
#endif // _Cuda_PixelBufferObject_h_

