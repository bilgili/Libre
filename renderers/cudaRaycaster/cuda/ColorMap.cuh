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

#ifndef _Cuda_ColorMap_h_
#define _Cuda_ColorMap_h_

#include "cuda.h"

#include <lexis/render/ColorMap.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <memory>

namespace livre
{
namespace cuda
{
class ColorMap
{
public:
    ColorMap();
    ~ColorMap();
    void upload( const lexis::render::ColorMap& colorMap );
    cudaTextureObject_t getTexture() const  { return _texture; }

private:
    cudaTextureObject_t _texture;
    cudaArray_t _array;
};
}
}
#endif // _Cuda_ColorMap_h_

