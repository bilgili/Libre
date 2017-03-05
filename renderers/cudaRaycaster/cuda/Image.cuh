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

#ifndef _Cuda_Image_h_
#define _Cuda_Image_h_

#include "cuda.h"
#include "math.cuh"

#include <cuda_runtime_api.h>
#include <livre/core/mathTypes.h>

#define PI 3.141592654f

namespace livre
{
namespace cuda
{

class Image
{
public:
    Image( const ::livre::Image& image )
    {}

    void getTexture() const  { return _textureObject; }

private:
    cudaTextureObject_t _textureObject;
};

}
}
#endif // _Cuda_LightSource_h_

