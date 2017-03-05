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

#ifndef _Cuda_LightSource_h_
#define _Cuda_LightSource_h_

#include "cuda.h"
#include "math.cuh"

#include <cuda_runtime_api.h>
#include <livre/core/mathTypes.h>

#define PI 3.141592654f

namespace livre
{
namespace cuda
{
struct LightData
{
    float3 color;
    float3 dir;
    float weight;
    float distance;
};

template< class LightSource >
::livre::cuda::LightSource* getCudaLight( const LightSource& ls ) const;

class LightSource
{
public:
    CUDA_CALL LightSource( const unsigned int samples )
        : _samples( samples )
    {}

    CUDA_CALL virtual ~LightSource()
    {}

    CUDA_CALL virtual void getSamples( LightData* array, const float3& position ) const = 0;
    CUDA_CALL unsigned int getSampleCount() const { return _samples; }

protected:
    unsigned int _samples;
};

class PointLightSource : public LightSource
{
public:
    CUDA_CALL PointLightSource( const float3& position, const float3& color )
        : LightSource( 1u )
        , _position( position )
        , _color( color )
    {}

    CUDA_CALL void getSamples( LightData* array, const float3& position ) const final
    {
        LightData& data = array[ 0 ];
        data.color = _color;
        data.dir = normalize( _position - position );
        data.weight = 1.0 / ( 4.0 * PI );
        data.distance = distance( _position,  position );
    }

private:
    const float3 _position;
    const float3 _color;
};

class DirectLightSource : public LightSource
{
public:
    CUDA_CALL DirectLightSource( const float3& direction, const float3& color )
        : LightSource( 1u )
        , _direction( direction )
        , _color( color )
    {}

    CUDA_CALL void getSamples( LightData* array, const float3& ) const final
    {
        LightData& data = array[ 0 ];
        data.color = _color;
        data.dir = _direction;
        data.weight = 1.0;
        data.distance = 1e10;
    }

private:

    const float3 _direction;
    const float3 _color;
};

class DirectLightSource : public LightSource
{
public:
    CUDA_CALL DirectLightSource( const CudaImage& cudaImage )
        : LightSource( 1u )
        , _direction( direction )
        , _color( color )
    {}

    CUDA_CALL void getSamples( LightData* array, const float3& ) const final
    {
        LightData& data = array[ 0 ];
        data.color = _color;
        data.dir = _direction;
        data.weight = 1.0;
        data.distance = 1e10;
    }

private:

    const float3 _direction;
    const float3 _color;
};

}
}
#endif // _Cuda_LightSource_h_

