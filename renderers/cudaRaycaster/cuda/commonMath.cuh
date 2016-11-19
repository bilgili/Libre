#ifndef _CommonMath_Cuh_
#define _CommonMath_Cuh_

#include "math.cuh"

#include <cuda_runtime_api.h>

namespace livre
{
namespace cuda
{

#define EARLY_EXIT 0.999f
#define EPSILON 0.0000000001f

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
// implementation from nvidia sample
inline __device__ bool intersectBox( const float3& origin,
                                     const float3& dir,
                                     const float3& boxMin,
                                     const float3& boxMax,
                                     float& tnear,
                                     float& tfar )
{

    // compute intersection of ray with all six bbox planes
    const float3& invR = recp( dir );
    const float3& tbot = invR * ( boxMin - origin );
    const float3& ttop = invR * ( boxMax - origin );

    // re-order intersections to find smallest and largest on each axis
    const float3& tmin = fminf( ttop, tbot );
    const float3& tmax = fmaxf( ttop, tbot );

    // find the largest tmin and the smallest tmax
    const float largestTmin = fmaxf( fmaxf( tmin.x, tmin.y ), fmaxf( tmin.x, tmin.z ));
    const float smallestTmax = fminf( fminf( tmax.x, tmax.y ), fminf( tmax.x, tmax.z ));

    tnear = largestTmin;
    tfar = smallestTmax;

    return smallestTmax >= largestTmin;
}

inline __device__ float4 composite( const float4& src, const float4& dst, const float alphaCorrection )
{
    // The alpha correction function behaves badly around maximum alpha
    const float corr = 1.0f - fminf( src.w, 1.0f - 1.0f / 256.0 );
    const float alpha = 1.0f - pow( corr , alphaCorrection );
    const float3& xyz = make_float3( dst ) + make_float3( src ) * alpha * ( 1.0 - dst.w );
    const float dstw = dst.w + alpha * ( 1.0f - dst.w );
    return {  xyz.x, xyz.y, xyz.z, dstw };
}
}
}


#endif // _CommonMath_Cuh_

