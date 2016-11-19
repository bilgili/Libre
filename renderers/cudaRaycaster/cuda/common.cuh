#ifndef _Common_Cuh_
#define _Common_Cuh_

#include <livre/core/mathTypes.h>
#include <vector>

namespace livre
{
namespace cuda
{
/** Cuda representation of the render nodes */
struct NodeData
{
    Vector3f textureMin;
    Vector3f textureSize;
    Vector3f aabbMin;
    Vector3f aabbSize;
};

typedef std::vector< NodeData > NodeDatas;

/** View information for rendering */
struct ViewData
{
    Vector3f eyePosition;
    Vector4ui glViewport;
    Matrix4f invProjMatrix;
    Matrix4f modelViewMatrix;
    Matrix4f invViewMatrix;
    Vector3f aabbMin;
    Vector3f aabbMax;
    float nearPlane;
};

/** Render information */
struct RenderData
{
    unsigned int samplesPerRay;
    unsigned int samplesPerPixel;
    unsigned int maxSamplesPerRay;
    unsigned int datatype;
    Vector2f dataSourceRange;
};
}
}


#endif // _Common_Cuh_


