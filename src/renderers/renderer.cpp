#include "renderers/renderer.h"

#include "macro_helper.h"
#include "core/vec3.h"

#define checkCudaErrors(result) { gpuAssert((result), __FILE__, __LINE__); }

namespace rays {

void Renderer::init(int width, int height, const Scene &scene)
{
    m_width = width;
    m_height = height;

    const int pixelCount = m_width * m_height;

    checkCudaErrors(cudaMalloc((void **)&dev_radiances, pixelCount * sizeof(Vec3)));
}

std::vector<float> Renderer::getRadianceBuffer() const
{
    std::vector<float> radiances(3 * m_width * m_height, 0.f);
    Vec3 *copied = (Vec3 *)malloc(sizeof(Vec3) * m_width * m_height);
    checkCudaErrors(cudaMemcpy(
        copied,
        dev_radiances,
        sizeof(Vec3) * m_width * m_height,
        cudaMemcpyDeviceToHost
    ));

    for (int row = 0; row < m_height; row++) {
        for (int col = 0; col < m_width; col++) {
            const int pixelIndex = row * m_width + col;
            const Vec3 vector = copied[pixelIndex];
            radiances[pixelIndex * 3 + 0] = vector.x();
            radiances[pixelIndex * 3 + 1] = vector.y();
            radiances[pixelIndex * 3 + 2] = vector.z();
        }
    }

    return radiances;
}

}
