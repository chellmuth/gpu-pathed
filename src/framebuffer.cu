#include "framebuffer.h"

#include "core/color.h"

namespace rays {

__global__ static void kernel(
    uchar4 *fb,
    Vec3 *passRadiances,
    Vec3 *radiances,
    int width, int height,
    int passSamples,
    int previousSamples
) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    if ((row >= height) || (col >= width)) { return; }

    const int pixelIndex = row * width + col;

    const Vec3 pass = passRadiances[pixelIndex];

    if (previousSamples > 0) {
        const Vec3 current = radiances[pixelIndex];
        const int totalSamples = previousSamples + passSamples;

        const Vec3 next = current * previousSamples / totalSamples
            + pass * (1.f * passSamples / totalSamples);

        radiances[pixelIndex] = next;
    } else {
        radiances[pixelIndex] = pass;
    }

    const Vec3 finalRadiance = Color::toSRGB(radiances[pixelIndex]);

    fb[pixelIndex].x = max(0.f, min(1.f, finalRadiance.x())) * 255;
    fb[pixelIndex].y = max(0.f, min(1.f, finalRadiance.y())) * 255;
    fb[pixelIndex].z = max(0.f, min(1.f, finalRadiance.z())) * 255;
    fb[pixelIndex].w = 255;
}

void updateFramebuffer(
    uchar4 *d_fb,
    Vec3 *d_passRadiances,
    Vec3 *d_radiances,
    int passSamples,
    int previousSamples,
    int width, int height,
    CUstream stream
) {
    const int threadX = 16;
    const int threadY = 16;

    const dim3 blocks(width / threadX + 1, height / threadY + 1);
    const dim3 threads(threadX, threadY);

    kernel<<<blocks, threads, 0, stream>>>(
        d_fb,
        d_passRadiances,
        d_radiances,
        width,
        height,
        passSamples,
        previousSamples
    );
}

}
