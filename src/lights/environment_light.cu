#include "lights/environment_light.h"

#include <iostream>

#include "tinyexr.h"

#include "math/distribution.h"
#include "macro_helper.h"

#define checkCUDA(result) { gpuAssert((result), __FILE__, __LINE__); }

namespace rays {

EnvironmentLight EnvironmentLightParams::createEnvironmentLight() const {
    float *data;
    int width, height;

    const char *error = nullptr;
    const int code = LoadEXR(&data, &width, &height, m_filename.c_str(), &error);
    if (code == TINYEXR_SUCCESS) {
        std::cout << "Loaded environment light { "
                  << " width: " << width
                  << " height: " << height
                  << " }" << std::endl;
    } else {
        fprintf(stderr, "ENVIRONMENT MAP ERROR: %s\n", error);
        FreeEXRErrorMessage(error);
    }

    float *d_data;
    const size_t dataSize = width * height * 4 * sizeof(float);
    checkCUDA(cudaMalloc((void **)&d_data, dataSize));
    checkCUDA(cudaMemcpy(
        d_data,
        data,
        dataSize,
        cudaMemcpyHostToDevice
    ));


    PhiThetaDistributionBuilder distributionBuilder(data, width, height);

    free(data);

    const EnvironmentLight environmentLight(d_data, width, height);
    return environmentLight;
}

}
