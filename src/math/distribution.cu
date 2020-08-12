#include "math/distribution.h"

#include "macro_helper.h"

#define checkCUDA(result) { gpuAssert((result), __FILE__, __LINE__); }

namespace rays {

Distribution DistributionParams::createDistribution() const
{
    float totalMass = 0.f;

    for (float value : m_values) {
        totalMass += value;
    }

    std::vector<float> cmf(m_values.size());
    for (int i = 0; i < cmf.size(); i++) {
        cmf[i] = m_values[i] / totalMass;
        if (i > 0) {
            cmf[i] += cmf[i - 1];
        }
    }

    float *d_cmf;
    const size_t cmfSize = m_values.size() * sizeof(float);
    checkCUDA(cudaMalloc((void **)&d_cmf, cmfSize));
    checkCUDA(cudaMemcpy(
        d_cmf,
        cmf.data(),
        cmfSize,
        cudaMemcpyHostToDevice
    ));

    return Distribution(d_cmf, m_values.size(), totalMass);
}

}
