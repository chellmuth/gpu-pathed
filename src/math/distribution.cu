#include "math/distribution.h"

#include "macro_helper.h"

#define checkCUDA(result) { gpuAssert((result), __FILE__, __LINE__); }

namespace rays {

float DistributionParams::getTotalMass() const {
    float totalMass = 0.f;

    for (float value : m_values) {
        totalMass += value;
    }

    return totalMass;
}

Distribution DistributionParams::createDistribution() const
{
    const float totalMass = getTotalMass();

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

Distribution2D Distribution2DBuilder::buildDistribution2D() const
{
    std::vector<Distribution> xDistributions;
    xDistributions.reserve(m_height);

    std::vector<float> rowSums(m_height);
    for (int row = 0; row < m_height; row++) {
        const size_t rowIndex = row * m_width;
        std::vector<float> colValues(m_width);

        for (int col = 0; col < m_width; col++) {
            colValues[col] = m_data[rowIndex + col];
        }

        DistributionParams xParams(colValues);
        xDistributions.push_back(xParams.createDistribution());
        rowSums[row] = xParams.getTotalMass();
    }

    DistributionParams yParams(rowSums);
    Distribution yDistribution = yParams.createDistribution();

    Distribution *d_xDistributions;
    const size_t xDistributionSize = m_height * sizeof(Distribution);
    checkCUDA(cudaMalloc((void **)&d_xDistributions, xDistributionSize));
    checkCUDA(cudaMemcpy(
        d_xDistributions,
        xDistributions.data(),
        xDistributionSize,
        cudaMemcpyHostToDevice
    ));

    const Distribution2D distribution2D(yDistribution, d_xDistributions, m_width, m_height);
    return distribution2D;
}

PhiThetaDistribution PhiThetaDistributionBuilder::buildPhiThetaDistribution() const
{
    Distribution2DBuilder builder(m_data, m_width, m_height);
    Distribution2D distribution2D = builder.buildDistribution2D();
    return PhiThetaDistribution(distribution2D);
}

}
