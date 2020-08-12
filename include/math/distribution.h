#pragma once

#include <assert.h>
#include <vector>

namespace rays {

class Distribution {
public:
    Distribution(float *cmf, size_t size, float totalMass)
        : m_cmf(cmf), m_size(size), m_totalMass(totalMass)
    {}

    __device__ size_t sample(float *pmf, float xi) const {
        if (m_totalMass == 0.f) { assert(0); }

        for (size_t i = 0; i < m_size; i++) {
            if (xi <= m_cmf[i]) {
                *pmf = this->pmf(i);

                return i;
            }
        }

        assert(0);
        return 0;
    }

    __device__ float pmf(int index) const {
        if (m_totalMass == 0.f) { return 0.f; }

        const float cmf = m_cmf[index];
        if (index == 0) { return cmf; }

        return cmf - m_cmf[index - 1];
    }

    __device__ float getTotalMass() const { return m_totalMass; }
    __device__ int getSize() const { return m_size; }

private:
    float *m_cmf;
    size_t m_size;

    float m_totalMass;
};

class DistributionParams {
public:
    DistributionParams(const std::vector<float> &values)
        : m_values(values)
    {}

    Distribution createDistribution() const;

private:
    std::vector<float> m_values;
};

}
