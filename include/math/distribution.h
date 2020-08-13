#pragma once

#include <assert.h>
#include <vector>

#include "core/vec3.h"
#include "math/spherical_coordinates.h"

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

    float getTotalMass() const;
    Distribution createDistribution() const;

private:
    std::vector<float> m_values;
};

class Distribution2D {
public:
    struct Sample {
        int x;
        int y;

        bool operator ==(const Sample &s) const {
            return x == s.x && y == s.y;
        }
    };

    Distribution2D(
        Distribution yDistribution,
        Distribution *xDistributions,
        int width, int height
    ) : m_yDistribution(yDistribution),
        m_xDistributions(xDistributions),
        m_width(width),
        m_height(height)
    {}

    __device__ Sample sample(float *pmf, float2 xis) const {
        float xPMF, yPMF;
        int y = m_yDistribution.sample(&yPMF, xis.x);
        int x = m_xDistributions[y].sample(&xPMF, xis.y);

        *pmf = xPMF * yPMF;
        return {x, y};
    }

    __device__ float pmf(int xIndex, int yIndex) const {
        if (m_yDistribution.getTotalMass() == 0.f) { return 0.f; }

        const float yPMF = m_yDistribution.pmf(yIndex);
        const float xPMF = m_xDistributions[yIndex].pmf(xIndex);

        return xPMF * yPMF;
    }

    __device__ int getWidth() const { return m_width; }
    __device__ int getHeight() const { return m_height; }

private:
    Distribution m_yDistribution;
    Distribution *m_xDistributions;

    int m_width;
    int m_height;
};

class Distribution2DBuilder {
public:
    Distribution2DBuilder(const float *data, int width, int height)
        : m_data(data),
          m_width(width),
          m_height(height)
    {}

    Distribution2D buildDistribution2D() const;

private:
    const float *m_data;
    int m_width;
    int m_height;
};

class PhiThetaDistribution {
public:
    PhiThetaDistribution(Distribution2D distribution2D)
        : m_distribution2D(distribution2D)
    {}

    __device__ float pdf(const Vec3 &wi) const {
        float phi, theta;
        Coordinates::cartesianToSpherical(wi, &phi, &theta);

        const float xCanonical = phi / (M_PI * 2.f);
        const float yCanonical = theta / M_PI;

        return pdf(xCanonical, yCanonical) / (sinf(theta) * M_PI * M_PI * 2.f);
    }

    __device__ float pdf(float xCanonical, float yCanonical) const {
        const int width = m_distribution2D.getWidth();
        const int height = m_distribution2D.getHeight();

        const int xStep = fminf(floorf(xCanonical * width), width - 1);
        const int yStep = fminf(floorf(yCanonical * height), height - 1);

        const float pmf = m_distribution2D.pmf(xStep, yStep);
        const float pdf = pmf * width * height;

        return pdf;
    }

private:
    Distribution2D m_distribution2D;
};

class PhiThetaDistributionBuilder {
public:
    PhiThetaDistributionBuilder(const float *data, int width, int height)
        : m_data(data),
          m_width(width),
          m_height(height)
    {}

    PhiThetaDistribution buildPhiThetaDistribution() const;

private:
    const float *m_data;
    int m_width;
    int m_height;
};

}
