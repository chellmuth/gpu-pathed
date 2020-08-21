#pragma once

#include <stdio.h>

#include <curand_kernel.h>

#include "core/vec3.h"
#include "materials/bsdf_sample.h"
#include "materials/params.h"
#include "math/spherical_coordinates.h"
#include "optics/fresnel.h"
#include "renderers/random.h"
#include "tangent_frame.h"
#include "util.h"


namespace rays {

class MicrofacetDistribution {
public:
    __device__ MicrofacetDistribution() {}

    MicrofacetDistribution(float alpha)
        : m_alpha(alpha)
    {}

    __device__ float G(const Vec3 &wi, const Vec3 &wo, const Vec3 &wh) const {
        if (util::sign(dot(wo, wh)) != util::sign(TangentFrame::cosTheta(wo))) {
            return 0.f;
        }
        if (util::sign(dot(wi, wh)) != util::sign(TangentFrame::cosTheta(wi))) {
            return 0.f;
        }
        return G1(wi) * G1(wo);
    }

    __device__ float D(const Vec3 &wi, const Vec3 &wh) const {
        const float alpha = sampleAlpha(TangentFrame::absCosTheta(wi));
        const float alpha2 = alpha * alpha;

        const float cos2Theta = TangentFrame::cos2Theta(wh);
        const float cos4Theta = cos2Theta * cos2Theta;

        const float tan2Theta = TangentFrame::tan2Theta(wh);
        if (isinf(tan2Theta)) { return 0.f; }

        const float numerator = expf(-tan2Theta / alpha2);
        const float denominator = M_PI * alpha2 * cos4Theta;

        return numerator / denominator;
    }

    __device__ Vec3 sampleWh(const Vec3 &wo, float xi1, float xi2) const {
        const float phi = xi1 * M_PI * 2.f;

        const float alpha = sampleAlpha(TangentFrame::absCosTheta(wo));
        const float alpha2 = alpha * alpha;
        const float sqrtTerm = (-alpha2 * logf(1 - xi2));
        const float theta = atanf(sqrtf(sqrtTerm));

        const Vec3 sample = Coordinates::sphericalToCartesianZ(phi, theta); // fixme
        return sample;
    }

    __device__ float pdf(const Vec3 &wi, const Vec3 &wh) const {
        return D(wi, wh) * fabsf(TangentFrame::cosTheta(wh));
    }

private:
    __device__ float sampleAlpha(float absCosThetaI) const {
        return (1.2f - 0.2f * sqrtf(absCosThetaI)) * m_alpha;
    }

    __device__ float G1(const Vec3 &v) const {
        const float a = 1.f / (m_alpha * TangentFrame::absTanTheta(v));

        if (a >= 1.6f) { return 1.f; }

        const float numerator = 3.535f * a + 2.181f * a * a;
        const float denominator = 1.f + 2.276f * a + 2.577f * a * a;

        return numerator / denominator;
    }

    float m_alpha;
};

class Microfacet {
public:
    __device__ Microfacet() {}

    Microfacet(const MaterialParams &params)
        : m_distribution(params.getAlpha())
    {}

    __device__ Vec3 getEmit() const { return Vec3(0.f); }

    __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const {
        if (wo.z() < 0.f || wi.z() < 0.f) {
            return Vec3(0.f);
        }

        const float cosThetaO = TangentFrame::absCosTheta(wo);
        const float cosThetaI = TangentFrame::absCosTheta(wi);
        const Vec3 wh = normalized(wo + wi);

        if (cosThetaO == 0.f || cosThetaI == 0.f) { return Vec3(0.f); }
        if (wh.isZero()) { return Vec3(0.f); }

        const float cosThetaIncident = util::clamp(dot(wi, wh), 0.f, 1.f);
        const float fresnel(Fresnel::dielectricReflectance(cosThetaIncident, 1.f, 1.5f)); // fixme
        const float distribution = m_distribution.D(wi, wh);
        const float masking = m_distribution.G(wi, wo, wh);

        const Vec3 throughput = Vec3(1.f)
            * distribution
            * masking
            * fresnel
            / (4 * cosThetaI * cosThetaO);

        return throughput;
    }

    __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const {
        if (wo.z() < 0.f || wi.z() < 0.f) {
            return 0.f;
        }

        const float cosThetaO = TangentFrame::absCosTheta(wo);
        const float cosThetaI = TangentFrame::absCosTheta(wi);
        const Vec3 wh = normalized(wo + wi);

        return m_distribution.pdf(wi, wh) / (4.f * absDot(wo, wh));
    }

    __device__ BSDFSample sample(const Vec3 &wo, curandState &randState) const {
        const float xi1 = curand_uniform(&randState);
        const float xi2 = curand_uniform(&randState);
        return sample(wo, xi1, xi2);
    }

    __device__ BSDFSample sample(const Vec3 &wo, unsigned int &seed) const {
        const float xi1 = rnd(seed);
        const float xi2 = rnd(seed);
        return sample(wo, xi1, xi2);
    }

    __device__ BSDFSample sample(const Vec3 &wo, const float xi1, const float xi2) const {
        const Vec3 wh = m_distribution.sampleWh(wo, xi1, xi2);
        const Vec3 wi = wo.reflect(wh);

        return BSDFSample{
            .wiLocal = wi,
            .pdf = m_distribution.pdf(wo, wh) / (4.f * absDot(wi, wh)),
            .f = f(wo, wi),
            .isDelta = false
        };
    }

    __device__ bool isDelta() const { return false; }

private:
    MicrofacetDistribution m_distribution;
};

}
