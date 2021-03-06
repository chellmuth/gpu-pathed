#pragma once

#include <curand_kernel.h>

#include "hit_record.h"
#include "materials/bsdf_sample.h"
#include "materials/params.h"
#include "optics/fresnel.h"
#include "optics/snell.h"
#include "renderers/random.h"
#include "tangent_frame.h"
#include "util.h"
#include "core/vec3.h"

namespace rays {

struct Glass {
public:
    __host__ __device__ Glass() {}
    __host__ __device__ Glass(float ior) : m_ior(ior) {}

    Glass(const MaterialParams &params) : m_ior(params.getIOR()) {}

    __device__ Vec3 getEmit(const HitRecord &hit) const { return Vec3(0.f); }
    __host__ __device__ Vec3 getEmit() const { return Vec3(0.f); }

    __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const {
        return Vec3(0.f);
    }

    __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const {
        return 0.f;
    }

    __device__ BSDFSample sample(const Vec3 &wo, curandState &randState) const {
        return sample(wo, curand_uniform(&randState));
    }

    __device__ BSDFSample sample(const Vec3 &wo, unsigned int &seed) const {
        const float xi1 = rnd(seed);
        return sample(wo, xi1);
    }

    __device__ BSDFSample sample(const Vec3 &wo, float xi1) const {
        float etaIncident = 1.f;
        float etaTransmitted = m_ior;

        if (wo.z() < 0.f) {
            const float temp = etaIncident;
            etaIncident = etaTransmitted;
            etaTransmitted = temp;
        }

        Vec3 wi(0.f);
        const bool doesRefract = Snell::refract(
            wo,
            &wi,
            etaIncident,
            etaTransmitted
        );

        const float fresnelReflectance = Fresnel::dielectricReflectance(
            TangentFrame::absCosTheta(wo),
            etaIncident,
            etaTransmitted
        );

        if (xi1 < fresnelReflectance) {
            wi = wo.reflect(Vec3(0.f, 0.f, 1.f));

            const float cosTheta = TangentFrame::absCosTheta(wi);
            const Vec3 throughput = cosTheta == 0.f
                ? 0.f
                : Vec3(fresnelReflectance / cosTheta)
            ;
            const BSDFSample sample = {
                .wiLocal = wi,
                .pdf = fresnelReflectance,
                .f = throughput,
                .isDelta = true
            };

            return sample;
        } else {
            const float fresnelTransmittance = 1.f - fresnelReflectance;

            const float cosTheta = TangentFrame::absCosTheta(wi);
            const Vec3 throughput = cosTheta == 0.f
                ? 0.f
                : Vec3(fresnelTransmittance / cosTheta)
            ;

            // PBRT page 961 "Non-symmetry Due to Refraction"
            // Always incident / transmitted because we swap at top of
            // function if we're going inside-out
            const float nonSymmetricEtaCorrection = util::square(
                etaIncident / etaTransmitted
            );

            const BSDFSample sample = {
                .wiLocal = wi,
                .pdf = fresnelTransmittance,
                .f = throughput * nonSymmetricEtaCorrection,
                .isDelta = true
            };

            return sample;
        }
    }

    __device__ bool isDelta() const { return true; }

    float getIOR() const { return m_ior; }
    void setIOR(float ior) { m_ior = ior; }

private:
    float m_ior = 1.f;
};

}
