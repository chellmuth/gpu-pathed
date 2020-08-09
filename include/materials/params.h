#pragma once

#include "vec3.h"

namespace rays {

enum class MaterialParam {
    Albedo,
    Emit,
    IOR
};

class MaterialParams {
public:
};

class LambertianParams : public MaterialParams {
public:
    LambertianParams(const Vec3 &albedo, const Vec3 &emit)
        : m_albedo(albedo),
          m_emit(emit)
    {}

private:
    Vec3 m_albedo;
    Vec3 m_emit;
};

class MirrorParams : public MaterialParams {
};

class GlassParams : public MaterialParams {
public:
    GlassParams(float ior) : m_ior(ior) {}

private:
    float m_ior;
};

}
