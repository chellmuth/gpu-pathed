#pragma once

#include "materials/types.h"
#include "vec3.h"

namespace rays {

enum class MaterialParam {
    Albedo,
    Emit,
    IOR
};

class MaterialParams {
public:
    virtual MaterialType getMaterialType() const = 0;

    virtual Vec3 getAlbedo() const { return Vec3(0.f); }
    virtual Vec3 getEmit() const { return Vec3(0.f); }
    virtual float getIOR() const { return 0.f; }
};

class LambertianParams : public MaterialParams {
public:
    LambertianParams(const Vec3 &albedo, const Vec3 &emit)
        : m_albedo(albedo),
          m_emit(emit)
    {}

    MaterialType getMaterialType() const override {
        return MaterialType::Lambertian;
    }

    Vec3 getAlbedo() const override {
        return m_albedo;
    }

    Vec3 getEmit() const override {
        return m_emit;
    }

private:
    Vec3 m_albedo;
    Vec3 m_emit;
};

class MirrorParams : public MaterialParams {
    MaterialType getMaterialType() const override {
        return MaterialType::Mirror;
    }
};

class GlassParams : public MaterialParams {
public:
    GlassParams(float ior) : m_ior(ior) {}

    MaterialType getMaterialType() const override {
        return MaterialType::Glass;
    }

    float getIOR() const override {
        return m_ior;
    }

private:
    float m_ior;
};

}
