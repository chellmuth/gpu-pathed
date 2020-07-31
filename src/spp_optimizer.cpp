#include "spp_optimizer.h"

#include <algorithm>
#include <cmath>

namespace rays {

static constexpr float budgetMilliseconds = 80.f;

void SppOptimizer::track(int spp, float milliseconds)
{
    const float samplesPerMs = spp / milliseconds;
    m_nextEstimate = std::max(1, (int)std::floor(budgetMilliseconds * samplesPerMs));
}

int SppOptimizer::estimateSpp()
{
    return m_nextEstimate;
}

}
