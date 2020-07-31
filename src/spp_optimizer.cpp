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
    if (m_reset) {
        m_reset = false;
        return 1;
    }

    return m_nextEstimate;
}

void SppOptimizer::reset()
{
    m_reset = true;
}

}
