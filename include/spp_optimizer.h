#pragma once

namespace rays {

class SppOptimizer {
public:
    void track(int spp, float milliseconds);
    int estimateSpp();

private:
    int m_nextEstimate = 1;
};

}
