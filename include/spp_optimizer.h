#pragma once

namespace rays {

class SppOptimizer {
public:
    void track(int spp, float milliseconds);
    int estimateSpp();
    void reset();

private:
    int m_nextEstimate = 1;
    bool m_reset = false;
};

}
