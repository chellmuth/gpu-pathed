#pragma once

namespace rays { namespace MIS {

__device__ inline float balanceWeight(int n1, int n2, float pdf1, float pdf2)
{
    return (n1 * pdf1) / (n1 * pdf1 + n2 * pdf2);
}

} }
