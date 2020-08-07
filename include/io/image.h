#pragma once

#include <vector>
#include <string>

namespace rays { namespace Image {

void save(
    int width,
    int height,
    const std::vector<float> &radiances,
    const std::string &filename
);

} }
