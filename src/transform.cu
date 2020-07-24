#include "transform.h"

#include <iostream>

namespace rays {

Transform lookAt(
    const Vec3 &source,
    const Vec3 &target,
    const Vec3 &up
) {
    const Vec3 direction = normalized(source - target);

    if (direction == up) {
        std::cerr << "Look direction cannot equal up vector - quitting!" << std::endl;
        exit(1);
    }

    const Vec3 xAxis = normalized(cross(normalized(up), direction));
    const Vec3 yAxis = normalized(cross(direction, xAxis));

    float matrix[4][4] {
        { xAxis.x(), yAxis.x(), direction.x(), source.x() },
        { xAxis.y(), yAxis.y(), direction.y(), source.y() },
        { xAxis.z(), yAxis.z(), direction.z(), source.z() },
        { 0.f, 0.f, 0.f, 1.f }
    };

    return Transform(matrix);
}

}
