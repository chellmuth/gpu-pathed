import math

from gui.core.point import Point3

# y is up!

def cartesian_to_spherical(cartesian):
    phi = math.atan2(cartesian.z, cartesian.x)

    if phi < 0.:
        phi += 2. * math.pi
    if phi == 2. * math.pi:
        phi = 0.

    theta = math.acos(cartesian.y)

    return phi, theta

def spherical_to_cartesian(phi, theta):
    y = math.cos(theta)

    x = math.sin(theta) * math.cos(phi)
    z = math.sin(theta) * math.sin(phi)

    return Point3(x, y, z)
