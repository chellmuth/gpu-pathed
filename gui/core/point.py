import math

class Point3:
    @classmethod
    def from_vec3(cls, vec3):
        return cls(
            vec3.x(),
            vec3.y(),
            vec3.z()
        )

    @classmethod
    def cross(cls, v1, v2):
        return cls(
            v1.y * v2.z - v1.z * v2.y,
            -(v1.x * v2.z - v1.z * v2.x),
            v1.x * v2.y - v1.y * v2.x
        )

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def normalized(self):
        length = self.length()

        return Point3(
            self.x / length,
            self.y / length,
            self.z / length
        )

    def __add__(self, other):
        return Point3(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    def __sub__(self, other):
        return Point3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )

    def __mul__(self, scalar):
        return Point3(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar
        )

    def length(self):
        return math.sqrt(
            self.x ** 2 + self.y ** 2 + self.z ** 2
        )

    def __repr__(self):
        return f"Point3: ({self.x}, {self.y} {self.z})"

class Point2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Point2(
            self.x - other.x,
            self.y - other.y
        )

    def __truediv__(self, scalar):
        return Point2(
            self.x / scalar,
            self.y / scalar
        )

    def __mul__(self, scalar):
        return Point2(
            self.x * scalar,
            self.y * scalar
        )

    def __repr__(self):
        return f"Point2: ({self.x}, {self.y})"
