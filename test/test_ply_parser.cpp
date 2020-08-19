#include <fstream>
#include <iostream>

#include <catch2/catch.hpp>

#include <parsers/ply_parser.h>
#include <parsers/types.h>

using namespace rays;

TEST_CASE("parse an example ply", "[ply]") {
    PLYParser parser("../test/ply_example.ply");

    const PLYResult result = parser.parse();

    REQUIRE(result.vertices.size() == 8);
    REQUIRE(result.faces.size() == 4);

    const Vertex testVertex(-10, -4.14615, -10);
    REQUIRE(result.vertices[0] == testVertex);
    REQUIRE(result.faces[0].v0 == testVertex);
}
