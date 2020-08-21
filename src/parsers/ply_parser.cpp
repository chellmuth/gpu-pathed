#include "parsers/ply_parser.h"

#include <assert.h>
#include <fstream>
#include <regex>

namespace rays {

PLYResult PLYParser::parse()
{
    std::ifstream plyFile(m_plyFilename);

    // header begin
    std::string header;
    std::getline(plyFile, header);
    assert(header == "ply");

    // format
    std::string format;
    std::getline(plyFile, format);
    assert(format == "format binary_little_endian 1.0");

    // first element (vertex)
    std::string vertexElement;
    std::getline(plyFile, vertexElement);

    static std::regex vertexElementExpression("element vertex (\\d+)");
    std::smatch vertexElementMatch;
    std::regex_match(vertexElement, vertexElementMatch, vertexElementExpression);
    assert(!vertexElementMatch.empty());

    const int vertexCount = std::stoi(vertexElementMatch[1]);

    // vertex properties
    std::string propertyX;
    std::string propertyY;
    std::string propertyZ;

    std::getline(plyFile, propertyX);
    std::getline(plyFile, propertyY);
    std::getline(plyFile, propertyZ);

    assert(propertyX == "property float x");
    assert(propertyY == "property float y");
    assert(propertyZ == "property float z");

    std::string faceElement;
    std::getline(plyFile, faceElement);

    // second element (face)
    static std::regex faceElementExpression("element face (\\d+)");
    std::smatch faceElementMatch;
    std::regex_match(faceElement, faceElementMatch, faceElementExpression);
    assert(!faceElementMatch.empty());

    const int faceCount = std::stoi(faceElementMatch[1]);

    // face property
    std::string propertyVertexIndices;
    std::getline(plyFile, propertyVertexIndices);
    assert(
        propertyVertexIndices == "property list uint8 int vertex_indices" \
        || propertyVertexIndices == "property list uchar int vertex_indices"
    );

    // header end
    std::string endHeader;
    std::getline(plyFile, endHeader);
    assert(endHeader == "end_header");


    std::vector<Vertex> vertices;
    for (int i = 0; i < vertexCount; i++) {
        float point[3];
        plyFile.read((char *)&point, 4 * 3);

        const Vertex vertex(point[0], point[1], point[2]);
        vertices.push_back(vertex);
    }

    std::vector<Face> faces;
    for (int i = 0; i < faceCount; i++) {
        uint8_t faceSize;
        plyFile.read((char *)&faceSize, 1);

        assert(faceSize == 3);

        int index[faceSize];
        plyFile.read((char *)&index, faceSize * 4);

        for (int j = 0; j < faceSize; j ++) {
            assert(index[j] < vertexCount);
        }

        const Vertex v0 = vertices[index[0]];
        const Vertex v1 = vertices[index[1]];
        const Vertex v2 = vertices[index[2]];

        const Vertex e1 = v1 - v0;
        const Vertex e2 = v2 - v0;
        const Vertex normal = e1.cross(e2).normalized();

        const Face face(v0, v1, v2, normal, normal, normal);

        faces.push_back(face);
    }

    return PLYResult{
        vertices,
        faces
    };
}

}
