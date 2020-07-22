#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "parsers/types.h"

namespace rays {

struct ObjResult {
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
};

class ObjParser {
public:
    ObjParser(std::ifstream &objFile);

    ObjResult parse();

private:
    void parseLine(std::string &line);

    void processVertex(std::string &vertexArgs);
    void processFace(std::string &faceArgs);

    bool processDoubleFaceGeometryOnly(std::string &faceArgs);
    void processTriangle(int vertexIndex0, int vertexIndex1, int vertexIndex2);

    template <class T>
    void correctIndex(const std::vector<T> &indices, int *index);

    template <class T>
    void correctIndices(
        const std::vector<T> &indices,
        int *index0,
        int *index1,
        int *index2
    );

    std::ifstream &m_objFile;

    std::vector<Vertex> m_vertices;
    std::vector<Face> m_faces;
};

}
