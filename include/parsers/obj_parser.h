#pragma once

#include <map>
#include <string>
#include <vector>

#include "parsers/types.h"

namespace rays {

struct ObjResult {
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
    std::vector<Mtl> mtls;
    std::vector<int> mtlIndices;
};

class ObjParser {
public:
    ObjParser(std::string &objFilename);

    ObjResult parse();

private:
    void parseLine(std::string &line);

    void processVertex(std::string &vertexArgs);
    void processFace(std::string &faceArgs);
    void processMaterialLibrary(std::string &libraryArgs);

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

    std::string &m_objFilename;

    std::vector<Vertex> m_vertices;
    std::vector<Face> m_faces;

    int m_currentMtlIndex;
    std::vector<int> m_mtlIndices;

    std::vector<Mtl> m_mtls;
    std::map<std::string, int> m_mtlIndexLookup;

};

}
