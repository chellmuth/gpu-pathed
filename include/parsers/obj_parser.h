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
    void processNormal(std::string &normalArgs);
    void processFace(std::string &faceArgs);
    void processMaterialLibrary(std::string &libraryArgs);

    bool processSingleFaceTriplets(std::string &faceArgs);
    bool processDoubleFaceGeometryOnly(std::string &faceArgs);

    void processTriangle(int vertexIndex0, int vertexIndex1, int vertexIndex2);
    void processTriangle(
        int vertexIndex0, int vertexIndex1, int vertexIndex2,
        int normalIndex0, int normalIndex1, int normalIndex2
    );
    void processTriangle(
        int vertexIndex0, int vertexIndex1, int vertexIndex2,
        int normalIndex0, int normalIndex1, int normalIndex2,
        int UVIndex0, int UVIndex1, int UVIndex2
    );

    template <class T>
    void correctIndex(const std::vector<T> &indices, int *index);

    template <class T>
    void correctIndices(
        const std::vector<T> &indices,
        int *index0,
        int *index1,
        int *index2
    );

    std::string m_objFilename;

    std::vector<Vertex> m_vertices;
    std::vector<Vertex> m_normals;
    std::vector<Face> m_faces;

    int m_currentMtlIndex;
    std::vector<int> m_mtlIndices;

    std::vector<Mtl> m_mtls;
    std::map<std::string, int> m_mtlIndexLookup;

};

}
