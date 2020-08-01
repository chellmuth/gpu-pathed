#include "parsers/obj_parser.h"

#include <fstream>
#include <regex>

#include <boost/filesystem.hpp>

#include "parsers/mtl_parser.h"
#include "parsers/string_util.h"

namespace rays {

using string = std::string;

ObjParser::ObjParser(std::string &objFilename)
    : m_objFilename(objFilename),
      m_currentMtlIndex(0)
{}

ObjResult ObjParser::parse()
{
    std::ifstream objFile(m_objFilename);

    string line;
    while(std::getline(objFile, line)) {
        parseLine(line);
    }

    ObjResult result;
    result.vertices = m_vertices;
    result.faces = m_faces;
    result.mtls = m_mtls;
    result.mtlIndices = m_mtlIndices;

    return result;
}

void ObjParser::parseLine(string &line)
{
    if (line.empty()) { return; }

    string::size_type spaceIndex = line.find_first_of(" \t");
    if (spaceIndex == string::npos) { return; }

    string command = line.substr(0, spaceIndex);
    if (command[0] == '#') { return; }

    string rest = StringUtil::lTrim(line.substr(spaceIndex + 1));

    if (command == "v") {
        processVertex(rest);
    } else if (command == "vn") {
        processNormal(rest);
    } else if (command == "f") {
        processFace(rest);
    } else if (command == "mtllib") {
        processMaterialLibrary(rest);
    } else if (command == "usemtl") {
        m_currentMtlIndex = m_mtlIndexLookup[rest];
    }
}

void ObjParser::processVertex(string &vertexArgs)
{
    string::size_type index = 0;
    string rest = vertexArgs;

    float x = std::stof(rest, &index);

    rest = rest.substr(index);
    float y = std::stof(rest, &index);

    rest = rest.substr(index);
    float z = std::stof(rest, &index);

    m_vertices.push_back(Vertex(x, y, z));
}

void ObjParser::processNormal(string &normalArgs)
{
    string::size_type index = 0;
    string rest = normalArgs;

    float x = std::stof(rest, &index);

    rest = rest.substr(index);
    float y = std::stof(rest, &index);

    rest = rest.substr(index);
    float z = std::stof(rest, &index);

    m_normals.push_back(Vertex(x, y, z));
}

void ObjParser::processFace(string &faceArgs)
{
    if (processDoubleFaceGeometryOnly(faceArgs)) { return; }
    if (processSingleFaceTriplets(faceArgs)) { return; }
}

bool ObjParser::processDoubleFaceGeometryOnly(string &faceArgs)
{
    static const std::regex expression("(-?\\d+)\\s+(-?\\d+)\\s+(-?\\d+)\\s+(-?\\d+)\\s*");
    std::smatch match;
    std::regex_match(faceArgs, match, expression);

    if (match.empty()) {
        return false;
    }

    int index0 = std::stoi(match[1]);
    int index1 = std::stoi(match[2]);
    int index2 = std::stoi(match[3]);
    int index3 = std::stoi(match[4]);

    processTriangle(index0, index1, index2);
    processTriangle(index0, index2, index3);

    return true;
}

bool ObjParser::processSingleFaceTriplets(std::string &faceArgs)
{
    static std::regex expression("(-?\\d+)/(-?\\d+)/(-?\\d+) (-?\\d+)/(-?\\d+)/(-?\\d+) (-?\\d+)/(-?\\d+)/(-?\\d+)\\s*");
    std::smatch match;
    std::regex_match (faceArgs, match, expression);

    if (match.empty()) {
        return false;
    }

    int vertexIndex0 = std::stoi(match[1]);
    int vertexIndex1 = std::stoi(match[4]);
    int vertexIndex2 = std::stoi(match[7]);

    int UVIndex0 = std::stoi(match[2]);
    int UVIndex1 = std::stoi(match[5]);
    int UVIndex2 = std::stoi(match[8]);

    int normalIndex0 = std::stoi(match[3]);
    int normalIndex1 = std::stoi(match[6]);
    int normalIndex2 = std::stoi(match[9]);

    processTriangle(
        vertexIndex0, vertexIndex1, vertexIndex2,
        normalIndex0, normalIndex1, normalIndex2,
        UVIndex0, UVIndex1, UVIndex2
    );

    return true;
}

void ObjParser::processTriangle(int vertexIndex0, int vertexIndex1, int vertexIndex2)
{
    correctIndices(m_vertices, &vertexIndex0, &vertexIndex1, &vertexIndex2);

    const Vertex v0(m_vertices[vertexIndex0]);
    const Vertex v1(m_vertices[vertexIndex1]);
    const Vertex v2(m_vertices[vertexIndex2]);

    const Vertex e1 = v1 - v0;
    const Vertex e2 = v2 - v0;

    const Vertex normal = e1.cross(e2).normalized();

    const Face face(v0, v1, v2, normal, normal, normal);
    m_faces.push_back(face);
    m_mtlIndices.push_back(m_currentMtlIndex);
}

void ObjParser::processTriangle(
    int vertexIndex0, int vertexIndex1, int vertexIndex2,
    int normalIndex0, int normalIndex1, int normalIndex2
) {
    correctIndices(m_vertices, &vertexIndex0, &vertexIndex1, &vertexIndex2);
    correctIndices(m_normals, &normalIndex0, &normalIndex1, &normalIndex2);

    const Vertex v0(m_vertices[vertexIndex0]);
    const Vertex v1(m_vertices[vertexIndex1]);
    const Vertex v2(m_vertices[vertexIndex2]);

    const Vertex n0(m_normals[normalIndex0]);
    const Vertex n1(m_normals[normalIndex1]);
    const Vertex n2(m_normals[normalIndex2]);

    const Face face(v0, v1, v2, n0, n1, n2);
    m_faces.push_back(face);
    m_mtlIndices.push_back(m_currentMtlIndex);
}

void ObjParser::processTriangle(
    int vertexIndex0, int vertexIndex1, int vertexIndex2,
    int normalIndex0, int normalIndex1, int normalIndex2,
    int UVIndex0, int UVIndex1, int UVIndex2
) {
    // TODO: Handle UVs
    processTriangle(
        vertexIndex0,
        vertexIndex1,
        vertexIndex2,
        normalIndex0,
        normalIndex1,
        normalIndex2
    );
}

template <class T>
void ObjParser::correctIndex(const std::vector<T> &indices, int *index)
{
    if (*index < 0) {
        *index += indices.size();
    } else {
        *index -= 1;
    }
}

template <class T>
void ObjParser::correctIndices(
    const std::vector<T> &indices,
    int *index0,
    int *index1,
    int *index2
) {
    correctIndex(indices, index0);
    correctIndex(indices, index1);
    correctIndex(indices, index2);
}

void ObjParser::processMaterialLibrary(string &libraryArgs)
{
    using path = boost::filesystem::path;
    const path objPath(m_objFilename);

    const string filename = libraryArgs;
    const path mtlPath = objPath.parent_path() / filename;

    MtlParser mtlParser(mtlPath.native());
    MtlResult result = mtlParser.parse();

    m_mtls = result.mtls;
    m_mtlIndexLookup = result.indexLookup;
}

}
