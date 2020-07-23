#include "parsers/obj_parser.h"

#include <map>
#include <regex>

#include "parsers/string_util.h"

namespace rays {

using string = std::string;

ObjParser::ObjParser(std::ifstream &objFile)
    : m_objFile(objFile),
      m_currentMaterialIndex(0)
{}

ObjResult ObjParser::parse()
{
    string line;
    while(std::getline(m_objFile, line)) {
        parseLine(line);
    }

    ObjResult result;
    result.vertices = m_vertices;
    result.faces = m_faces;
    result.materialIndices = m_materialIndices;

    std::vector<Mtl> materials = {
        Mtl(0.725, 0.71, 0.68), // floor
        Mtl(0.725, 0.71, 0.68), // ceiling
        Mtl(0.725, 0.71, 0.68), // back wall
        Mtl(0.14, 0.45, 0.091), // right wall
        Mtl(0.63, 0.065, 0.05), // left wall
        Mtl(0.725, 0.71, 0.68), // short box
        Mtl(0.725, 0.71, 0.68), // tall box
        Mtl(0.78, 0.78, 0.78, 17.f, 12.f, 4.f) // light
    };

    result.materials = materials;

    return result;
}

void ObjParser::parseLine(string &line)
{
    if (line.empty()) { return; }

    string::size_type spaceIndex = line.find_first_of(" \t");
    if (spaceIndex == string::npos) { return; }

    string command = line.substr(0, spaceIndex);
    if (command[0] == '#') { return; }

    string rest = lTrim(line.substr(spaceIndex + 1));

    if (command == "v") {
        processVertex(rest);
    } else if (command == "f") {
        processFace(rest);
    } else if (command == "usemtl") {
        std::map<std::string, int> materialLookup = {
            {"floor", 0},
            {"ceiling", 1},
            {"backWall", 2},
            {"rightWall", 3},
            {"leftWall", 4},
            {"shortBox", 5},
            {"tallBox", 6},
            {"light", 7},
        };

        m_currentMaterialIndex = materialLookup[rest];
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

void ObjParser::processFace(string &faceArgs)
{
    if (processDoubleFaceGeometryOnly(faceArgs)) { return; }
}

bool ObjParser::processDoubleFaceGeometryOnly(std::string &faceArgs)
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

void ObjParser::processTriangle(int vertexIndex0, int vertexIndex1, int vertexIndex2)
{
    correctIndices(m_vertices, &vertexIndex0, &vertexIndex1, &vertexIndex2);

    Vertex v0(m_vertices[vertexIndex0]);
    Vertex v1(m_vertices[vertexIndex1]);
    Vertex v2(m_vertices[vertexIndex2]);

    Face face(v0, v1, v2);
    m_faces.push_back(face);
    m_materialIndices.push_back(m_currentMaterialIndex);
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

}
