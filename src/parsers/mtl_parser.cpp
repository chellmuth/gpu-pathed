#include "parsers/mtl_parser.h"

#include <assert.h>
#include <iostream>
#include <utility>

#include "parsers/string_util.h"

namespace rays {

MtlParser::MtlParser(const std::string &mtlFilename)
    : m_mtlFile(mtlFilename)
{}

MtlResult MtlParser::parse()
{
    std::string line;
    while(std::getline(m_mtlFile, line)) {
        parseLine(line);
    }

    MtlResult result;
    int index = 0;
    for (const std::pair<std::string, Mtl> &item : m_mtlLookup) {
        result.mtls.push_back(item.second);
        result.indexLookup[item.first] = index++;
    }

    return result;
}

void MtlParser::parseLine(std::string &line)
{
    std::queue<std::string> tokens = StringUtil::tokenize(line);
    if (tokens.empty()) { return; }

    std::string command = tokens.front();
    tokens.pop();

    if (command[0] == '#') { return; }

    if (command == "newmtl") {
        processNewMaterial(tokens);
    } else if (command == "Kd") {
        processDiffuse(tokens);
    } else if (command == "Ke") {
        processEmit(tokens);
    } else if (
        command == "Ka" ||
        command == "Ks" ||
        command == "Ni" ||
        command == "Ns" ||
        command == "illum"
    ) {
        // Skip
    } else {
        std::cerr << "Unknown mtl command: " << command << std::endl;
    }
}

void MtlParser::processNewMaterial(std::queue<std::string> &arguments)
{
    assert(arguments.size() >= 1);

    std::string name = arguments.front();
    m_currentMtlName = name;

    Mtl m;
    m_mtlLookup[m_currentMtlName] = m;
}

void MtlParser::processDiffuse(std::queue<std::string> &arguments)
{
    assert(arguments.size() >= 3);

    float r = std::stof(arguments.front());
    arguments.pop();

    float g = std::stof(arguments.front());
    arguments.pop();

    float b = std::stof(arguments.front());
    arguments.pop();

    Mtl &currentMtl = m_mtlLookup[m_currentMtlName];
    currentMtl.r = r;
    currentMtl.g = g;
    currentMtl.b = b;
}

void MtlParser::processEmit(std::queue<std::string> &arguments)
{
    assert(arguments.size() >= 3);

    float r = std::stof(arguments.front());
    arguments.pop();

    float g = std::stof(arguments.front());
    arguments.pop();

    float b = std::stof(arguments.front());
    arguments.pop();

    Mtl &currentMtl = m_mtlLookup[m_currentMtlName];
    currentMtl.emitR = r;
    currentMtl.emitG = g;
    currentMtl.emitB = b;
}

}
