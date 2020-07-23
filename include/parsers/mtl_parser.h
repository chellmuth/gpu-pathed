#pragma once

#include <fstream>
#include <map>
#include <queue>
#include <string>
#include <vector>

#include "parsers/types.h"

namespace rays {

struct MtlResult {
    std::map<std::string, int> indexLookup;
    std::vector<Mtl> mtls;
};

class MtlParser {
public:
    MtlParser(const std::string &mtlFilename);
    MtlResult parse();

private:
    void parseLine(std::string &line);
    void processNewMaterial(std::queue<std::string> &arguments);
    void processDiffuse(std::queue<std::string> &arguments);
    void processEmit(std::queue<std::string> &arguments);

    std::ifstream m_mtlFile;

    std::string m_currentMtlName;
    std::map<std::string, Mtl> m_mtlLookup;
};

}
