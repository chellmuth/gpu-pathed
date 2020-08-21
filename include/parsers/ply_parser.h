#pragma once

#include "parsers/types.h"

#include <string>
#include <vector>

namespace rays {

struct PLYResult {
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
};

class PLYParser {
public:
    PLYParser(const std::string &plyFilename)
        : m_plyFilename(plyFilename)
    {}

    PLYResult parse();

private:
    std::string m_plyFilename;
};

}
