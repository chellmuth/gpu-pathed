#include "parsers/string_util.h"

namespace rays {

std::string lTrim(const std::string &token)
{
    std::string::size_type firstContentIndex = token.find_first_not_of(" \t");
    if (firstContentIndex == 0) {
        return std::string(token);
    } else if (firstContentIndex == std::string::npos) {
        return "";
    }

    return token.substr(firstContentIndex);
}

}
