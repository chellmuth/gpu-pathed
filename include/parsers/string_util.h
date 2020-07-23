#pragma once

#include <string>
#include <queue>

namespace rays { namespace StringUtil {

std::queue<std::string> tokenize(const std::string &line);
std::string lTrim(const std::string &token);

} }
