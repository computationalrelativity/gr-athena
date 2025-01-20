// C++ standard headers
#include <algorithm>
#include <sstream>

// Athena++ headers
#include "utils.hpp"

void tokenize(const std::string & to_tok,
              char token,
              std::vector<std::string> & vs)
{
  std::istringstream iss(to_tok);
  std::string s;

  while (std::getline(iss, s, token))
  {
    vs.push_back(s);
  }
}

long count_char(const std::string & source,
                const char to_count)
{
  return std::count(source.begin(), source.end(), to_count);
}

//
// :D
//