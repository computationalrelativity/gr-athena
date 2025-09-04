//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file parameter_input.cpp
//  \brief implementation of functions in class ParameterInput
//
// PURPOSE: Member functions of this class are used to read and parse the input file.
//   Functionality is loosely modeled after FORTRAN namelist.
//
// EXAMPLE of input file in 'Athena++' format:
//   <blockname1>      # block name; must be on a line by itself
//                     # everything after a hash symbol is a comment and is ignored
//   name1=value       # each parameter name must be on a line by itself
//   name2 = value1    # whitespace around the = is optional
//                     # blank lines are OK
//   # my comment here   comment lines are OK
//   # name3 = value3    values (and blocks) that are commented out are ignored
//
//   <blockname2>      # start new block
//   name1 = value1    # note that same parameter names can appear in different blocks
//   name2 = value2    # empty lines (like following) are OK
//
//   <blockname1>      # same blockname can re-appear, although NOT recommended
//   name3 = value3    # this would be the 3rd parameter name in blockname1
//   name1 = value4    # if parameter name is repeated, previous value is overwritten!
//
// LIMITATIONS:
//   - parameter specification (name=val #comment) must all be on a single line
//
// HISTORY:
//   - Nov 2002:  Created for Athena1.0/Cambridge release by Peter Teuben
//   - 2003-2008: Many improvements and extensions by T. Gardiner and J.M. Stone
//   - Jan 2014:  Rewritten in C++ for the Athena++ code by J.M. Stone
//========================================================================================

// C headers

// C++ headers
#include <algorithm>  // transform
#include <cmath>      // std::fmod()
#include <cstdlib>    // atoi(), nullptr, std::size_t
#include <fstream>    // ifstream
#include <iomanip>
#include <iostream>   // endl, ostream
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // string

// Athena++ headers
#include "athena_arrays.hpp"
#include "utils/utils.hpp"
#include "parameter_input.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

// ----------------------------------------------------------------------------
// Implement some convenience functions
namespace {

template <typename T>
std::string str_join(const T& v, const std::string& delim)
{
  std::ostringstream s;
  for (const auto& i : v)
  {
    if (&i != &v[0]) {
      s << delim;
    }
    s << i;
  }
  return s.str();
}

void str_replace(std::string & str,
                 const std::string &from,
                 const std::string &to)
{
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos)
  {
    str.replace(start_pos, from.length(), to);
    start_pos +=
        to.length(); // Handles case where 'to' is a substring of 'from'
  }
}


// any lines with only leading spaces should be dropped
bool is_line_empty(const std::string& str)
{
  return !str.empty() && std::all_of(str.begin(), str.end(), [](char c)
  {
      return c == '\n' || c == '\r' || c == ' ' || c == '\t';
  });
}

void line_to_var_comment(
  const std::string& input,
  std::string& content,
  std::string& comment)
{
  size_t pos = input.find('#'); // Find the first occurrence of '#'

  if (pos != std::string::npos)
  {
    content = input.substr(0, pos);  // Extract everything before '#'
    comment = input.substr(pos);     // Extract the comment including '#'
  } else {
      content = input;  // No comment found, so full string is content
      comment = "";     // No comment
  }

  size_t end = content.find_last_not_of(" \t");
  if (end != std::string::npos) {
      content = content.substr(0, end + 1);
  } else {
      content.clear(); // If it's all spaces, clear it
  }
}

template<typename T>
T numeric_from_str(const std::string & to_parse)
{
  std::istringstream iss(to_parse);
  T val;
  iss >> val;
  return val;
}

template<typename T>
std::string str_from_numeric(T to_parse)
{
  std::ostringstream oss;
  oss.precision(std::numeric_limits<T>::digits10);
  oss << std::scientific << to_parse;
  return oss.str();
}

} // namespace

//----------------------------------------------------------------------------------------
// ParameterInput constructor

ParameterInput::ParameterInput() :pfirst_block{}, last_filename_{} {
#ifdef OPENMP_PARALLEL
  omp_init_lock(&lock_);
#endif
}

// ParameterInput destructor- iterates through nested singly linked lists of blocks/lines
// and deletes each InputBlock node (whose destructor below deletes linked list "line"
// nodes)

ParameterInput::~ParameterInput() {
  InputBlock *pib = pfirst_block;
  while (pib != nullptr) {
    InputBlock *pold_block = pib;
    pib = pib->pnext;
    delete pold_block;
  }
#ifdef OPENMP_PARALLEL
  omp_destroy_lock(&lock_);
#endif
}

// InputBlock destructor- iterates through singly linked list of "line" nodes and deletes
// them

InputBlock::~InputBlock() {
  InputLine *pil = pline;
  while (pil != nullptr) {
    InputLine *pold_line = pil;
    pil = pil->pnext;
    delete pold_line;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void ParameterInput::LoadFromStream(std::istream &is)
//  \brief Load input parameters from a stream

//  Input block names are allocated and stored in a singly linked list of InputBlocks.
//  Within each InputBlock the names, values, and comments of each parameter are allocated
//  and stored in a singly linked list of InputLines.

void ParameterInput::LoadFromStream(std::istream &is) {
  std::string line, block_name, param_name, param_value, param_comment;
  std::size_t first_char, last_char;
  std::stringstream msg;
  InputBlock *pib{};
  // int line_num{-1};
  int blocks_found{0};

  while (is.good()) {
    std::getline(is, line);
    // line_num++;
    if (line.find('\t') != std::string::npos) {
      line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());
      // msg << "### FATAL ERROR in function [ParameterInput::LoadFromStream]"
      //     << std::endl << "Tab characters are forbidden in input files";
      // ATHENA_ERROR(msg);
    }
    if (line.empty()) continue;                             // skip blank line
    first_char = line.find_first_not_of(" ");               // skip white space
    if (first_char == std::string::npos) continue;          // line is all white space
    if (line.compare(first_char, 1, "#") == 0) continue;      // skip comments
    if (line.compare(first_char, 9, "<par_end>") == 0) break; // stop on <par_end>

    if (line.compare(first_char, 1, "<") == 0) {              // a new block
      first_char++;
      last_char = (line.find_first_of(">", first_char));
      block_name.assign(line, first_char, last_char-1);       // extract block name

      if (last_char == std::string::npos) {
        msg << "### FATAL ERROR in function [ParameterInput::LoadFromStream]"
            << std::endl << "Block name '" << block_name
            << "' in the input stream'" << "' not properly ended";
        ATHENA_ERROR(msg);
      }

      pib = FindOrAddBlock(block_name);  // find or add block to singly linked list

      if (pib == nullptr) {
        msg << "### FATAL ERROR in function [ParameterInput::LoadFromStream]"
            << std::endl << "Block name '" << block_name
            << "' could not be found/added";
        ATHENA_ERROR(msg);
      }
      blocks_found++;
      continue;  // skip to next line if block name was found
    } // end "a new block was found"

    // if line does not contain a block name or skippable information (comments,
    // whitespace), it must contain a parameter value
    if (blocks_found == 0) {
        msg << "### FATAL ERROR in function [ParameterInput::LoadFromStream]"
            << std::endl << "Input file must specify a block name before the first"
            << " parameter = value line";
        ATHENA_ERROR(msg);
    }

    // If we have a '[' but no ']' then we have a multi-line specification of a param
    // Concatenate following lines to this one
    if ((count_char(line, '[') == 1) && (count_char(line, ']') == 0))
    {
      // we concatenate until '[' and ']' live on a single line
      std::stringstream lines;
      std::stringstream comment_lines;

      bool skip_first = true;
      while ((count_char(line, ']') == 0) && is.good())
      {
        std::string line_var, line_comment;

        if (!skip_first)
          std::getline(is, line);
        skip_first = false;

        if (line.size() == 0) continue;

        // size_t pos_com = line.find_first_of('#');
        first_char = line.find_first_not_of(" ");

        if (is_line_empty(line)) continue;
        if (line.compare(first_char, 1, "#") == 0) continue;

        if ((line.compare(first_char, 1, "<") == 0) ||
            (line.compare(first_char, 9, "<par_end>") == 0))
        {
          std::ostringstream err;
          err << "Malformed multi-line parameter in: " << block_name;
          ATHENA_ERROR(err);
        }

        line_to_var_comment(line, line_var, line_comment);
        str_replace(line_var, " ", "");
        lines << line_var;

        if (line_comment.size() > 0)
          comment_lines << line_comment << " ";
      }

      lines << comment_lines.str();

      // we have the concatenated line
      line.assign(lines.str());
    }

    // parse line and add name/value/comment strings (if found) to current block name
    ParseLine(pib, line, param_name, param_value, param_comment);
    AddParameter(pib, param_name, param_value, param_comment);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void ParameterInput::LoadFromFile(IOWrapper &input)
//  \brief Read the parameters from an input file or restarting file.
//         Return the position at the end of the header, which is used in restarting

void ParameterInput::LoadFromFile(IOWrapper &input) {
  std::stringstream par, msg;
  constexpr int kBufSize = 40960;
  char buf[kBufSize];
  IOWrapperSizeT header = 0, ret, loc;

  // search <par_end> or EOF.
  do {
    if (Globals::my_rank == 0) // only the master process reads the header from the file
      ret = input.Read(buf, sizeof(char), kBufSize);
#ifdef MPI_PARALLEL
    // then broadcasts it
    MPI_Bcast(&ret, sizeof(IOWrapperSizeT), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(buf, ret, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
    par.write(buf, ret); // add the buffer into the stream
    header += ret;
    std::string sbuf = par.str(); // create string for search
    loc = sbuf.find("<par_end>", 0); // search from the top of the stream
    if (loc != std::string::npos) { // found <par_end>
      header = loc + 10; // store the header length
      break;
    }
    if (header > kBufSize*10) {
      msg << "### FATAL ERROR in function [ParameterInput::LoadFromFile]"
          << "<par_end> is not found in the first 400KBytes." << std::endl
          << "Probably the file is broken or a wrong file is specified" << std::endl;
      ATHENA_ERROR(msg);
    }
  } while (ret == kBufSize); // till EOF (or par_end is found)

  // Now par contains the parameter inputs + some additional including <par_end>
  // Read the stream and load the parameters
  LoadFromStream(par);
  // Seek the file to the end of the header
  input.Seek(header);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn InputBlock* ParameterInput::FindOrAddBlock(std::string name)
//  \brief find or add specified InputBlock.  Returns pointer to block.

InputBlock* ParameterInput::FindOrAddBlock(std::string name) {
  InputBlock *pib, *plast;
  plast = pfirst_block;
  pib = pfirst_block;

  // Search singly linked list of InputBlocks to see if name exists, return if found.
  while (pib != nullptr) {
    if (name.compare(pib->block_name) == 0) return pib;
    plast = pib;
    pib = pib->pnext;
  }

  // Create new block in list if not found above
  pib = new InputBlock;
  pib->block_name.assign(name);  // store the new block name
  pib->pline = nullptr;             // Terminate the InputLine list
  pib->pnext = nullptr;             // Terminate the InputBlock list

  // if this is the first block in list, save pointer to it in class
  if (pfirst_block == nullptr) {
    pfirst_block = pib;
  } else {
    plast->pnext = pib;      // link new node into list
  }

  return pib;
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::ParseLine(InputBlock *pib, std::string line,
//           std::string& name, std::string& value, std::string& comment)
//  \brief parse "name = value # comment" format, return name/value/comment strings.

void ParameterInput::ParseLine(InputBlock *pib, std::string line,
                               std::string& name, std::string& value,
                               std::string& comment) {
  std::size_t first_char, last_char, equal_char, hash_char, len;

  first_char = line.find_first_not_of(" ");   // find first non-white space
  equal_char = line.find_first_of("=");       // find "=" char
  hash_char  = line.find_first_of("#");       // find "#" (optional)

  // copy substring into name, remove white space at end of name
  len = equal_char - first_char;
  name.assign(line, first_char, len);

  last_char = name.find_last_not_of(" ");
  name.erase(last_char+1, std::string::npos);

  // copy substring into value, remove white space at start and end
  len = hash_char - equal_char - 1;
  value.assign(line, equal_char+1, len);

  first_char = value.find_first_not_of(" ");
  value.erase(0, first_char);

  last_char = value.find_last_not_of(" ");
  value.erase(last_char+1, std::string::npos);

  // copy substring into comment, if present
  if (hash_char != std::string::npos) {
    comment = line.substr(hash_char);
  } else {
    comment = "";
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::AddParameter(InputBlock *pb, std::string name,
//   std::string value, std::string comment)
//  \brief add name/value/comment tuple to the InputLine singly linked list in block *pb.
//  If a parameter with the same name already exists, the value and comment strings
//  are replaced (overwritten).

void ParameterInput::AddParameter(InputBlock *pb, std::string name,
                                  std::string value, std::string comment) {
  InputLine *pl, *plast;
  // Search singly linked list of InputLines to see if name exists.  This also sets *plast
  // to point to the tail node (but not storing a pointer to the tail node in InputBlock)
  pl = pb->pline;
  plast = pb->pline;
  while (pl != nullptr) {
    if (name.compare(pl->param_name) == 0) {   // param name already exists
      pl->param_value.assign(value);           // replace existing param value
      pl->param_comment.assign(comment);       // replace exisiting param comment
      if (value.length() > pb->max_len_parvalue) pb->max_len_parvalue = value.length();
      return;
    }
    plast = pl;
    pl = pl->pnext;
  }

  // Create new node in singly linked list if name does not already exist
  pl = new InputLine;
  pl->param_name.assign(name);
  pl->param_value.assign(value);
  pl->param_comment.assign(comment);
  pl->pnext = nullptr;

  // if this is the first parameter in list, save pointer to it in block.
  if (pb->pline == nullptr) {
    pb->pline = pl;
    pb->max_len_parname = name.length();
    pb->max_len_parvalue = value.length();
  } else {
    plast->pnext = pl;  // link new node into list
    if (name.length() > pb->max_len_parname) pb->max_len_parname = name.length();
    if (value.length() > pb->max_len_parvalue) pb->max_len_parvalue = value.length();
  }

  return;
}

//----------------------------------------------------------------------------------------
//! void ParameterInput::ModifyFromCmdline(int argc, char *argv[])
//  \brief parse commandline for changes to input parameters
// Note this function is very forgiving (no warnings!) if there is an error in format

void ParameterInput::ModifyFromCmdline(int argc, char *argv[]) {
  std::string input_text, block,name, value;
  std::stringstream msg;
  InputBlock *pb;
  InputLine *pl;

  for (int i=1; i<argc; i++) {
    input_text = argv[i];
    std::size_t slash_posn = input_text.find_first_of("/");   // find "/" character
    std::size_t equal_posn = input_text.find_first_of("=");   // find "=" character

    // skip if either "/" or "=" do not exist in input
    if ((slash_posn == std::string::npos) || (equal_posn == std::string::npos)) continue;

    // extract block/name/value strings
    block = input_text.substr(0, slash_posn);
    name  = input_text.substr(slash_posn+1, (equal_posn - slash_posn - 1));
    value = input_text.substr(equal_posn+1, std::string::npos);

    // Attempt to inject missing blocks / parameters if they do not exist -----
    pb = FindOrAddBlock(block);

    if(!DoesParameterExist(block, name))
    {
      std::string comment {"# Added from cmd-line."};
      AddParameter(pb, name, value, comment);
    }
    else
    {
      // get pointer to node with same parameter name in singly linked list of
      // InputLines
      pl = pb->GetPtrToLine(name);
      pl->param_value.assign(value);   // replace existing value
    }
    // ------------------------------------------------------------------------

    if (value.length() > pb->max_len_parvalue) pb->max_len_parvalue = value.length();
  }
}

//----------------------------------------------------------------------------------------
//! \fn InputBlock* ParameterInput::GetPtrToBlock(std::string name)
//  \brief return pointer to specified InputBlock if it exists

InputBlock* ParameterInput::GetPtrToBlock(std::string name) {
  InputBlock *pb;
  for (pb = pfirst_block; pb != nullptr; pb = pb->pnext) {
    if (name.compare(pb->block_name) == 0) return pb;
  }
  return nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::DoesParameterExist(std::string block, std::string name)
//  \brief check whether parameter of given name in given block exists

int ParameterInput::DoesParameterExist(std::string block, std::string name) {
  InputLine *pl;
  InputBlock *pb;
  pb = GetPtrToBlock(block);
  if (pb == nullptr) return 0;
  pl = pb->GetPtrToLine(name);
  return (pl == nullptr ? 0 : 1);
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::GetInteger(std::string block, std::string name)
//  \brief returns integer value of string stored in block/name

int ParameterInput::GetInteger(std::string block, std::string name) {
  InputBlock* pb;
  InputLine* pl;
  std::stringstream msg;

  Lock();

  // get pointer to node with same block name in singly linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetInteger]" << std::endl
        << "Block name '" << block << "' not found when trying to set value "
        << "for parameter '" << name << "'";
    ATHENA_ERROR(msg);
  }

  // get pointer to node with same parameter name in singly linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetInteger]" << std::endl
        << "Parameter name '" << name << "' not found in block '" << block << "'";
    ATHENA_ERROR(msg);
  }

  std::string val=pl->param_value;
  Unlock();

  // Convert string to integer and return value
  return atoi(val.c_str());
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::GetReal(std::string block, std::string name)
//  \brief returns real value of string stored in block/name

Real ParameterInput::GetReal(std::string block, std::string name) {
  InputBlock* pb;
  InputLine* pl;
  std::stringstream msg;

  Lock();

  // get pointer to node with same block name in singly linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetReal]" << std::endl
        << "Block name '" << block << "' not found when trying to set value "
        << "for parameter '" << name << "'";
    ATHENA_ERROR(msg);
  }

  // get pointer to node with same parameter name in singly linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetReal]" << std::endl
        << "Parameter name '" << name << "' not found in block '" << block << "'";
    ATHENA_ERROR(msg);
  }

  std::string val=pl->param_value;
  Unlock();

  // Convert string to real and return value
  return numeric_from_str<Real>(val.c_str());
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::GetBoolean(std::string block, std::string name)
//  \brief returns boolean value of string stored in block/name

bool ParameterInput::GetBoolean(std::string block, std::string name) {
  InputBlock* pb;
  InputLine* pl;
  std::stringstream msg;

  Lock();

  // get pointer to node with same block name in singly linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetBoolean]" << std::endl
        << "Block name '" << block << "' not found when trying to set value "
        << "for parameter '" << name << "'";
    ATHENA_ERROR(msg);
  }

  // get pointer to node with same parameter name in singly linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetBoolean]" << std::endl
        << "Parameter name '" << name << "' not found in block '" << block << "'";
    ATHENA_ERROR(msg);
  }

  std::string val=pl->param_value;
  Unlock();

  // check is string contains integers 0 or 1 (instead of true or false) and return
  if (val.compare(0, 1, "0")==0 || val.compare(0, 1, "1")==0) {
    return static_cast<bool>(atoi(val.c_str()));
  }

  // convert string to all lower case
  std::transform(val.begin(), val.end(), val.begin(), ::tolower);
  // Convert string to bool and return value
  bool b;
  std::istringstream is(val);
  is >> std::boolalpha >> b;

  return (b);
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::GetString(std::string block, std::string name)
//  \brief returns string stored in block/name

std::string ParameterInput::GetString(std::string block, std::string name) {
  InputBlock* pb;
  InputLine* pl;
  std::stringstream msg;

  Lock();

  // get pointer to node with same block name in singly linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetString]" << std::endl
        << "Block name '" << block << "' not found when trying to set value "
        << "for parameter '" << name << "'";
    ATHENA_ERROR(msg);
  }

  // get pointer to node with same parameter name in singly linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    msg << "### FATAL ERROR in function [ParameterInput::GetString]" << std::endl
        << "Parameter name '" << name << "' not found in block '" << block << "'";
    ATHENA_ERROR(msg);
  }

  std::string val=pl->param_value;
  Unlock();

  // return value
  return val;
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::GetOrAddInteger(std::string block, std::string name,
//    int default_value)
//  \brief returns integer value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

int ParameterInput::GetOrAddInteger(std::string block, std::string name, int def_value) {
  InputBlock* pb;
  InputLine *pl;
  std::stringstream ss_value;
  int ret;

  Lock();
  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    std::string val = pl->param_value;
    ret = atoi(val.c_str());
  } else {
    pb = FindOrAddBlock(block);
    ss_value << def_value;
    AddParameter(pb, name, ss_value.str(), "# Default");
    ret = def_value;
  }
  Unlock();
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::GetOrAddReal(std::string block, std::string name,
//    Real def_value)
//  \brief returns real value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

Real ParameterInput::GetOrAddReal(std::string block, std::string name, Real def_value) {
  InputBlock* pb;
  InputLine *pl;
  std::stringstream ss_value;
  Real ret;

  Lock();
  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    std::string val = pl->param_value;
    ret = numeric_from_str<Real>(val.c_str());
  } else {
    pb = FindOrAddBlock(block);
    ss_value << def_value;
    AddParameter(pb, name, ss_value.str(), "# Default");
    ret = def_value;
  }
  Unlock();
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::GetOrAddBoolean(std::string block, std::string name,
//    bool def_value)
//  \brief returns boolean value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

bool ParameterInput::GetOrAddBoolean(std::string block,std::string name, bool def_value) {
  InputBlock* pb;
  InputLine *pl;
  std::stringstream ss_value;
  bool ret;

  Lock();
  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    std::string val = pl->param_value;
    if (val.compare(0, 1, "0")==0 || val.compare(0, 1, "1")==0) {
      ret = static_cast<bool>(atoi(val.c_str()));
    } else {
      std::transform(val.begin(), val.end(), val.begin(), ::tolower);
      std::istringstream is(val);
      is >> std::boolalpha >> ret;
    }
  } else {
    pb = FindOrAddBlock(block);
    ss_value << def_value;
    AddParameter(pb, name, ss_value.str(), "# Default");
    ret = def_value;
  }
  Unlock();
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::GetOrAddString(std::string block, std::string name,
//                                                 std::string def_value)
//  \brief returns string value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

std::string ParameterInput::GetOrAddString(std::string block, std::string name,
                                           std::string def_value) {
  InputBlock* pb;
  InputLine *pl;
  std::stringstream ss_value;
  std::string ret;

  Lock();
  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    ret = pl->param_value;
  } else {
    pb = FindOrAddBlock(block);
    AddParameter(pb, name, def_value, "# Default");
    ret = def_value;
  }
  Unlock();
  return ret;
}

// Assuming block/name exists, return string array
void ParameterInput::GetExistingStringArray(
  const std::string & block,
  const std::string & name,
  t_vec_str & vec)
{
  InputBlock* pb;
  InputLine *pl;

  pb = GetPtrToBlock(block);
  pl = pb->GetPtrToLine(name);

  // Check open / close bracket exists; otherwise panic
  std::stringstream ss_par;

  if (count_char(pl->param_value, '[') == 1)
  {
    ss_par << pl->param_value;

    while ((pl != nullptr) &&
           (count_char(pl->param_value, ']') != 1))
    {
      pl = pl->pnext;

      if (pl != nullptr)
      {
        ss_par << pl->param_value;
      }
    }
  }
  else
  {
    // No open bracket? try extraction as flat scalar
    ss_par << "[" << pl->param_value << "]";
  }

  if (pl == nullptr)
  {
    std::ostringstream err;
    err << "Malformed parameter: " << block << "/" << name;
    ATHENA_ERROR(err);
  }

  std::string to_tokenize { ss_par.str() };

  // strip braces, spaces & split to elements
  for (auto c : {"[", "]", " "})
  {
    str_replace(to_tokenize, c, "");
  }

  std::istringstream iss(to_tokenize);
  std::string s;

  while (std::getline(iss, s, ','))
  {
    vec.push_back(s);
  }
}

void ParameterInput::AddParameterStringArray(
  const std::string & block,
  const std::string & name,
  const t_vec_str & values)
{
  InputBlock* pb;
  pb = FindOrAddBlock(block);
  std::stringstream ss_def_value;
  ss_def_value << "[" << str_join(values, ",") << "]";
  AddParameter(pb,
               name,
               ss_def_value.str(),
               "# Default");
}

ParameterInput::t_vec_str ParameterInput::GetOrAddStringArray(
  std::string block,
  std::string name,
  t_vec_str def_values)
{
  InputBlock* pb;
  InputLine *pl;
  std::stringstream ss_value;
  t_vec_str ret;

  Lock();
  if (DoesParameterExist(block, name))
  {
    GetExistingStringArray(block, name, ret);
  }
  else
  {
    AddParameterStringArray(block, name, def_values);

    // copy input default to return
    for (const auto& dval : def_values)
    {
      ret.push_back(dval);
    }
  }
  Unlock();

  return ret;
}

ParameterInput::t_vec_Real ParameterInput::GetOrAddRealArray(
  std::string block,
  std::string name,
  t_vec_Real def_values)
{
  InputBlock* pb;
  InputLine *pl;
  t_vec_Real ret;

  // Convert def_values to string representation for internal storage
  t_vec_str s_irep_def_values;
  for (auto el : def_values)
  {
    s_irep_def_values.push_back(std::to_string(el));
  }

  Lock();
  if (DoesParameterExist(block, name))
  {
    t_vec_str s_ret;
    GetExistingStringArray(block, name, s_ret);
    for (const auto& sval : s_ret)
    {
      ret.push_back(numeric_from_str<Real>(sval.c_str()));
    }
  }
  else
  {
    AddParameterStringArray(block, name, s_irep_def_values);

    // copy input default to return
    for (const auto& dval : def_values)
    {
      ret.push_back(dval);
    }
  }
  Unlock();

  return ret;
}

ParameterInput::t_vec_int ParameterInput::GetOrAddIntegerArray(
  std::string block,
  std::string name,
  t_vec_int def_values)
{
  InputBlock* pb;
  InputLine *pl;
  t_vec_int ret;

  // Convert def_values to string representation for internal storage
  t_vec_str s_irep_def_values;
  for (auto el : def_values)
  {
    s_irep_def_values.push_back(std::to_string(el));
  }

  Lock();
  if (DoesParameterExist(block, name))
  {
    t_vec_str s_ret;
    GetExistingStringArray(block, name, s_ret);
    for (const auto& sval : s_ret)
    {
      ret.push_back(static_cast<int>(atoi(sval.c_str())));
    }
  }
  else
  {
    AddParameterStringArray(block, name, s_irep_def_values);

    // copy input default to return
    for (const auto& dval : def_values)
    {
      ret.push_back(dval);
    }
  }
  Unlock();

  return ret;
}

ParameterInput::t_vec_bool ParameterInput::GetOrAddBooleanArray(
  std::string block,
  std::string name,
  t_vec_bool def_values)
{
  InputBlock* pb;
  InputLine *pl;
  t_vec_bool ret;

  // Convert def_values to string representation for internal storage
  t_vec_str s_irep_def_values;
  for (auto el : def_values)
  {
    s_irep_def_values.push_back(std::to_string(el));
  }

  Lock();
  if (DoesParameterExist(block, name))
  {
    t_vec_str s_ret;
    GetExistingStringArray(block, name, s_ret);
    for (auto& sval : s_ret)
    {
      bool bval;

      if (sval.compare(0, 1, "0")==0 || sval.compare(0, 1, "1")==0)
      {
        bval = static_cast<bool>(atoi(sval.c_str()));
      }
      else
      {
        std::transform(sval.begin(), sval.end(), sval.begin(), ::tolower);
        std::istringstream is(sval);
        is >> std::boolalpha >> bval;
      }

      ret.push_back(bval);
    }
  }
  else
  {
    AddParameterStringArray(block, name, s_irep_def_values);

    // copy input default to return
    for (const auto& dval : def_values)
    {
      ret.push_back(dval);
    }
  }
  Unlock();

  return ret;
}

template<>
ParameterInput::t_vec_Real ParameterInput::GetOrAddArray
  <ParameterInput::t_vec_Real>(
    std::string block, std::string name, ParameterInput::t_vec_Real def_values
  )
{
  return GetOrAddRealArray(block, name, def_values);
}

template<>
ParameterInput::t_vec_int ParameterInput::GetOrAddArray
  <ParameterInput::t_vec_int>(
    std::string block, std::string name, ParameterInput::t_vec_int def_values
  )
{
  return GetOrAddIntegerArray(block, name, def_values);
}

template<>
ParameterInput::t_vec_bool ParameterInput::GetOrAddArray
  <ParameterInput::t_vec_bool>(
    std::string block, std::string name, ParameterInput::t_vec_bool def_values
  )
{
  return GetOrAddBooleanArray(block, name, def_values);
}

template<>
ParameterInput::t_vec_str ParameterInput::GetOrAddArray
  <ParameterInput::t_vec_str>(
    std::string block, std::string name, ParameterInput::t_vec_str def_values
  )
{
  return GetOrAddStringArray(block, name, def_values);
}

// Convenience functions for direct usage with AthenaArray
template<class T>
AthenaArray<T> ParameterInput::GetOrAddArray(
  const std::string & block,
  const std::string & name,
  const AthenaArray<T> & def_values)
{
  const int sz = def_values.GetDim1();
  AthenaArray<T> ret;

  std::vector<T> v_def_values;
  for (int i=0; i<sz; ++i)
  {
    v_def_values.push_back(def_values(i));
  }

  std::vector<T> v_ret {GetOrAddArray(block, name, v_def_values)};
  // GCC has some issue; limit maximum size explicitly
  const int v_sz = std::max(
    std::min(static_cast<int>(v_ret.size()), max_pars_array), 0);

  // const int v_sz = v_ret.size();
  ret.NewAthenaArray(v_sz);

  for (int i=0; i<v_sz; ++i)
  {
    ret(i) = v_ret[i];
  }

  return ret;
}

// Main interfaces
AthenaArray<Real> ParameterInput::GetOrAddRealArray(
  const std::string & block,
  const std::string & name,
  const AthenaArray<Real> & def_values)
{
  return GetOrAddArray(block, name, def_values);
}

AthenaArray<int> ParameterInput::GetOrAddIntegerArray(
  const std::string & block,
  const std::string & name,
  const AthenaArray<int> & def_values)
{
  return GetOrAddArray(block, name, def_values);
}

AA_B ParameterInput::GetOrAddBooleanArray(
  const std::string & block,
  const std::string & name,
  const AA_B & def_values)
{
  return GetOrAddArray(block, name, def_values);
}

AthenaArray<std::string> ParameterInput::GetOrAddStringArray(
  const std::string & block,
  const std::string & name,
  const AthenaArray<std::string> & def_values)
{
  return GetOrAddArray(block, name, def_values);
}

// Wrap input type to array
AthenaArray<Real> ParameterInput::GetOrAddRealArray(
  const std::string & block,
  const std::string & name,
  const Real & def_value,
  const int size)
{
  AthenaArray<Real> dv(size);
  for (int n=0; n<size; ++n)
  {
    dv(n) = def_value;
  }
  return GetOrAddArray(block, name, dv);
}

AthenaArray<int> ParameterInput::GetOrAddIntegerArray(
  const std::string & block,
  const std::string & name,
  const int & def_value,
  const int size)
{
  AthenaArray<int> dv(size);
  for (int n=0; n<size; ++n)
  {
    dv(n) = def_value;
  }
  return GetOrAddArray(block, name, dv);
}

AA_B ParameterInput::GetOrAddBooleanArray(
  const std::string & block,
  const std::string & name,
  const bool & def_value,
  const int size)
{
  AA_B dv(size);
  for (int n=0; n<size; ++n)
  {
    dv(n) = def_value;
  }
  return GetOrAddArray(block, name, dv);
}

AthenaArray<std::string> ParameterInput::GetOrAddStringArray(
  const std::string & block,
  const std::string & name,
  const std::string & def_value,
  const int size)
{
  AthenaArray<std::string> dv(size);
  for (int n=0; n<size; ++n)
  {
    dv(n) = def_value;
  }
  return GetOrAddArray(block, name, dv);
}

// ----------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::SetInteger(std::string block, std::string name, int value)
//  \brief updates an integer parameter; creates it if it does not exist

int ParameterInput::SetInteger(std::string block, std::string name, int value) {
  InputBlock* pb;
  std::stringstream ss_value;

  Lock();
  pb = FindOrAddBlock(block);
  ss_value << value;
  AddParameter(pb, name, ss_value.str(), "# Updated during run time");
  Unlock();
  return value;
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::SetReal(std::string block, std::string name, Real value)
//  \brief updates a real parameter; creates it if it does not exist

Real ParameterInput::SetReal(std::string block, std::string name, Real value) {
  InputBlock* pb;

  Lock();
  pb = FindOrAddBlock(block);
  std::string s_value {str_from_numeric(value)};
  AddParameter(pb, name, s_value, "# Updated during run time");
  Unlock();
  return value;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::SetBoolean(std::string block, std::string name, bool value)
//  \brief updates a boolean parameter; creates it if it does not exist

bool ParameterInput::SetBoolean(std::string block, std::string name, bool value) {
  InputBlock* pb;
  std::stringstream ss_value;

  Lock();
  pb = FindOrAddBlock(block);
  ss_value << value;
  AddParameter(pb, name, ss_value.str(), "# Updated during run time");
  Unlock();
  return value;
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::SetString(std::string block, std::string name,
//                                            std::string  value)
//  \brief updates a string parameter; creates it if it does not exist

std::string ParameterInput::SetString(std::string block, std::string name,
                                      std::string value) {
  InputBlock* pb;

  Lock();
  pb = FindOrAddBlock(block);
  AddParameter(pb, name, value, "# Updated during run time");
  Unlock();
  return value;
}

//----------------------------------------------------------------------------------------
//  \brief updates a real parameter; creates it if it does not exist

void ParameterInput::SetRealArray(
  std::string block, std::string name, AthenaArray<Real> & arr
)
{
  InputBlock* pb;

  Lock();
  pb = FindOrAddBlock(block);
  std::vector<std::string> s_arr;
  for (int n=0; n<arr.GetSize(); ++n)
  {
    s_arr.push_back(str_from_numeric(arr(n)));
  }

  std::stringstream ss;
  ss << "[" << str_join(s_arr, ",") << "]";
  AddParameter(pb, name, ss.str(), "# Updated during run time");
  Unlock();
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::RollbackNextTime()
//  \brief rollback next_time by dt for each output block

void ParameterInput::RollbackNextTime() {
  InputBlock *pb = pfirst_block;
  InputLine* pl;
  std::stringstream msg;
  Real next_time;

  while (pb != nullptr) {
    if (pb->block_name.compare(0, 6, "output") == 0) {
      pl = pb->GetPtrToLine("next_time");
      if (pl == nullptr) {
        msg << "### FATAL ERROR in function [ParameterInput::RollbackNextTime]"
            << std::endl << "Parameter name 'next_time' not found in block '"
            << pb->block_name << "'";
        ATHENA_ERROR(msg);
      }
      next_time = numeric_from_str<Real>(pl->param_value.c_str());
      pl = pb->GetPtrToLine("dt");
      if (pl == nullptr) {
        msg << "### FATAL ERROR in function [ParameterInput::RollbackNextTime]"
            << std::endl << "Parameter name 'dt' not found in block '"
            << pb->block_name << "'";
        ATHENA_ERROR(msg);
      }
      next_time -= numeric_from_str<Real>(pl->param_value.c_str());
      msg << next_time;
      //AddParameter(pb, "next_time", msg.str().c_str(), "# Updated during run time");
      SetReal(pb->block_name, "next_time", next_time);
    }
    pb = pb->pnext;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::ForwardNextTime()
//  \brief add dt to next_time until next_time >  mesh_time - dt for each output block

void ParameterInput::ForwardNextTime(Real mesh_time) {
  InputBlock *pb = pfirst_block;
  InputLine* pl;
  Real next_time;
  Real dt0, dt;
  bool fresh = false;

  while (pb != nullptr) {
    if (pb->block_name.compare(0, 6, "output") == 0) {
      std::stringstream msg;
      pl = pb->GetPtrToLine("next_time");
      if (pl == nullptr) {
        next_time = mesh_time;
        // This is a freshly added output
        fresh = true;
      } else {
        next_time = numeric_from_str<Real>(pl->param_value.c_str());
      }
      pl = pb->GetPtrToLine("dt");
      if (pl == nullptr) {
        msg << "### FATAL ERROR in function [ParameterInput::ForwardNextTime]"
            << std::endl << "Parameter name 'dt' not found in block '"
            << pb->block_name << "'";
        ATHENA_ERROR(msg);
      }
      dt0 = numeric_from_str<Real>(pl->param_value.c_str());
      dt = dt0 * static_cast<int>((mesh_time - next_time) / dt0) + dt0;
      if (dt > 0) {
        next_time += dt;
        // If the user has added a new/fresh output round to multiple of dt0,
        // and make sure that mesh_time - dt0 < next_time < mesh_time,
        // to ensure immediate writing
        if (fresh) next_time -= std::fmod(next_time, dt0) + dt0;
      }
      msg << next_time;
      AddParameter(pb, "next_time", msg.str().c_str(), "# Updated during run time");
    }
    pb = pb->pnext;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::ParameterDump(std::ostream& os)
//  \brief output entire InputBlock/InputLine hierarchy to specified stream

void ParameterInput::ParameterDump(std::ostream& os) {
  InputBlock *pb;
  InputLine *pl;
  std::string param_name,param_value;
  std::size_t len;

  os<< "#------------------------- PAR_DUMP -------------------------" << std::endl;
  os<< "# GR-Athena++ GIT_HASH: " << GIT_HASH << std::endl;

  for (pb = pfirst_block; pb != nullptr; pb = pb->pnext) { // loop over InputBlocks
    os<< "\n# =========================================================" << std::endl;
    os<< "<" << pb->block_name << ">" << std::endl;     // write block name
    for (pl = pb->pline; pl != nullptr; pl = pl->pnext) {   // loop over InputLines
      param_name.assign(pl->param_name);
      param_value.assign(pl->param_value);

      len = pb->max_len_parname - param_name.length() + 1;
      param_name.append(len,' ');                      // pad name to align vertically
      len = pb->max_len_parvalue - param_value.length() + 1;
      param_value.append(len,' ');                     // pad value to align vertically

      os<< param_name << "= " << param_value << pl->param_comment <<  std::endl;
    }
  }

  os<< "#------------------------- PAR_DUMP -------------------------" << std::endl;
  os<< "<par_end>" << std::endl;    // finish with par-end (useful in restart files)
}

//----------------------------------------------------------------------------------------
//! \fn InputLine* InputBlock::GetPtrToLine(std::string name)
//  \brief return pointer to InputLine containing specified parameter if it exists

InputLine* InputBlock::GetPtrToLine(std::string name) {
  for (InputLine* pl = pline; pl != nullptr; pl = pl->pnext) {
    if (name.compare(pl->param_name) == 0) return pl;
  }
  return nullptr;
}


//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::Lock()
//  \brief Lock ParameterInput for reading and writing
void ParameterInput::Lock() {
#ifdef OPENMP_PARALLEL
  omp_set_lock(&lock_);
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::Unlock()
//  \brief Unlock ParameterInput for reading and writing
void ParameterInput::Unlock() {
#ifdef OPENMP_PARALLEL
  omp_unset_lock(&lock_);
#endif
  return;
}
