#ifndef GR_TASK_NAMES_HPP_
#define GR_TASK_NAMES_HPP_

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../task_list.hpp"

namespace TaskNames::GeneralRelativity::PostAMR {

const TaskID NONE(0);
const TaskID CLEAR_ALLBND(1);

const TaskID ALG_CONSTR(5);
const TaskID Z4C_TO_ADM(6);
const TaskID UPDATE_SRC(7);
const TaskID ADM_CONSTR(8);

const TaskID Z4C_WEYL(9);

}  // namespace TaskNames::GeneralRelativity

#endif  // GR_TASK_NAMES_HPP_
