#ifndef GR_TASK_NAMES_HPP_
#define GR_TASK_NAMES_HPP_

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../task_list.hpp"

namespace TaskNames::GeneralRelativity::GR_Z4c {

const TaskID NONE(0);
const TaskID CLEAR_ALLBND(1);
const TaskID CALC_Z4CRHS(2);
const TaskID INT_Z4C(3);
const TaskID SEND_Z4C(4);
const TaskID RECV_Z4C(5);
const TaskID SETB_Z4C(6);

const TaskID PROLONG(7);
const TaskID PHY_BVAL(8);

const TaskID ALG_CONSTR(9);
const TaskID Z4C_TO_ADM(10);
const TaskID USERWORK(11);
const TaskID NEW_DT(12);

const TaskID ADM_CONSTR(13);
const TaskID FLAG_AMR(14);

const TaskID ASSERT_FIN(15);
const TaskID Z4C_WEYL(16);
const TaskID CCE_DUMP(18);

}  // TaskNames::GeneralRelativity::GR_Z4c

namespace TaskNames::GeneralRelativity::GRMHD_Z4c {

const TaskID NONE(0);
const TaskID CLEAR_ALLBND(1);
const TaskID UPDATE_SRC(3);

const TaskID CALC_HYDFLX(4);
const TaskID CALC_FLDFLX(5);
const TaskID CALC_RADFLX(6);
const TaskID CALC_CHMFLX(7);

const TaskID SEND_HYDFLX(8);
const TaskID SEND_FLDFLX(9);

const TaskID RECV_HYDFLX(10);
const TaskID RECV_FLDFLX(11);

const TaskID ALG_CONSTR(12);
const TaskID Z4C_TO_ADM(13);

const TaskID SRCTERM_HYD(14);

const TaskID ADM_CONSTR(15);
const TaskID ASSERT_FIN(16);
const TaskID Z4C_WEYL(17);

const TaskID INT_HYD(18);
const TaskID INT_FLD(19);

const TaskID SEND_HYD(22);
const TaskID SEND_FLD(23);

const TaskID RECV_HYD(26);
const TaskID RECV_FLD(27);

const TaskID SETB_HYD(30);
const TaskID SETB_FLD(31);

const TaskID PROLONG_HYD(34);
const TaskID CONS2PRIM(35);
const TaskID PHY_BVAL_HYD(36);
const TaskID USERWORK(37);
const TaskID NEW_DT(38);
const TaskID FLAG_AMR(39);

const TaskID CALC_SCLRFLX(49);
const TaskID SEND_SCLRFLX(50);
const TaskID RECV_SCLRFLX(51);
const TaskID INT_SCLR(52);
const TaskID SEND_SCLR(53);
const TaskID RECV_SCLR(54);
const TaskID SETB_SCLR(55);

const TaskID CALC_Z4CRHS(57);
const TaskID INT_Z4C(58);
const TaskID SEND_Z4C(59);
const TaskID RECV_Z4C(60);
const TaskID SETB_Z4C(61);

const TaskID PROLONG_Z4C(62);
const TaskID PHY_BVAL_Z4C(63);

}  // namespace TaskNames::GeneralRelativity::GRMHD_Z4c

namespace TaskNames::GeneralRelativity::GRMHD_Z4c_Split {

namespace Phase_MHD {

const TaskID NONE(0);

const TaskID RECV_HYDFLX(1);
const TaskID RECV_FLDFLX(2);
const TaskID RECV_SCLRFLX(3);

const TaskID CALC_HYDFLX(10);
const TaskID CALC_FLDFLX(11);
const TaskID CALC_RADFLX(12);
const TaskID CALC_CHMFLX(13);
const TaskID CALC_SCLRFLX(14);

const TaskID SEND_HYDFLX(20);
const TaskID SEND_FLDFLX(21);
const TaskID SEND_SCLRFLX(22);

const TaskID INT_HYD(30);
const TaskID INT_FLD(31);
const TaskID INT_SCLR(32);

const TaskID SRCTERM_HYD(45);

const TaskID CLEAR_ALLBND(60);

} // TaskNames::GeneralRelativity::GRMHD_Z4c_Split::Phase_MHD

namespace Phase_MHD_com {

const TaskID NONE(0);

const TaskID CONS2PRIMP(4);

const TaskID RECV_HYD(5);
const TaskID RECV_FLD(6);
const TaskID RECV_SCLR(7);

const TaskID SEND_HYD(40);
const TaskID SEND_FLD(41);
const TaskID SEND_SCLR(42);

const TaskID SETB_HYD(50);
const TaskID SETB_FLD(51);
const TaskID SETB_SCLR(52);

const TaskID PROLONG_HYD(55);
const TaskID PHY_BVAL_HYD(56);

const TaskID CLEAR_ALLBND(60);

} // TaskNames::GeneralRelativity::GRMHD_Z4c_Split::Phase_MHD_com

namespace Phase_Z4c {

const TaskID NONE(0);

const TaskID RECV_Z4C(1);

const TaskID CALC_Z4CRHS(10);
const TaskID INT_Z4C(11);

const TaskID SEND_Z4C(20);

const TaskID SETB_Z4C(30);
const TaskID PROLONG_Z4C(31);
const TaskID PHY_BVAL_Z4C(32);

const TaskID ALG_CONSTR(40);

const TaskID Z4C_TO_ADM(50);

const TaskID CLEAR_ALLBND(60);

} // TaskNames::GeneralRelativity::GRMHD_Z4c_Split::Phase_Z4c

namespace Finalize {

const TaskID NONE(0);

const TaskID CONS2PRIMG(3);

const TaskID UPDATE_SRC(4);

const TaskID ADM_CONSTR(15);
const TaskID ASSERT_FIN(16);
const TaskID Z4C_WEYL(17);

const TaskID USERWORK(37);
const TaskID NEW_DT(38);
const TaskID FLAG_AMR(39);

} // TaskNames::GeneralRelativity::GRMHD_Z4c_Split::Finalize

}  // namespace TaskNames::GeneralRelativity::GRMHD_Z4c_Split


namespace TaskNames::GeneralRelativity::Aux_Z4c {

const TaskID NONE(0);
const TaskID WEYL_DECOMP(9);

}  // namespace TaskNames::GeneralRelativity::Aux_Z4c

namespace TaskNames::GeneralRelativity::PostAMR_Z4c {

const TaskID NONE(0);
// const TaskID CLEAR_ALLBND(1);

const TaskID ADM_CONSTR(8);
const TaskID Z4C_WEYL(9);

}  // namespace TaskNames::GeneralRelativity::PostAMR_Z4c

#endif  // GR_TASK_NAMES_HPP_
