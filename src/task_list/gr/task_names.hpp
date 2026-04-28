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

const TaskID PREP_Z4C_DERIV(17);
const TaskID INIT_Z4C_DERIV(19);

}  // TaskNames::GeneralRelativity::GR_Z4c

namespace TaskNames::GeneralRelativity::GRMHD_Z4c_Split {

namespace Phase_MHD {

const TaskID NONE(0);

const TaskID RECV_HYDFLX(1);
const TaskID RECV_FLDFLX(2);
const TaskID RECV_SCLRFLX(3);

const TaskID CALC_FLDFLX(11);

const TaskID SEND_HYDFLX(20);
const TaskID SEND_FLDFLX(21);
const TaskID SEND_SCLRFLX(22);

// collective hydro/scalar integration
const TaskID CALC_HYDSCLRFLX(24);
const TaskID INT_HYDSCLR(25);
const TaskID ADD_FLX_DIV(26);
const TaskID SRCTERM_HYD(27);

const TaskID INT_FLD(31);

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

const TaskID CCE_DUMP(52);

const TaskID PREP_Z4C_DERIV(51);
const TaskID INIT_Z4C_DERIV(53);

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

// ---------------------------------------------------------------------------
// Monolithic GRMHD+Z4c task list - single DAG per substep, no internal
// barriers.  All four split phases (MHD, Z4c, MHD_com, Finalize) are fused
// into one task graph with explicit data-flow dependencies.
// ---------------------------------------------------------------------------
namespace TaskNames::GeneralRelativity::GRMHD_Z4c_Monolithic {

const TaskID NONE(0);

// -- MHD compute branch (bits 1-12) ----------------------------------------
const TaskID INT_HYDSCLR(1);
const TaskID CALC_HYDSCLRFLX(2);
const TaskID RECV_HYDFLX(3);      // flux correction recv [multilevel]
const TaskID RECV_SCLRFLX(4);     // scalar flux recv     [multilevel + scalars]
const TaskID SEND_HYDFLX(5);      // flux correction send [multilevel]
const TaskID SEND_SCLRFLX(6);     // scalar flux send     [multilevel + scalars]
const TaskID ADD_FLX_DIV(7);
const TaskID SRCTERM_HYD(8);
const TaskID CALC_FLDFLX(9);      // EMF corner calc      [B-field]
const TaskID SEND_FLDFLX(10);     // EMF flux correction   [B-field]
const TaskID RECV_FLDFLX(11);     // EMF flux recv         [B-field]
const TaskID INT_FLD(12);         // CT field integration  [B-field]

// -- Z4c compute branch (bits 13-24) ---------------------------------------
const TaskID INIT_Z4C_DERIV(13);
const TaskID CALC_Z4CRHS(14);
const TaskID INT_Z4C(15);
const TaskID SEND_Z4C(16);
const TaskID RECV_Z4C(17);
const TaskID SETB_Z4C(18);
const TaskID PROLONG_Z4C(19);     // [multilevel]
const TaskID PHY_BVAL_Z4C(20);
const TaskID ALG_CONSTR(21);
const TaskID PREP_Z4C_DERIV(22);
const TaskID Z4C_TO_ADM(23);
const TaskID CCE_DUMP(24);        // [CCE_ENABLED]

// -- MHD ghost-zone send + C2P (bits 25-32) --------------------------------
const TaskID SEND_HYD(25);        // conserved send - fires before C2P
const TaskID CLEAR_FLXCORR(26);   // wait on flux-correction sends [multilevel]
const TaskID CONS2PRIMP(27);      // C2P on physical interior
const TaskID RECV_HYD(28);        // ghost-zone recv (polls from NONE)
const TaskID SETB_HYD(29);
const TaskID PROLONG_HYD(30);     // [multilevel]
const TaskID PHY_BVAL_HYD(31);

// -- Finalize (bits 32-40) -------------------------------------------------
const TaskID CONS2PRIMG(32);      // C2P on ghost zones
const TaskID UPDATE_SRC(33);      // GetMatter (re-couple ADM sources)
const TaskID ADM_CONSTR(34);
const TaskID Z4C_WEYL(35);
const TaskID USERWORK(36);
const TaskID NEW_DT(37);
const TaskID FLAG_AMR(38);        // [adaptive]

// -- Cleanup (bits 39-40) --------------------------------------------------
const TaskID CLEAR_Z4C(39);       // wait on Z4c ghost sends
const TaskID CLEAR_MAININT(40);   // wait on MainInt ghost sends

}  // namespace TaskNames::GeneralRelativity::GRMHD_Z4c_Monolithic


namespace TaskNames::GeneralRelativity::Aux_Z4c {

const TaskID NONE(0);
const TaskID WEYL_DECOMP(9);

}  // namespace TaskNames::GeneralRelativity::Aux_Z4c

namespace TaskNames::GeneralRelativity::PostAMR_Z4c {

const TaskID NONE(0);
// const TaskID CLEAR_ALLBND(1);

const TaskID PREP_Z4C_DERIV(1);

const TaskID ADM_CONSTR(8);
const TaskID Z4C_WEYL(9);

}  // namespace TaskNames::GeneralRelativity::PostAMR_Z4c

#endif  // GR_TASK_NAMES_HPP_
