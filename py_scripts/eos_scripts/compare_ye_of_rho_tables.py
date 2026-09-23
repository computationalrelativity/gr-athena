import numpy as np
import matplotlib.pyplot as plt
from pyparsing import line

table_1 = np.loadtxt("/home/mi58rip/gr-athena/eos_tables/y_e_vs_rho_1.46Msun_Tc1.0d10_VULCAN2D.dat")
table_2 = np.loadtxt("/home/mi58rip/gr-athena/eos_tables/y_e_vs_rho_1.46Msun_Tc1.3d10_VULCAN2D.dat")
table_3 = np.loadtxt("/home/mi58rip/gr-athena/eos_tables/y_e_vs_rho_1.46Msun_Tc5.0d09_VULCAN2D.dat")
table_4 = np.loadtxt("/home/mi58rip/gr-athena/eos_tables/y_e_vs_rho_1.92Msun_Tc1.3d10_VULCAN2D.dat")
table_5 = np.loadtxt("/home/mi58rip/gr-athena/eos_tables/y_e_vs_rho_1.92Msun_Tc5.0d09_VULCAN2D.dat")

temp_1_MeV = 1.0e10 / 1.16045e10
temp_2_MeV = 1.3e10 / 1.16045e10
temp_3_MeV = 5.0e09 / 1.16045e10
temp_4_MeV = 1.3e10 / 1.16045e10
temp_5_MeV = 5.0e09 / 1.16045e10

table_luis = np.loadtxt("/home/mi58rip/gr-athena/eos_tables/ye_of_rho_luis.dat")

ye_1 = table_1[:, 1]
ye_2 = table_2[:, 1]
ye_3 = table_3[:, 1]
ye_4 = table_4[:, 1]
ye_5 = table_5[:, 1]

ye_luis = table_luis[:, 1]

rho_1 = table_1[:, 0]
rho_2 = table_2[:, 0]
rho_3 = table_3[:, 0]
rho_4 = table_4[:, 0]
rho_5 = table_5[:, 0]

rho_luis = 10**table_luis[:, 0]

def Ye_of_rho(
    rho,
    log10_rho1,
    log10_rho2,
    Ye_2,
    Ye_c,
    Ye_H,
    Ye_1=0.5,
    log10_rhoH=15.0,
):

    log10_rho = np.log10(rho)

    # --------------------------------------------------------
    # x parameter
    # --------------------------------------------------------

    x = (
        2.0 * log10_rho
        - log10_rho2
        - log10_rho1
    ) / (log10_rho2 - log10_rho1)

    x = np.clip(x, -1.0, 1.0)

    abs_x = np.abs(x)

    # --------------------------------------------------------
    # Low/intermediate density branch
    # --------------------------------------------------------

    Ye_low = (
        0.5 * (Ye_2 + Ye_1)
        + 0.5 * x * (Ye_2 - Ye_1)
        + Ye_c
        * (
            1.0
            - abs_x
            + 4.0 * abs_x
            * (abs_x - 0.5)
            * (abs_x - 1.0)
        )
    )

    # --------------------------------------------------------
    # High-density linear branch
    # --------------------------------------------------------

    m = (
        Ye_H - Ye_2
    ) / (
        log10_rhoH - log10_rho2
    )

    Ye_high = (
        Ye_2
        + m * (log10_rho - log10_rho2)
    )

    # --------------------------------------------------------
    # Piecewise function
    # --------------------------------------------------------

    return np.where(
        log10_rho > log10_rho2,
        Ye_high,
        Ye_low,
    )

initial_guess = np.array([
    7.700,              # log10_rho1
    13.200 - 7.700,     # delta = log10_rho2 - log10_rho1
    0.285,              # Ye_2
    0.035,              # Ye_c
    0.265,              # Ye_H
])

ye_analytical = Ye_of_rho(
    rho_luis, 
    initial_guess[0],
    initial_guess[0] + initial_guess[1],
    initial_guess[2],
    initial_guess[3],
    initial_guess[4]
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rho_1, ye_1, label=f'1.46Msun Tc={temp_1_MeV:.1f} MeV', color='blue')
ax.plot(rho_2, ye_2, label=f'1.46Msun Tc={temp_2_MeV:.1f} MeV', color='orange')
ax.plot(rho_3, ye_3, label=f'1.46Msun Tc={temp_3_MeV:.1f} MeV', color='green')
ax.plot(rho_4, ye_4, label=f'1.92Msun Tc={temp_4_MeV:.1f} MeV', color='red')
ax.plot(rho_5, ye_5, label=f'1.92Msun Tc={temp_5_MeV:.1f} MeV', color='purple')
ax.plot(rho_luis, ye_luis, label='THC', color='black', linestyle='--')
ax.plot(rho_luis, ye_analytical, label='Analytical', color='gray', linestyle='-.')

ax.set_xscale('log')
ax.set_xlabel(r'$\rho$ [g/cm$^3$]', fontsize=14)
ax.set_ylabel(r'$Y_e$', fontsize=14)
ax.set_title(r'Comparison of $Y_e$ vs $\rho$', fontsize=16)
ax.legend()

ax.minorticks_on()

plt.grid(linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('/home/mi58rip/gr-athena/plots/eos/ye_vs_rho_comparison.png', dpi=600)