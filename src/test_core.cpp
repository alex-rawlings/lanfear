// Standalone C++ sanity check (no python): sample a Hernquist sphere, build the
// SCF expansion, and compare against the analytic Hernquist potential/force.
// In G = M = a = 1 units the Hernquist model has
//   Phi(r)   = -1 / (1 + r)
//   |a_r|(r) =  1 / (1 + r)^2
//   M(<r)    = r^2 / (1 + r)^2

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "lanfear/scf_potential.hpp"

int main() {
    const int N = 200000;
    const int n_max = 12;
    const int l_max = 4;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<double> x(N), y(N), z(N), mass(N, 1.0 / N);
    for (int i = 0; i < N; ++i) {
        // Invert M(<r)/M = r^2/(1+r)^2  ->  r = sqrt(u)/(1 - sqrt(u)).
        const double su = std::sqrt(uni(rng));
        const double r = su / (1.0 - su);
        const double mu = 2.0 * uni(rng) - 1.0;        // cos(theta), isotropic
        const double ph = 2.0 * M_PI * uni(rng);
        const double st = std::sqrt(std::max(0.0, 1.0 - mu * mu));
        x[i] = r * st * std::cos(ph);
        y[i] = r * st * std::sin(ph);
        z[i] = r * mu;
    }

    lanfear::SCFPotential pot(n_max, l_max, x, y, z, mass);

    printf("  r        Phi_scf      Phi_exact    relerr     a_scf      a_exact\n");
    double max_pot_err = 0.0, max_acc_err = 0.0;
    for (double r : {0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0}) {
        const double phi_scf = pot.potential(r, 0.0, 0.0);
        const double phi_exact = -1.0 / (1.0 + r);
        const auto a = pot.acceleration(r, 0.0, 0.0);
        const double a_scf = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        const double a_exact = 1.0 / ((1.0 + r) * (1.0 + r));
        const double perr = std::abs((phi_scf - phi_exact) / phi_exact);
        const double aerr = std::abs((a_scf - a_exact) / a_exact);
        max_pot_err = std::max(max_pot_err, perr);
        max_acc_err = std::max(max_acc_err, aerr);
        printf("%6.2f  %11.5f  %11.5f  %8.2e  %9.5f  %9.5f\n", r, phi_scf,
               phi_exact, perr, a_scf, a_exact);
    }
    printf("\nmax relative error: potential %.2e, acceleration %.2e\n",
           max_pot_err, max_acc_err);

    // Add an off-centre BH and confirm it perturbs the force near it.
    pot.add_black_hole(0.05, 0.5, 0.0, 0.0, 1e-2);
    const auto a_near = pot.acceleration(0.55, 0.0, 0.0);
    printf("with off-centre BH at (0.5,0,0): a(0.55,0,0) = (%.4f, %.4f, %.4f)\n",
           a_near[0], a_near[1], a_near[2]);

    const bool ok = (max_pot_err < 0.02) && (max_acc_err < 0.05);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
