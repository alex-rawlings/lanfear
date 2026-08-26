#pragma once

// Self-consistent-field (SCF) potential using the Hernquist & Ostriker (1992,
// ApJ 386, 375; "HO92") biorthonormal basis. This basis is well suited to
// spherical-ish (mildly triaxial) systems. Equation numbers in the comments
// refer to HO92.
//
// Internal units follow HO92: G = 1, the total *field* mass = 1 and the scale
// radius = 1. The caller is responsible for normalising positions by the scale
// radius and masses by the total field mass before constructing the potential;
// the python layer does this.
//
// The central black hole is deliberately *not* part of the basis expansion: an
// HO expansion cannot represent a point mass. Instead the field (everything
// except the BH) is expanded here, and the BH is added afterwards as a
// spline-softened point mass at an arbitrary position via add_black_hole().

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "array3d.hpp"
#include "special_functions.hpp"
#include "spline_softening.hpp"

namespace lanfear {

struct BlackHole {
    double mass;                 // in units of the total field mass
    std::array<double, 3> pos;   // in units of the scale radius
    double softening;            // spline (Gadget4) softening length, scale-radius units
};

class SCFPotential {
public:
    // Build the expansion from field particles (BH already removed). Positions
    // and masses are expected in HO units (see header note).
    SCFPotential(int n_max, int l_max, const std::vector<double>& x,
                 const std::vector<double>& y, const std::vector<double>& z,
                 const std::vector<double>& mass);

    // Reconstruct directly from serialised coefficient tables (flat, size
    // (n_max+1)*(l_max+1)*(l_max+1), row-major (n,l,m)). Skips the particle
    // sum; used to broadcast an already-built potential across MPI ranks. Black
    // holes are re-added by the caller.
    SCFPotential(int n_max, int l_max, const std::vector<double>& s_cos,
                 const std::vector<double>& s_sin);

    int n_max() const { return n_max_; }
    int l_max() const { return l_max_; }

    // Flat copies of the particle coefficient tables (for serialisation).
    std::vector<double> coefficients_cos() const { return s_cos_.flat(); }
    std::vector<double> coefficients_sin() const { return s_sin_.flat(); }

    // Add a spline-softened point mass at an arbitrary position (HO units).
    void add_black_hole(double mass, double x, double y, double z,
                        double softening);
    std::size_t num_black_holes() const { return black_holes_.size(); }
    const std::vector<BlackHole>& black_holes() const { return black_holes_; }

    // Potential and acceleration at a Cartesian point (HO units).
    double potential(double x, double y, double z) const;
    std::array<double, 3> acceleration(double x, double y, double z) const;

private:
    int n_max_, l_max_;  // inclusive maxima; loops run 0..n_max_ etc.
    Array3D<double> s_cos_;  // C0(n,l,m): particle cos-sum, HO92 Eq. 3.17
    Array3D<double> s_sin_;  // D0(n,l,m): particle sin-sum
    std::vector<double> A_nl_;   // normalisation table, HO92 Eq. 2.31 (inverse)
    std::vector<double> N_lm_;   // normalisation table, HO92 Eq. 3.15
    std::vector<BlackHole> black_holes_;

    double& A_nl(int n, int l) { return A_nl_[n * (l_max_ + 1) + l]; }
    double A_nl(int n, int l) const { return A_nl_[n * (l_max_ + 1) + l]; }
    double& N_lm(int l, int m) { return N_lm_[l * (l_max_ + 1) + m]; }
    double N_lm(int l, int m) const { return N_lm_[l * (l_max_ + 1) + m]; }

    // Fill the A_nl / N_lm normalisation tables (deterministic in n_max,l_max).
    void init_norm_tables();

    // Radial basis function and its derivative (HO92 Eq. 2.25 / 3.29). Used at
    // construction; the hot evaluation path uses fill_radial() instead.
    static double phi_tilde(double r, int n, int l);
    static double dphi_tilde_dr(double r, int n, int l);
    static double xi(double r) { return (r - 1.0) / (r + 1.0); }

    // Tabulate Phi_tilde_nl(r) (and, if deriv, dPhi_tilde_nl/dr) for all (n,l)
    // at radius r into row-major [l*(n_max_+1)+n] buffers, using the Gegenbauer
    // recurrence in n. This replaces O(n_max^2 * l_max) per-term boost calls
    // with O(n_max * l_max) arithmetic -- the key performance path for orbit
    // integration.
    void fill_radial(double r, bool deriv, std::vector<double>& phit,
                     std::vector<double>& dphit) const;

    // Convert Cartesian -> (r, theta, phi).
    static std::array<double, 3> to_spherical(double x, double y, double z);
};

// --- basis functions --------------------------------------------------------

inline double SCFPotential::phi_tilde(double r, int n, int l) {
    // HO92 Eq. 2.25 radial part (with the sqrt(4 pi) normalisation carried
    // consistently through to the coefficient sums).
    return -std::sqrt(4.0 * M_PI) * std::pow(r, l) /
           std::pow(1.0 + r, 2 * l + 1) *
           gegenbauer(n, 2.0 * l + 1.5, xi(r));
}

inline double SCFPotential::dphi_tilde_dr(double r, int n, int l) {
    // HO92 Eq. 3.29. Uses dC_n^{a}/dxi = 2a C_{n-1}^{a+1} and dxi/dr =
    // 2/(1+r)^2.
    const double x = xi(r);
    const double base = phi_tilde(r, n, l);
    const double c_n = gegenbauer(n, 2.0 * l + 1.5, x);
    const double c_nm1 = (n > 0) ? gegenbauer(n - 1, 2.0 * l + 2.5, x) : 0.0;
    const double ratio = (c_n != 0.0) ? c_nm1 / c_n : 0.0;
    return base * (static_cast<double>(l) / r - (2.0 * l + 1.0) / (1.0 + r) +
                   4.0 * (2.0 * l + 1.5) / std::pow(1.0 + r, 2) * ratio);
}

inline std::array<double, 3> SCFPotential::to_spherical(double x, double y,
                                                        double z) {
    const double r = std::sqrt(x * x + y * y + z * z);
    const double theta = (r > 0.0) ? std::acos(z / r) : 0.0;
    const double phi = std::atan2(y, x);
    return {r, theta, phi};
}

// --- construction -----------------------------------------------------------

inline void SCFPotential::init_norm_tables() {
    // Normalisation tables (HO92 Eq. 2.23, 2.31, 3.15).
    for (int n = 0; n <= n_max_; ++n) {
        for (int l = 0; l <= l_max_; ++l) {
            const double K_nl =
                0.5 * n * (n + 4.0 * l + 3.0) + (l + 1.0) * (2.0 * l + 1.0);
            const double num = std::pow(2.0, 8.0 * l + 6.0) * std::tgamma(n + 1.0) *
                               (n + 2.0 * l + 1.5) *
                               std::pow(std::tgamma(2.0 * l + 1.5), 2.0);
            const double den = 4.0 * M_PI * K_nl * std::tgamma(n + 4.0 * l + 3.0);
            A_nl(n, l) = -num / den;
        }
    }
    for (int l = 0; l <= l_max_; ++l) {
        for (int m = 0; m <= l; ++m) {
            const double dm0 = (m == 0) ? 1.0 : 2.0;
            N_lm(l, m) = (2.0 * l + 1.0) / (4.0 * M_PI) * dm0 *
                         std::tgamma(l - m + 1.0) / std::tgamma(l + m + 1.0);
        }
    }
}

inline SCFPotential::SCFPotential(int n_max, int l_max,
                                  const std::vector<double>& s_cos,
                                  const std::vector<double>& s_sin)
    : n_max_(n_max),
      l_max_(l_max),
      s_cos_(n_max + 1, l_max + 1, l_max + 1),
      s_sin_(n_max + 1, l_max + 1, l_max + 1),
      A_nl_((n_max + 1) * (l_max + 1), 0.0),
      N_lm_((l_max + 1) * (l_max + 1), 0.0) {
    if (n_max < 0 || l_max < 0)
        throw std::invalid_argument("n_max and l_max must be non-negative");
    s_cos_.load_flat(s_cos);
    s_sin_.load_flat(s_sin);
    init_norm_tables();
}

inline SCFPotential::SCFPotential(int n_max, int l_max,
                                  const std::vector<double>& x,
                                  const std::vector<double>& y,
                                  const std::vector<double>& z,
                                  const std::vector<double>& mass)
    : n_max_(n_max),
      l_max_(l_max),
      s_cos_(n_max + 1, l_max + 1, l_max + 1),
      s_sin_(n_max + 1, l_max + 1, l_max + 1),
      A_nl_((n_max + 1) * (l_max + 1), 0.0),
      N_lm_((l_max + 1) * (l_max + 1), 0.0) {
    if (n_max < 0 || l_max < 0)
        throw std::invalid_argument("n_max and l_max must be non-negative");
    const std::size_t np = x.size();
    if (y.size() != np || z.size() != np || mass.size() != np)
        throw std::invalid_argument("position/mass arrays must match in length");

    init_norm_tables();

    // Precompute per-particle spherical coordinates.
    std::vector<double> r(np), cos_theta(np), phi(np);
    for (std::size_t k = 0; k < np; ++k) {
        const auto rtp = to_spherical(x[k], y[k], z[k]);
        r[k] = rtp[0];
        cos_theta[k] = std::cos(rtp[1]);
        phi[k] = rtp[2];
    }

    // Particle contribution to the coefficients (HO92 Eq. 3.17). The n,l,m
    // triple loop is the outer parallel dimension; each thread reduces over
    // particles into its own (n,l,m) cell, so no races.
    const int L = l_max_;
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int n = 0; n <= n_max_; ++n) {
        for (int l = 0; l <= L; ++l) {
            for (int m = 0; m <= l; ++m) {
                double acc_cos = 0.0, acc_sin = 0.0;
                for (std::size_t k = 0; k < np; ++k) {
                    const double prefac = mass[k] * phi_tilde(r[k], n, l) *
                                          assoc_legendre(l, m, cos_theta[k]);
                    const double mphi = m * phi[k];
                    acc_cos += prefac * std::cos(mphi);
                    acc_sin += prefac * std::sin(mphi);
                }
                s_cos_(n, l, m) = acc_cos;
                s_sin_(n, l, m) = acc_sin;
            }
        }
    }
}

inline void SCFPotential::add_black_hole(double mass, double x, double y,
                                         double z, double softening) {
    if (mass < 0.0) throw std::invalid_argument("BH mass must be non-negative");
    if (softening < 0.0)
        throw std::invalid_argument("BH softening must be non-negative");
    black_holes_.push_back({mass, {x, y, z}, softening});
}

// --- field evaluation -------------------------------------------------------

inline void SCFPotential::fill_radial(double r, bool deriv,
                                      std::vector<double>& phit,
                                      std::vector<double>& dphit) const {
    const int Nn = n_max_ + 1;
    phit.assign(static_cast<std::size_t>(l_max_ + 1) * Nn, 0.0);
    if (deriv) dphit.assign(static_cast<std::size_t>(l_max_ + 1) * Nn, 0.0);

    const double sq4pi = std::sqrt(4.0 * M_PI);
    const double x = xi(r);
    const double rp1 = 1.0 + r;

    // Per-l Gegenbauer tables (reused across calls to avoid reallocation).
    static thread_local std::vector<double> C, Cp;
    C.resize(Nn);
    Cp.resize(Nn);

    for (int l = 0; l <= l_max_; ++l) {
        const double alpha = 2.0 * l + 1.5;
        // C_n^{alpha}(x) via the standard recurrence.
        C[0] = 1.0;
        if (Nn > 1) C[1] = 2.0 * alpha * x;
        for (int n = 2; n < Nn; ++n)
            C[n] = (2.0 * x * (n + alpha - 1.0) * C[n - 1] -
                    (n + 2.0 * alpha - 2.0) * C[n - 2]) / n;

        if (deriv) {
            const double a2 = alpha + 1.0;  // 2l + 5/2
            Cp[0] = 1.0;
            if (Nn > 1) Cp[1] = 2.0 * a2 * x;
            for (int n = 2; n < Nn; ++n)
                Cp[n] = (2.0 * x * (n + a2 - 1.0) * Cp[n - 1] -
                         (n + 2.0 * a2 - 2.0) * Cp[n - 2]) / n;
        }

        const double prefac =
            -sq4pi * std::pow(r, l) / std::pow(rp1, 2 * l + 1);
        const int base = l * Nn;
        for (int n = 0; n < Nn; ++n) {
            const double pt = prefac * C[n];
            phit[base + n] = pt;
            if (deriv) {
                const double ratio = (n > 0 && C[n] != 0.0) ? Cp[n - 1] / C[n] : 0.0;
                dphit[base + n] =
                    pt * (static_cast<double>(l) / r - (2.0 * l + 1.0) / rp1 +
                          4.0 * alpha / (rp1 * rp1) * ratio);
            }
        }
    }
}

inline double SCFPotential::potential(double x, double y, double z) const {
    const auto rtp = to_spherical(x, y, z);
    const double r = rtp[0];
    const double cos_theta = std::cos(rtp[1]);
    const double phi = rtp[2];
    const int Nn = n_max_ + 1;

    static thread_local std::vector<double> phit, dphit, w;
    fill_radial(r, false, phit, dphit);
    w.resize(Nn);

    double pot = 0.0;
    for (int l = 0; l <= l_max_; ++l) {
        const int base = l * Nn;
        for (int n = 0; n < Nn; ++n) w[n] = A_nl(n, l) * phit[base + n];
        for (int m = 0; m <= l; ++m) {
            // C_lm(r), D_lm(r): radial sum over n, HO92 Eq. 3.13.
            double C = 0.0, D = 0.0;
            for (int n = 0; n < Nn; ++n) {
                C += w[n] * s_cos_(n, l, m);
                D += w[n] * s_sin_(n, l, m);
            }
            C *= N_lm(l, m);
            D *= N_lm(l, m);
            const double mphi = m * phi;
            pot += assoc_legendre(l, m, cos_theta) *
                   (C * std::cos(mphi) + D * std::sin(mphi));
        }
    }

    // Spline-softened point-mass contribution(s).
    for (const auto& bh : black_holes_) {
        const double dx = x - bh.pos[0];
        const double dy = y - bh.pos[1];
        const double dz = z - bh.pos[2];
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        pot += bh.mass * spline_softened_potential(r, bh.softening);
    }
    return pot;
}

inline std::array<double, 3> SCFPotential::acceleration(double x, double y,
                                                        double z) const {
    const auto rtp = to_spherical(x, y, z);
    const double r = rtp[0];
    const double theta = rtp[1];
    const double phi = rtp[2];
    const double sin_theta = std::sin(theta);
    const double cos_theta = std::cos(theta);
    const double cos_phi = std::cos(phi);
    const double sin_phi = std::sin(phi);

    const int Nn = n_max_ + 1;
    static thread_local std::vector<double> phit, dphit, w, wd;
    double a_r = 0.0, a_theta = 0.0, a_phi = 0.0;
    if (r > 0.0) {
        fill_radial(r, true, phit, dphit);
        w.resize(Nn);
        wd.resize(Nn);
        for (int l = 0; l <= l_max_; ++l) {
            const int base = l * Nn;
            for (int n = 0; n < Nn; ++n) {
                const double a = A_nl(n, l);
                w[n] = a * phit[base + n];
                wd[n] = a * dphit[base + n];
            }
            for (int m = 0; m <= l; ++m) {
                double C = 0.0, D = 0.0, E = 0.0, F = 0.0;
                for (int n = 0; n < Nn; ++n) {
                    C += w[n] * s_cos_(n, l, m);
                    D += w[n] * s_sin_(n, l, m);
                    E += wd[n] * s_cos_(n, l, m);
                    F += wd[n] * s_sin_(n, l, m);
                }
                const double nlm = N_lm(l, m);
                C *= nlm; D *= nlm; E *= nlm; F *= nlm;
                const double cmp = std::cos(m * phi);
                const double smp = std::sin(m * phi);
                const double Plm = assoc_legendre(l, m, cos_theta);
                // HO92 Eq. 3.21-3.23 (spherical acceleration components).
                a_r -= Plm * (E * cmp + F * smp);
                a_theta += assoc_legendre_deriv(l, m, cos_theta) *
                           (C * cmp + D * smp);
                a_phi += m * Plm * (D * cmp - C * smp);
            }
        }
        // a_theta = -(1/r) dPhi/dtheta; with dP/dtheta = -sin(theta) dP/dx this
        // gives +sin(theta)/r times the accumulated dP/dx sum.
        a_theta *= sin_theta / r;
        // a_phi's numerator vanishes on the symmetry axis (P_l^m(+/-1)=0 for
        // m>0); guard the 1/sin(theta) so the 0/0 stays finite.
        a_phi = (std::abs(sin_theta) > 1e-12) ? a_phi / (-r * sin_theta) : 0.0;
    }

    // Cartesian projection (HO92 Eq. 3.18-3.20).
    std::array<double, 3> acc;
    acc[0] = sin_theta * cos_phi * a_r + cos_theta * cos_phi * a_theta -
             sin_phi * a_phi;
    acc[1] = sin_theta * sin_phi * a_r + cos_theta * sin_phi * a_theta +
             cos_phi * a_phi;
    acc[2] = cos_theta * a_r - sin_theta * a_theta;

    // Spline-softened point-mass contribution(s).
    for (const auto& bh : black_holes_) {
        const double dx = x - bh.pos[0];
        const double dy = y - bh.pos[1];
        const double dz = z - bh.pos[2];
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double fac = bh.mass * spline_softened_force_factor(r, bh.softening);
        acc[0] -= fac * dx;
        acc[1] -= fac * dy;
        acc[2] -= fac * dz;
    }
    return acc;
}

}  // namespace lanfear
