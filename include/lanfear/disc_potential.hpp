#pragma once

// Disc-adapted basis-function expansion for flattened systems, using the same
// SCF machinery as scf_potential.hpp but a disc-shaped (Miyamoto-Nagai) basis.
//
// The basis members are Miyamoto & Nagai (1975) density-potential pairs of unit
// mass and scales (a, b): a sets the radial scale, b the thickness (a=0 gives a
// Plummer sphere, b->0 a razor-thin Kuzmin disc), so a handful of members span
// shapes from round to thin discs.
//
// Because the basis is *not* orthonormal, the coefficients solve the Galerkin
// system  G c = b  with
//   b_a = sum_k m_k Phi_a(x_k)           (the SCF particle sum, done here)
//   G_ab = integral rho_a Phi_b dV       (fixed Gram matrix, formed in python)
// -G is symmetric positive definite (a mutual-energy matrix), so the small
// solve is done robustly in numpy. This class computes b, stores c, and
// evaluates Phi = sum_a c_a Phi_a and its gradient.
//
// Internal units follow the rest of the code: G = M_field = scale_radius = 1;
// the central black hole is added separately as a spline-softened point mass.

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "black_hole.hpp"
#include "spline_softening.hpp"

namespace lanfear {

// --- Miyamoto-Nagai primitives (unit mass, G = 1) ---------------------------

// Potential of a unit-mass MN component with scales (a, b) at (x, y, z).
inline double mn_potential(double x, double y, double z, double a, double b) {
    const double zeta = std::sqrt(z * z + b * b);
    const double az = a + zeta;
    const double D = std::sqrt(x * x + y * y + az * az);
    return -1.0 / D;
}

// Acceleration -grad(Phi) of a unit-mass MN component.
inline std::array<double, 3> mn_acceleration(double x, double y, double z,
                                             double a, double b) {
    const double zeta = std::sqrt(z * z + b * b);
    const double az = a + zeta;
    const double D2 = x * x + y * y + az * az;
    const double invD3 = 1.0 / (D2 * std::sqrt(D2));
    return {-x * invD3, -y * invD3, -az * z / zeta * invD3};
}

// Density of a unit-mass MN component (MN 1975, eq. 4); R^2 = x^2 + y^2.
inline double mn_density(double R, double z, double a, double b) {
    const double zeta = std::sqrt(z * z + b * b);
    const double az = a + zeta;
    const double num = b * b * (a * R * R + (a + 3.0 * zeta) * az * az);
    const double s = R * R + az * az;
    const double den = 4.0 * M_PI * std::pow(s, 2.5) * zeta * zeta * zeta;
    return num / den;
}

// --- Disc basis-function potential ------------------------------------------

class DiscPotential {
public:
    // Basis scale lists a[i], b[i] (equal length). Coefficients start at zero
    // and are set after solving the Galerkin system.
    DiscPotential(std::vector<double> a, std::vector<double> b)
        : a_(std::move(a)), b_(std::move(b)), coeff_(a_.size(), 0.0) {
        if (a_.size() != b_.size())
            throw std::invalid_argument("a and b must have equal length");
        if (a_.empty()) throw std::invalid_argument("basis must be non-empty");
        for (std::size_t i = 0; i < a_.size(); ++i) {
            if (a_[i] < 0.0 || b_[i] <= 0.0)
                throw std::invalid_argument("require a>=0 and b>0");
        }
    }

    std::size_t size() const { return a_.size(); }
    const std::vector<double>& basis_a() const { return a_; }
    const std::vector<double>& basis_b() const { return b_; }
    const std::vector<double>& coefficients() const { return coeff_; }

    void set_coefficients(std::vector<double> c) {
        if (c.size() != a_.size())
            throw std::invalid_argument("coefficient count mismatch");
        coeff_ = std::move(c);
    }

    // SCF particle sum b_a = sum_k m_k Phi_a(x_k), evaluated in parallel.
    std::vector<double> scf_sum(const std::vector<double>& x,
                               const std::vector<double>& y,
                               const std::vector<double>& z,
                               const std::vector<double>& mass) const {
        const std::size_t np = x.size();
        if (y.size() != np || z.size() != np || mass.size() != np)
            throw std::invalid_argument("position/mass arrays must match");
        const std::size_t M = a_.size();
        std::vector<double> b(M, 0.0);
        #pragma omp parallel for schedule(static)
        for (std::size_t j = 0; j < M; ++j) {
            double acc = 0.0;
            for (std::size_t k = 0; k < np; ++k)
                acc += mass[k] * mn_potential(x[k], y[k], z[k], a_[j], b_[j]);
            b[j] = acc;
        }
        return b;
    }

    void add_black_hole(double mass, double x, double y, double z,
                        double softening) {
        bh_.add_black_hole(mass, x, y, z, softening);
    }
    std::size_t num_black_holes() const { return bh_.num_black_holes(); }
    const std::vector<BlackHole>& black_holes() const { return bh_.black_holes(); }

    double potential(double x, double y, double z) const {
        double p = 0.0;
        for (std::size_t j = 0; j < a_.size(); ++j)
            p += coeff_[j] * mn_potential(x, y, z, a_[j], b_[j]);
        return p + bh_.potential(x, y, z);
    }

    std::array<double, 3> acceleration(double x, double y, double z) const {
        std::array<double, 3> acc{0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < a_.size(); ++j) {
            const auto ai = mn_acceleration(x, y, z, a_[j], b_[j]);
            acc[0] += coeff_[j] * ai[0];
            acc[1] += coeff_[j] * ai[1];
            acc[2] += coeff_[j] * ai[2];
        }
        bh_.add_acceleration(x, y, z, acc);
        return acc;
    }

private:
    std::vector<double> a_, b_, coeff_;
    BlackHoleSet bh_;
};

}  // namespace lanfear
