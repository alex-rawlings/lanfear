#pragma once

// Special functions used by the self-consistent-field (SCF) potential
// expansion. Kept in one small header so the physics in scf_potential.hpp reads
// close to the equations in Hernquist & Ostriker (1992, ApJ 386, 375).

#include <cmath>

#include <boost/math/special_functions/gegenbauer.hpp>

namespace lanfear {

// Gegenbauer (ultraspherical) polynomial C_n^{(alpha)}(x). The radial basis of
// the HO expansion uses alpha = 2l + 3/2.
inline double gegenbauer(int n, double alpha, double x) {
    if (n < 0) return 0.0;
    return boost::math::gegenbauer(static_cast<unsigned>(n), alpha, x);
}

// Associated Legendre function P_l^m(x) *including* the Condon-Shortley phase
// (-1)^m. The C++ standard library omits that phase whereas the HO formulae
// assume it, so we reinstate it here. Using std::assoc_legendre is markedly
// faster than the Boost equivalent.
inline double assoc_legendre(int l, int m, double x) {
    if (m > l) return 0.0;
    const double phase = (m % 2 == 0) ? 1.0 : -1.0;
    return phase * std::assoc_legendre(static_cast<unsigned>(l),
                                       static_cast<unsigned>(m), x);
}

// Derivative dP_l^m/dx of the associated Legendre function, from the recurrence
//   (x^2 - 1) dP_l^m/dx = l x P_l^m - (l + m) P_{l-1}^m
// (DLMF 14.10.3). This uses only same-m terms, so it is automatically phase-
// consistent with assoc_legendre above regardless of the Condon-Shortley
// convention -- unlike the P_l^{m+1} form, which is easy to get wrong. Guarded
// against the coordinate poles at x = +/-1, where the meridional derivative is
// handled separately. P_{l-1}^m vanishes when m > l-1, which assoc_legendre
// returns correctly.
inline double assoc_legendre_deriv(int l, int m, double x) {
    if (l == 0) return 0.0;
    const double denom = x * x - 1.0;
    if (std::abs(denom) < 1e-15) return 0.0;
    const double Plm = assoc_legendre(l, m, x);
    const double Plm1 = assoc_legendre(l - 1, m, x);  // P_{l-1}^m
    return (l * x * Plm - (l + m) * Plm1) / denom;
}

}  // namespace lanfear
