#pragma once

// Direct-summation potential of softened point masses, the reference that the
// basis expansions are validated against.

#include <cmath>
#include <cstddef>

#include "spline_softening.hpp"

namespace lanfear {

// Potential at (x, y, z) from softened point masses at (sx[k], sy[k], sz[k])
// with mass sm[k], k = 0..n_sources-1, spline-softened (Gadget4 kernel, see
// spline_softening.hpp) with softening length `softening`.
inline double direct_potential(double x, double y, double z, const double* sx,
                               const double* sy, const double* sz,
                               const double* sm, std::size_t n_sources,
                               double softening) {
    double phi = 0.0;
    for (std::size_t k = 0; k < n_sources; ++k) {
        const double dx = x - sx[k];
        const double dy = y - sy[k];
        const double dz = z - sz[k];
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        phi += sm[k] * spline_softened_potential(r, softening);
    }
    return phi;
}

}  // namespace lanfear
