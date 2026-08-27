#pragma once

// Cubic-spline gravitational softening kernel, as used in Gadget2/3/4
// (Springel 2005; Springel et al. 2021, "GADGET-4"), originally the
// Monaghan & Lattanzio (1985) SPH spline applied to gravity (Hernquist &
// Katz 1989). The softening length `h` is the compact-support radius: for
// r >= h the potential/force are exactly Newtonian, and for r < h they are
// softened smoothly to a finite value at r = 0.

#include <cmath>

namespace lanfear {

// Softened potential of a unit point mass at separation r, softening length
// h (equal to -1/r for r >= h; h <= 0 disables softening entirely).
inline double spline_softened_potential(double r, double h) {
    if (h <= 0.0) return (r > 0.0) ? -1.0 / r : 0.0;
    if (r >= h) return -1.0 / r;
    const double h_inv = 1.0 / h;
    const double u = r * h_inv;
    if (u < 0.5) {
        return h_inv * (-2.8 + u * u * (5.3333333333333333 +
                                        u * u * (6.4 * u - 9.6)));
    }
    return h_inv * (-3.2 + 0.0666666666666667 / u +
                     u * u * (10.6666666666666667 +
                              u * (-16.0 + u * (9.6 - 2.1333333333333333 * u))));
}

// Softened "1/r^3" force factor: the acceleration contribution of a unit
// point mass at separation vector (dx, dy, dz) with |r| = r is
// -factor * (dx, dy, dz) (h <= 0 disables softening entirely).
inline double spline_softened_force_factor(double r, double h) {
    if (h <= 0.0) return (r > 0.0) ? 1.0 / (r * r * r) : 0.0;
    if (r >= h) return 1.0 / (r * r * r);
    const double h_inv = 1.0 / h;
    const double h3_inv = h_inv * h_inv * h_inv;
    const double u = r * h_inv;
    if (u < 0.5) {
        return h3_inv * (10.6666666666666667 + u * u * (32.0 * u - 38.4));
    }
    return h3_inv * (21.3333333333333333 - 48.0 * u + 38.4 * u * u -
                      10.6666666666666667 * u * u * u -
                      0.0666666666666667 / (u * u * u));
}

}  // namespace lanfear
