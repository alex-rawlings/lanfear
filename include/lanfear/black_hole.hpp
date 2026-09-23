#pragma once

// Shared softened point-mass (black hole) storage and field evaluation, used
// by every potential class (SCFPotential, DiscPotential, CompositePotential).
// Each class holds a BlackHoleSet member and delegates add_black_hole/
// num_black_holes/black_holes to it, and folds its potential()/acceleration()
// contribution into its own field sum -- this was previously duplicated
// (identical struct + loop) in scf_potential.hpp and disc_potential.hpp.

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "spline_softening.hpp"

namespace lanfear {

struct BlackHole {
    double mass;                 // in units of the owning potential's mass unit
    std::array<double, 3> pos;   // in units of the owning potential's length unit
    double softening;            // spline (Gadget4) softening length, same length unit
};

class BlackHoleSet {
public:
    void add_black_hole(double mass, double x, double y, double z,
                        double softening) {
        if (mass < 0.0) throw std::invalid_argument("BH mass must be non-negative");
        if (softening < 0.0)
            throw std::invalid_argument("BH softening must be non-negative");
        black_holes_.push_back({mass, {x, y, z}, softening});
    }
    std::size_t num_black_holes() const { return black_holes_.size(); }
    const std::vector<BlackHole>& black_holes() const { return black_holes_; }

    // Summed softened point-mass potential at (x, y, z).
    double potential(double x, double y, double z) const {
        double p = 0.0;
        for (const auto& bh : black_holes_) {
            const double dx = x - bh.pos[0];
            const double dy = y - bh.pos[1];
            const double dz = z - bh.pos[2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            p += bh.mass * spline_softened_potential(r, bh.softening);
        }
        return p;
    }

    // Adds the summed softened point-mass acceleration at (x, y, z) into acc.
    void add_acceleration(double x, double y, double z,
                          std::array<double, 3>& acc) const {
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
    }

private:
    std::vector<BlackHole> black_holes_;
};

}  // namespace lanfear
