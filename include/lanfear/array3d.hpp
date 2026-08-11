#pragma once

// Minimal contiguous 3D array used to hold the SCF coefficient tables indexed
// by (n, l, m). Row-major, zero-initialised.

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace lanfear {

template <typename T>
class Array3D {
    std::size_t d0_, d1_, d2_;
    std::vector<T> data_;

public:
    Array3D() : d0_(0), d1_(0), d2_(0) {}
    Array3D(std::size_t d0, std::size_t d1, std::size_t d2)
        : d0_(d0), d1_(d1), d2_(d2), data_(d0 * d1 * d2, T{}) {}

    T& operator()(std::size_t i, std::size_t j, std::size_t k) {
        return data_[(i * d1_ + j) * d2_ + k];
    }
    const T& operator()(std::size_t i, std::size_t j, std::size_t k) const {
        return data_[(i * d1_ + j) * d2_ + k];
    }

    std::size_t dim0() const { return d0_; }
    std::size_t dim1() const { return d1_; }
    std::size_t dim2() const { return d2_; }
    std::size_t size() const { return data_.size(); }

    // Access to the underlying row-major (i,j,k) buffer, for serialisation.
    const std::vector<T>& flat() const { return data_; }
    void load_flat(const std::vector<T>& src) {
        if (src.size() != data_.size())
            throw std::invalid_argument("Array3D::load_flat size mismatch");
        data_ = src;
    }
};

}  // namespace lanfear
