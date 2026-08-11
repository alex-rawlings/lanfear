#pragma once

// Frequency analysis for orbit classification: a self-contained radix-2 FFT and
// a NAFF-style spectral line extractor (Laskar 1990; Valluri & Merritt 1998).
//
// NAFF recovers the leading complex-exponential components of a time series to
// far better than the FFT bin resolution by (1) locating each peak with a
// Hann-windowed FFT, (2) refining the frequency by maximising the windowed
// projection |<f, e^{i w t}>| with a golden-section search, and (3) subtracting
// the recovered component before finding the next. For orbit classification the
// signal analysed per principal axis is the complex w_k = x_k + i v_k / w0,
// whose leading line is the fundamental frequency of that axis (signed, so
// prograde/retrograde loops are distinguishable).

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace lanfear {

using cdouble = std::complex<double>;

// In-place iterative radix-2 Cooley-Tukey FFT; a.size() must be a power of two.
inline void fft(std::vector<cdouble>& a, bool invert) {
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * M_PI / len * (invert ? 1.0 : -1.0);
        const cdouble wlen(std::cos(ang), std::sin(ang));
        for (std::size_t i = 0; i < n; i += len) {
            cdouble w(1.0, 0.0);
            for (std::size_t k = 0; k < len / 2; ++k) {
                const cdouble u = a[i + k];
                const cdouble v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert)
        for (auto& x : a) x /= static_cast<double>(n);
}

struct SpectralLine {
    double frequency = 0.0;  // signed angular frequency (rad / time unit)
    double amplitude = 0.0;  // |a|
    double phase = 0.0;      // arg(a)
};

namespace detail {

inline std::size_t largest_pow2_le(std::size_t n) {
    std::size_t p = 1;
    while ((p << 1) <= n) p <<= 1;
    return p;
}

// Windowed projection <f, e^{i w t}> = sum_j chi_j f_j e^{-i w t_j}.
inline cdouble projection(const std::vector<cdouble>& f,
                          const std::vector<double>& chi, double dt,
                          double omega) {
    cdouble acc(0.0, 0.0);
    for (std::size_t j = 0; j < f.size(); ++j) {
        const double ph = -omega * (j * dt);
        acc += chi[j] * f[j] * cdouble(std::cos(ph), std::sin(ph));
    }
    return acc;
}

}  // namespace detail

// Extract up to n_lines leading spectral lines from complex signal `signal`
// (uniformly sampled at spacing dt), sorted by amplitude (largest first).
inline std::vector<SpectralLine> naff(std::vector<cdouble> signal, double dt,
                                      int n_lines) {
    // Work on the largest power-of-two prefix.
    const std::size_t N = detail::largest_pow2_le(signal.size());
    signal.resize(N);
    std::vector<SpectralLine> lines;
    if (N < 4 || n_lines <= 0) return lines;

    // Hann window and its sum (for the projection normalisation).
    std::vector<double> chi(N);
    double chi_sum = 0.0;
    for (std::size_t j = 0; j < N; ++j) {
        chi[j] = 0.5 * (1.0 - std::cos(2.0 * M_PI * j / (N - 1)));
        chi_sum += chi[j];
    }
    if (chi_sum <= 0.0) return lines;

    const double dw = 2.0 * M_PI / (N * dt);  // FFT bin width in angular freq

    for (int line = 0; line < n_lines; ++line) {
        // (1) Windowed FFT -> coarse peak bin.
        std::vector<cdouble> spec(N);
        for (std::size_t j = 0; j < N; ++j) spec[j] = signal[j] * chi[j];
        fft(spec, false);
        std::size_t kmax = 0;
        double best = -1.0;
        for (std::size_t k = 0; k < N; ++k) {
            const double m = std::norm(spec[k]);
            if (m > best) { best = m; kmax = k; }
        }
        // Signed coarse angular frequency (wrap bins above Nyquist to negative).
        const double k_signed =
            (kmax <= N / 2) ? static_cast<double>(kmax)
                            : static_cast<double>(kmax) - static_cast<double>(N);
        const double omega_coarse = k_signed * dw;

        // (2) Golden-section refine of |projection| in [coarse-dw, coarse+dw].
        double a = omega_coarse - dw, b = omega_coarse + dw;
        const double gr = (std::sqrt(5.0) - 1.0) / 2.0;
        double c = b - gr * (b - a), d = a + gr * (b - a);
        double fc = std::abs(detail::projection(signal, chi, dt, c));
        double fd = std::abs(detail::projection(signal, chi, dt, d));
        for (int it = 0; it < 60; ++it) {
            if (fc > fd) {
                b = d; d = c; fd = fc;
                c = b - gr * (b - a);
                fc = std::abs(detail::projection(signal, chi, dt, c));
            } else {
                a = c; c = d; fc = fd;
                d = a + gr * (b - a);
                fd = std::abs(detail::projection(signal, chi, dt, d));
            }
            if (b - a < 1e-12 * (std::abs(a) + std::abs(b) + 1e-12)) break;
        }
        const double omega = 0.5 * (a + b);

        // (3) Amplitude via normalised windowed projection, then subtract.
        const cdouble amp = detail::projection(signal, chi, dt, omega) / chi_sum;
        for (std::size_t j = 0; j < N; ++j) {
            const double ph = omega * (j * dt);
            signal[j] -= amp * cdouble(std::cos(ph), std::sin(ph));
        }
        lines.push_back({omega, std::abs(amp), std::arg(amp)});
    }

    std::sort(lines.begin(), lines.end(),
              [](const SpectralLine& p, const SpectralLine& q) {
                  return p.amplitude > q.amplitude;
              });
    return lines;
}

}  // namespace lanfear
