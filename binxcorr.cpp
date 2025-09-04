// binxcorr.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace py = pybind11;

// Pack ±1 int8 array into little-endian bitstream (1 for +1, 0 for -1)
static std::vector<uint64_t> pack_pm_to_bits(const int8_t* pm, size_t n) {
    const size_t words = (n + 63) >> 6;
    std::vector<uint64_t> out(words, 0ULL);
    for (size_t i = 0; i < n; ++i) {
        if (pm[i] > 0) {
            out[i >> 6] |= (1ULL << (i & 63)); // set bit if +1
        }
    }
    return out;
}

// Load a 64-bit window starting at bit position 'bitpos' from a packed bitstream
static inline uint64_t load64(const std::vector<uint64_t>& w, size_t bitpos) {
    const size_t idx = bitpos >> 6;
    const unsigned shift = static_cast<unsigned>(bitpos & 63);
    if (idx >= w.size()) return 0ULL;
    const uint64_t lo = w[idx];
    const uint64_t hi = (idx + 1 < w.size()) ? w[idx + 1] : 0ULL;
    if (shift == 0) return lo;
    return (lo >> shift) | (hi << (64 - shift));
}

// Sliding binary correlation over ±1 streams using XOR+POPCNT on packed bits.
// signal_pm: ±1 int8 array, pattern_pm: ±1 int8 array
// start_bit/end_bit: ROI in [start_bit, end_bit) over 'signal_pm' (defaults to full length)
py::array_t<float> correlate_sliding_bin(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> signal_pm,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> pattern_pm,
    int64_t start_bit = 0,
    int64_t end_bit   = -1
) {
    // Access buffers
    auto s_buf = signal_pm.request();
    auto p_buf = pattern_pm.request();
    const auto* s_ptr = static_cast<int8_t*>(s_buf.ptr);
    const auto* p_ptr = static_cast<int8_t*>(p_buf.ptr);
    const size_t N = static_cast<size_t>(s_buf.size);
    const size_t L = static_cast<size_t>(p_buf.size);

    if (L == 0 || N == 0 || N < L) {
        return py::array_t<float>(0);
    }

    // ROI clamp
    size_t start = static_cast<size_t>(std::max<int64_t>(0, start_bit));
    size_t end   = static_cast<size_t>(end_bit < 0 ? static_cast<int64_t>(N)
                                                   : std::min<int64_t>(end_bit, static_cast<int64_t>(N)));
    if (end <= start || (end - start) < L) {
        return py::array_t<float>(0);
    }

    const size_t positions = (end - L) - start + 1;

    // Pack both to bitstreams
    std::vector<uint64_t> S = pack_pm_to_bits(s_ptr, N);
    std::vector<uint64_t> P = pack_pm_to_bits(p_ptr, L);

    const size_t W = (L + 63) >> 6; // number of 64-bit words to cover L bits

    // Prepare output
    py::array_t<float> out(positions);
    auto out_buf = out.request();
    float* dst = static_cast<float*>(out_buf.ptr);

    // Precompute masks per word (last word may be partial)
    std::vector<uint64_t> masks(W, ~0ULL);
    if ((L & 63) != 0) {
        const unsigned rem = static_cast<unsigned>(L & 63);
        masks[W - 1] = (rem == 64 ? ~0ULL : ((1ULL << rem) - 1ULL));
    }

    // Main loop: for each slide position, XOR against pattern words and popcount mismatches
    for (size_t pos = 0; pos < positions; ++pos) {
        const size_t base = start + pos;
        int mismatches = 0;

        // XOR+POPCNT per 64-bit chunk
        for (size_t k = 0; k < W; ++k) {
            const size_t bitpos = base + (k << 6); // base + 64*k
            const uint64_t sigw = load64(S, bitpos);
            const uint64_t x = (sigw ^ P[k]) & masks[k];
#if defined(__GNUC__) || defined(__clang__)
            mismatches += __builtin_popcountll(x);
#else
            // Fallback popcount for MSVC (C++20 has std::popcount)
            x = x - ((x >> 1) & 0x5555555555555555ULL);
            x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
            mismatches += static_cast<int>((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
#endif
        }

        // Map Hamming mismatches to normalized correlation in [-1, 1]
        dst[pos] = 1.0f - 2.0f * (static_cast<float>(mismatches) / static_cast<float>(L));
    }

    return out;
}

// IMPORTANT: The module name here **must** be exactly 'binxcorr'
PYBIND11_MODULE(binxcorr, m) {
    m.doc() = "Fast binary sliding correlation (XOR+POPCNT) for ±1 streams";
    m.def("correlate_sliding_bin", &correlate_sliding_bin,
          py::arg("signal_pm"), py::arg("pattern_pm"),
          py::arg("start_bit") = 0, py::arg("end_bit") = -1,
          "Compute normalized sliding correlation between a ±1 signal and pattern.\n"
          "Returns a float array of length (ROI_len - L + 1).");
}
