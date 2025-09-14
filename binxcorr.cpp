// binxcorr.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <limits>

namespace py = pybind11;

// -------------------------- helpers --------------------------

// Pack ±1 int8 array into little-endian bitstream (1 for +1, 0 for -1)
static std::vector<uint64_t> pack_pm_to_bits(const int8_t* pm, size_t n) {
    const size_t words = (n + 63) >> 6;           // ceil(n/64)
    std::vector<uint64_t> out(words, 0ULL);
    for (size_t i = 0; i < n; ++i) {
        if (pm[i] > 0) {
            out[i >> 6] |= (1ULL << (i & 63));    // set bit if +1
        }
    }
    return out;
}

// Load a 64-bit window from a packed std::vector<uint64_t>
static inline uint64_t load64_vec(const std::vector<uint64_t>& w, size_t bitpos) {
    const size_t idx = bitpos >> 6;
    const unsigned shift = static_cast<unsigned>(bitpos & 63);
    if (idx >= w.size()) return 0ULL;
    const uint64_t lo = w[idx];
    const uint64_t hi = (idx + 1 < w.size()) ? w[idx + 1] : 0ULL;
    return (shift == 0) ? lo : ((lo >> shift) | (hi << (64 - shift)));
}

// Load a 64-bit window from a raw pointer to uint64_t words
static inline uint64_t load64_ptr(const uint64_t* w, size_t n_words, size_t bitpos) {
    const size_t idx = bitpos >> 6;
    const unsigned shift = static_cast<unsigned>(bitpos & 63);
    if (idx >= n_words) return 0ULL;
    const uint64_t lo = w[idx];
    const uint64_t hi = (idx + 1 < n_words) ? w[idx + 1] : 0ULL;
    return (shift == 0) ? lo : ((lo >> shift) | (hi << (64 - shift)));
}

// -------------------- Python-visible functions --------------------

// (1) Original API: returns full vector of correlations
py::array_t<float> correlate_sliding_bin(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> signal_pm,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> pattern_pm,
    int64_t start_bit = 0,
    int64_t end_bit   = -1,
    bool debug        = false
) {
    using clock = std::chrono::high_resolution_clock;
    auto t_all_0 = clock::now();

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
    auto t_pack_0 = clock::now();
    std::vector<uint64_t> S = pack_pm_to_bits(s_ptr, N);
    std::vector<uint64_t> P = pack_pm_to_bits(p_ptr, L);
    auto t_pack_1 = clock::now();

    const size_t W = (L + 63) >> 6; // number of 64-bit words to cover L bits

    // Prepare output and masks
    auto t_alloc_0 = clock::now();
    py::array_t<float> out(positions);
    auto out_buf = out.request();
    float* dst = static_cast<float*>(out_buf.ptr);

    std::vector<uint64_t> masks(W, ~0ULL);
    if ((L & 63) != 0) {
        const unsigned rem = static_cast<unsigned>(L & 63);
        masks[W - 1] = ((1ULL << rem) - 1ULL);
    }
    auto t_alloc_1 = clock::now();

    // Main loop (release GIL while crunching)
    auto t_loop_0 = clock::now();
    {
        py::gil_scoped_release release;

        for (size_t pos = 0; pos < positions; ++pos) {
            const size_t base = start + pos;
            int mismatches = 0;

            // XOR+POPCNT per 64-bit chunk
            for (size_t k = 0; k < W; ++k) {
                const size_t bitpos = base + (k << 6); // base + 64*k
                const uint64_t sigw = load64_vec(S, bitpos);
                const uint64_t x = (sigw ^ P[k]) & masks[k];
#if defined(__GNUC__) || defined(__clang__)
                mismatches += __builtin_popcountll(x);
#else
                uint64_t y = x;
                y = y - ((y >> 1) & 0x5555555555555555ULL);
                y = (y & 0x3333333333333333ULL) + ((y >> 2) & 0x3333333333333333ULL);
                mismatches += static_cast<int>((((y + (y >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
#endif
            }

            // Map Hamming mismatches to normalized correlation in [-1, 1]
            dst[pos] = 1.0f - 2.0f * (static_cast<float>(mismatches) / static_cast<float>(L));
        }
    }
    auto t_loop_1 = clock::now();

    if (debug) {
        auto ms = [](auto dt){
            return std::chrono::duration_cast<std::chrono::microseconds>(dt).count()/1000.0;
        };
        std::fprintf(stderr,
            "[binxcorr] pack: %.3f ms (signal+pattern), alloc/masks: %.3f ms, loop: %.3f ms, total: %.3f ms\n",
            ms(t_pack_1 - t_pack_0),
            ms(t_alloc_1 - t_alloc_0),
            ms(t_loop_1 - t_loop_0),
            ms(clock::now() - t_all_0)
        );
        std::fprintf(stderr,
            "[binxcorr] bits: N=%zu, L=%zu, positions=%zu, words(W)=%zu\n",
            N, L, positions, W
        );
    }

    return out;
}

// (NEW) Argmax-only API on ±1 arrays, with per-position early-skip:
std::pair<float, int64_t> correlate_sliding_bin_argmax(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> signal_pm,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> pattern_pm,
    int64_t start_bit,
    int64_t end_bit,
    bool debug
) {
    using clock = std::chrono::high_resolution_clock;
    auto t_all_0 = clock::now();

    // Buffers & sizes
    auto s_buf = signal_pm.request();
    auto p_buf = pattern_pm.request();
    const auto* s_ptr = static_cast<int8_t*>(s_buf.ptr);
    const auto* p_ptr = static_cast<int8_t*>(p_buf.ptr);
    const size_t N = static_cast<size_t>(s_buf.size);
    const size_t L = static_cast<size_t>(p_buf.size);
    if (L == 0 || N == 0 || N < L) return { -1.0f, -1 };

    // ROI clamp
    size_t start = static_cast<size_t>(std::max<int64_t>(0, start_bit));
    size_t end   = static_cast<size_t>(end_bit < 0 ? static_cast<int64_t>(N)
                                                   : std::min<int64_t>(end_bit, static_cast<int64_t>(N)));
    if (end <= start || (end - start) < L) return { -1.0f, -1 };
    const size_t positions = (end - L) - start + 1;

    // Pack
    auto t_pack_0 = clock::now();
    std::vector<uint64_t> S = pack_pm_to_bits(s_ptr, N);
    std::vector<uint64_t> P = pack_pm_to_bits(p_ptr, L);
    auto t_pack_1 = clock::now();

    const size_t W = (L + 63) >> 6;
    std::vector<uint64_t> masks(W, ~0ULL);
    if ((L & 63) != 0) {
        const unsigned rem = static_cast<unsigned>(L & 63);
        masks[W - 1] = ((1ULL << rem) - 1ULL);
    }

    int best_mismatches = static_cast<int>(L);
    size_t best_pos = 0;

    auto t_loop_0 = clock::now();
    {
        py::gil_scoped_release release;

        for (size_t pos = 0; pos < positions; ++pos) {
            const size_t base = start + pos;
            int mismatches = 0;

            for (size_t k = 0; k < W; ++k) {
                const size_t bitpos = base + (k << 6);
                const uint64_t sigw = load64_vec(S, bitpos);
                const uint64_t x = (sigw ^ P[k]) & masks[k];
#if defined(__GNUC__) || defined(__clang__)
                mismatches += __builtin_popcountll(x);
#else
                uint64_t y = x;
                y = y - ((y >> 1) & 0x5555555555555555ULL);
                y = (y & 0x3333333333333333ULL) + ((y >> 2) & 0x3333333333333333ULL);
                mismatches += static_cast<int>((((y + (y >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
#endif
                // per-position early-skip
                if (mismatches > best_mismatches) break;
            }

            if (mismatches < best_mismatches) {
                best_mismatches = mismatches;
                best_pos = pos;
                // Stage 2: stop everything if we hit a perfect match
                if (mismatches == 0) {
                    if (debug) {
                        auto now = clock::now();
                        auto ms = [](auto dt){ return std::chrono::duration_cast<std::chrono::microseconds>(dt).count()/1000.0; };
                        std::fprintf(stderr,
                            "[binxcorr argmax] EARLY perfect match at pos=%zu (abs=%zu); pack: %.3f ms, loop: %.3f ms\n",
                            pos, start + pos, ms(t_pack_1 - t_pack_0), ms(now - t_loop_0));
                    }
                    return { 1.0f, static_cast<int64_t>(start + pos) };
                }
            }
        }
    }
    auto t_loop_1 = clock::now();

    const float best_corr = 1.0f - 2.0f * (static_cast<float>(best_mismatches) / static_cast<float>(L));
    const int64_t best_index = static_cast<int64_t>(start + best_pos);

    if (debug) {
        auto ms = [](auto dt){ return std::chrono::duration_cast<std::chrono::microseconds>(dt).count()/1000.0; };
        std::fprintf(stderr,
            "[binxcorr argmax] pack: %.3f ms, loop: %.3f ms, positions=%zu, W=%zu, best_m=%d, best_corr=%.4f\n",
            ms(t_pack_1 - t_pack_0), ms(t_loop_1 - t_loop_0), positions, W, best_mismatches, best_corr);
    }

    return { best_corr, best_index };
}

// (2) Expose packer as a Python function that returns a numpy array<uint64_t> without copying.
py::array_t<uint64_t> pack_pm_bits_py(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> pm) {

    auto buf = pm.request();
    const auto* ptr = static_cast<int8_t*>(buf.ptr);
    const size_t n = static_cast<size_t>(buf.size);

    // Build vector on the heap so its lifetime is tied to the capsule
    auto vec = new std::vector<uint64_t>(pack_pm_to_bits(ptr, n));
    auto capsule = py::capsule(vec, [](void* p){
        delete reinterpret_cast<std::vector<uint64_t>*>(p);
    });

    return py::array_t<uint64_t>(
        { vec->size() },                 // shape
        { sizeof(uint64_t) },            // strides
        vec->data(),                     // data pointer
        capsule                          // owner
    );
}

// (3) Packed API: correlate on pre-packed streams (uint64) + early-exit via min_corr
py::array_t<float> correlate_sliding_bin_packed(
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> signal_u64,
    size_t N_bits,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> pattern_u64,
    size_t L_bits,
    int64_t start_bit = 0,
    int64_t end_bit   = -1,
    double min_corr   = -1.0,   // if set (e.g. 0.9), use early-exit when too many mismatches
    bool debug        = false
) {
    using clock = std::chrono::high_resolution_clock;

    auto s_buf = signal_u64.request();
    auto p_buf = pattern_u64.request();
    const auto* S = static_cast<uint64_t*>(s_buf.ptr);
    const auto* P = static_cast<uint64_t*>(p_buf.ptr);
    const size_t S_words = static_cast<size_t>(s_buf.size);
    const size_t P_words = static_cast<size_t>(p_buf.size);
    (void)P_words;

    const size_t N = N_bits;
    const size_t L = L_bits;
    if (L == 0 || N == 0 || N < L) return py::array_t<float>(0);

    size_t start = static_cast<size_t>(std::max<int64_t>(0, start_bit));
    size_t end   = static_cast<size_t>(end_bit < 0 ? static_cast<int64_t>(N)
                                                   : std::min<int64_t>(end_bit, static_cast<int64_t>(N)));
    if (end <= start || (end - start) < L) return py::array_t<float>(0);

    const size_t positions = (end - L) - start + 1;
    const size_t W = (L + 63) >> 6;

    std::vector<uint64_t> masks(W, ~0ULL);
    if ((L & 63) != 0) {
        unsigned rem = static_cast<unsigned>(L & 63);
        masks[W - 1] = (1ULL << rem) - 1ULL;
    }

    // Compute maximum allowed mismatches from min_corr
    int max_bad = std::numeric_limits<int>::max();
    if (min_corr > -1.0 && min_corr < 1.0) {
        double m = (1.0 - min_corr) * double(L) * 0.5;
        max_bad = int(std::floor(m));
    }

    py::array_t<float> out(positions);
    auto out_buf = out.request();
    float* dst = static_cast<float*>(out_buf.ptr);

    auto t_loop_0 = clock::now();

    {
        py::gil_scoped_release release;

        for (size_t pos = 0; pos < positions; ++pos) {
            const size_t base = start + pos;
            int mismatches = 0;

            for (size_t k = 0; k < W; ++k) {
                const uint64_t sigw = load64_ptr(S, S_words, base + (k << 6));
                const uint64_t x = (sigw ^ P[k]) & masks[k];
#if defined(__GNUC__) || defined(__clang__)
                mismatches += __builtin_popcountll(x);
#else
                uint64_t y = x;
                y = y - ((y >> 1) & 0x5555555555555555ULL);
                y = (y & 0x3333333333333333ULL) + ((y >> 2) & 0x3333333333333333ULL);
                mismatches += static_cast<int>((((y + (y >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
#endif
                if (mismatches > max_bad) {
                    break;
                }
            }

            dst[pos] = 1.0f - 2.0f * (static_cast<float>(mismatches) / static_cast<float>(L));
        }
    }

    auto t_loop_1 = clock::now();

    if (debug) {
        auto ms = [](auto dt){ return std::chrono::duration_cast<std::chrono::microseconds>(dt).count()/1000.0; };
        std::fprintf(stderr, "[binxcorr packed] loop: %.3f ms (positions=%zu, W=%zu, max_bad=%d)\n",
                     ms(t_loop_1 - t_loop_0), positions, W, max_bad);
    }

    return out;
}

// --- NEW: fast codeword decoder (L=64 optimized), builds output bytes & token starts ---
py::object decode_codewords_popcnt64(
    py::array_t<int8_t,  py::array::c_style | py::array::forcecast> chips_pm,  // ±1
    py::list tokens,                                                           // list[bytes], size K
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> codes_packed, // K x (L/8), packbits (big-endian bits-in-byte)
    int L,
    double thresh,
    bool return_token_positions
) {
    using clock = std::chrono::high_resolution_clock;

    // --- Access inputs ---
    auto s_buf = chips_pm.request();
    const int8_t* s_ptr = static_cast<int8_t*>(s_buf.ptr);
    const size_t  N     = static_cast<size_t>(s_buf.size);
    if (N < 8 || L <= 0) {
        if (return_token_positions) {
            return py::make_tuple(py::bytes(""), py::array_t<int64_t>(0));
        } else {
            return py::bytes("");
        }
    }
    auto cp_buf = codes_packed.request();
    if (cp_buf.ndim != 2) {
        throw std::runtime_error("codes_packed must be 2D: [K, L/8]");
    }
    const int K = static_cast<int>(cp_buf.shape[0]);
    const int B = static_cast<int>(cp_buf.shape[1]); // bytes per codeword
    if (B*8 < L) {
        throw std::runtime_error("codes_packed has insufficient bytes for L bits");
    }
    if (static_cast<int>(tokens.size()) != K) {
        throw std::runtime_error("tokens length must match codes_packed.shape[0]");
    }

    const uint8_t* cp_ptr = static_cast<uint8_t*>(cp_buf.ptr);

    // --- Build 64-bit codewords (assumes L<=64); interpret packbits(big-endian) correctly ---
    std::vector<uint64_t> codes64(K, 0ULL);
    for (int k = 0; k < K; ++k) {
        const uint8_t* row = cp_ptr + k * B;
        uint64_t w = 0ULL;
        int bit_written = 0;
        for (int bi = 0; bi < B && bit_written < L; ++bi) {
            uint8_t byte = row[bi];
            // big-endian inside the byte: bit7 -> earliest bit
            for (int b = 7; b >= 0 && bit_written < L; --b) {
                int bit = (byte >> b) & 1;
                if (bit) {
                    w |= (1ULL << bit_written);
                }
                ++bit_written;
            }
        }
        codes64[k] = w;
    }

    // --- Pack chips_pm to uint64 words (LSB=earliest bit) ---
    std::vector<uint64_t> S = pack_pm_to_bits(s_ptr, N);
    const size_t S_words = S.size();

    // --- Sliding windows: step=8 bits ---
    const int step = 8;
    const int n_pos = (N >= static_cast<size_t>(L))
                      ? static_cast<int>((N - L) / step + 1)
                      : 0;

    // threshold -> max allowed mismatches
    const int d_max = static_cast<int>(std::floor((1.0 - thresh) * double(L) * 0.5));

    // --- Find best code per-position ---
    std::vector<int>  best_idx (std::max(0, n_pos), 0);
    std::vector<int>  best_dist(std::max(0, n_pos), L);
    std::vector<char> hit_mask (std::max(0, n_pos), 0);

    auto t_loop0 = clock::now();
    {
        py::gil_scoped_release release;
        for (int pos8 = 0; pos8 < n_pos; ++pos8) {
            const size_t bitpos = static_cast<size_t>(pos8 * step);
            // load 64-bit window
            const size_t idx   = bitpos >> 6;
            const unsigned sh  = static_cast<unsigned>(bitpos & 63);
            uint64_t lo = (idx < S_words) ? S[idx] : 0ULL;
            uint64_t hi = (idx + 1 < S_words) ? S[idx + 1] : 0ULL;
            const uint64_t win = (sh == 0) ? lo : ((lo >> sh) | (hi << (64 - sh)));

            int best_d = L;
            int best_k = 0;

            // scan codes
            for (int k = 0; k < K; ++k) {
                const uint64_t x = (win ^ codes64[k]);
#if defined(__GNUC__) || defined(__clang__)
                const int d = __builtin_popcountll(x);
#else
                uint64_t y = x;
                y = y - ((y >> 1) & 0x5555555555555555ULL);
                y = (y & 0x3333333333333333ULL) + ((y >> 2) & 0x3333333333333333ULL);
                const int d = static_cast<int>((((y + (y >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
#endif
                if (d < best_d) { best_d = d; best_k = k; if (best_d == 0) break; }
            }
            best_dist[pos8] = best_d;
            best_idx [pos8] = best_k;
            hit_mask [pos8] = (best_d <= d_max) ? 1 : 0;
        }
    }
    auto t_loop1 = clock::now();

    // --- Build output bytes and token starts (byte offsets) ---
    std::string out;
    out.reserve(N / 8 + 2 * 64);

    std::vector<int64_t> token_starts; // in output byte offsets (for whitelist)
    token_starts.reserve(128);

    size_t pos_bits = 0;
    int    pos8     = 0;

    auto emit_byte_from_bits = [&](size_t bit_start){
        // pack 8 chips_pm bits: MSB-first like numpy.packbits(default)
        uint8_t v = 0;
        for (int b = 0; b < 8; ++b) {
            size_t i = bit_start + static_cast<size_t>(b);
            int bit = 0;
            if (i < N && s_ptr[i] > 0) bit = 1;
            v |= static_cast<uint8_t>(bit) << (7 - b);
        }
        out.push_back(static_cast<char>(v));
    };

    while (pos_bits < N) {
        if (pos8 < n_pos && hit_mask[pos8]) {
            // insert token bytes
            py::bytes py_tok = tokens[best_idx[pos8]].cast<py::bytes>();
            // get its data
            std::string tok = py_tok;
            // record start offset
            token_starts.push_back(static_cast<int64_t>(out.size()));
            out.append(tok);
            // advance by L bits / 8 bytes
            pos_bits += static_cast<size_t>(L);
            pos8     += (L / 8);
        } else {
            // no hit at this byte-position: emit 1 byte from 8 bits and advance 8
            if (pos_bits + 8 <= N) {
                emit_byte_from_bits(pos_bits);
                pos_bits += 8;
                pos8     += 1;
            } else {
                break;
            }
        }
    }

    if (return_token_positions) {
        py::array_t<int64_t> starts_arr(token_starts.size());
        auto buf = starts_arr.request();
        std::memcpy(buf.ptr, token_starts.data(), token_starts.size()*sizeof(int64_t));
        return py::make_tuple(py::bytes(out), starts_arr);
    } else {
        return py::bytes(out);
    }
}


// --------------------------- module ---------------------------

PYBIND11_MODULE(binxcorr, m) {
    m.doc() = "Fast binary sliding correlation (XOR+POPCNT) for ±1 streams";

    // Original API (packs inside):
    m.def("correlate_sliding_bin", &correlate_sliding_bin,
          py::arg("signal_pm"), py::arg("pattern_pm"),
          py::arg("start_bit") = 0, py::arg("end_bit") = -1,
          py::arg("debug") = false,
          "Compute normalized sliding correlation on ±1 arrays (packs internally).\n"
          "Returns float array of length (ROI_len - L + 1).");

    // (NEW) Argmax-only variant with per-position early-skip
    m.def("correlate_sliding_bin_argmax", &correlate_sliding_bin_argmax,
          py::arg("signal_pm"), py::arg("pattern_pm"),
          py::arg("start_bit") = 0, py::arg("end_bit") = -1,
          py::arg("debug") = false,
          "Return only the best correlation and its absolute index in the signal.\n"
          "Per-position early-skip after the first 64-bit word; "
          "EARLY RETURN if a perfect match is found (corr=1.0).");

    // Expose packer (±1 -> uint64 bitstream)
    m.def("pack_pm_bits", &pack_pm_bits_py, py::arg("pm"),
          "Pack ±1 int8 array into a uint64 numpy array (1=+1, 0=-1).");

    // Packed API (no packing cost inside) + early-exit via min_corr
    m.def("correlate_sliding_bin_packed", &correlate_sliding_bin_packed,
          py::arg("signal_u64"), py::arg("N_bits"),
          py::arg("pattern_u64"), py::arg("L_bits"),
          py::arg("start_bit") = 0, py::arg("end_bit") = -1,
          py::arg("min_corr") = -1.0, py::arg("debug") = false,
          "Sliding correlation on pre-packed uint64 streams. Supports early-exit with min_corr.\n"
          "Returns float array of length (ROI_len - L + 1).");

    m.def("decode_codewords_popcnt64", &decode_codewords_popcnt64,
        py::arg("chips_pm"),
        py::arg("tokens"),
        py::arg("codes_packed"),
        py::arg("L"),
        py::arg("thresh"),
        py::arg("return_token_positions") = false,
        "Decode marker codewords with packed XOR+POPCNT (step=8). "
        "Returns bytes, or (bytes, set[int]) if return_token_positions=True.");

}
