#pragma once

#include <cstdint>
#include "../core/mm.h"
#include <vector>
#include <type_traits>

/// Special algorithm for boolean matrices, uses bit-packing to achieve O(nkm/w) complexity, where w is the machine word size
///
/// Complexity: O(nkm / w)
/// Extra memory: O(n * k) bits
template<typename T>
    requires std::is_convertible_v<T, bool>
class MatrixMultiplierBitPacking : public MatrixMultiplier<T> {
public:
    void multiply_tiles(MatrixTile<T> res, const MatrixTile<T>& m1, const MatrixTile<T>& m2) override {
        size_t k_in_uints = (m2.get_height() + 63) / 64;

        Matrix<uint64_t> m1_bits(m1.get_height(), k_in_uints);
        for(size_t i = 0; i < m1.get_height(); i++) {
            for(size_t j = 0; j < k_in_uints; j++)
                m1_bits[i][j] = 0;
            for(size_t j = 0; j < m1.get_width(); j++)
                if(m1[i][j])
                    m1_bits[i][j / 64] |= static_cast<uint64_t>(1) << (j % 64);
        }

        std::vector<uint64_t> column(k_in_uints);
        for(size_t j = 0; j < m2.get_width(); j++) {
            for(uint64_t& x : column)
                x = 0;
            for(size_t i = 0; i < m2.get_height(); i++)
                if(m2[i][j])
                    column[i / 64] |= static_cast<uint64_t>(1) << (i % 64);

            for(size_t i = 0; i < m1.get_height(); i++) {
                uint64_t cur = 0;
                for(size_t k = 0; k < k_in_uints; k++)
                    cur |= m1_bits[i][k] & column[k];
                res[i][j] = cur > 0;
            }
        }
    }

    size_t estimate_multiplications(size_t n, size_t k, size_t m) override {
        // TODO: What to do with this?
        return 0;
    }
};
