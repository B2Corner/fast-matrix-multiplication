#pragma once

#include <cstdint>
#include "../core/mm.h"
#include <vector>
#include <type_traits>
#include <bit>
#include "../core/dynamic_bitset.h"
#include <functional>

/// The Four Russians algorithm for boolean matrices
///
/// Complexity: O(nkm / (w log n))
/// Extra memory: O(n * m) bits
template<typename T>
    requires std::is_convertible_v<T, bool>
class MatrixMultiplierFourRussians : public MatrixMultiplier<T> {
public:
    void multiply_tiles(MatrixTile<T> res, const MatrixTile<T>& m1, const MatrixTile<T>& m2) override {
        std::vector<DynamicBitset> res_bits(m1.get_height(), m2.get_width());
        for(size_t i = 0; i < m1.get_height(); i++)
            res_bits[i].reset();

        // computing the precalc: k/p groups, 2^p row unions for each
        // computing the result: n rows, k/p groups, 1 row union for each
        // total: k/p (2^p + n) unions
        size_t p = 1;
        while(((m1.get_width() + (p + 1) - 1) / (p + 1)) * ((1 << (p + 1)) + m1.get_height()) <
              ((m1.get_width() + p - 1) / p) * ((1 << p) + m1.get_height()))
            p++;

        std::vector<DynamicBitset> cur_block_unions(1 << p, m2.get_width());
        cur_block_unions[0].reset();
        std::vector<bool> cur_block_union_computed(1 << p);
        for(size_t j = 0; j < m1.get_width(); j += p) {
            size_t cur_p = std::min(p, m1.get_width() - j);

            for(size_t i = 0; i < (static_cast<size_t>(1) << cur_p); i++)
                cur_block_union_computed[i] = false;
            cur_block_union_computed[0] = true;
            for(size_t i = 0; i < cur_p; i++) {
                cur_block_union_computed[1 << i] = true;
                for(size_t q = 0; q < m2.get_width(); q++)
                    cur_block_unions[1 << i][q] = m2[j + i][q];
            }

            std::function<void(size_t)> compute_row;
            compute_row = [&](size_t ind) {
                if(cur_block_union_computed[ind])
                    return;
                int64_t last = std::countr_zero(ind);
                compute_row(ind ^ (static_cast<int64_t>(1) << last));
                cur_block_unions[ind] =
                    cur_block_unions[ind ^ (static_cast<int64_t>(1) << last)] | cur_block_unions[static_cast<int64_t>(1) << last];
                cur_block_union_computed[ind] = true;
            };

            for(size_t i = 0; i < m1.get_height(); i++) {
                int64_t cur_bits = 0;
                for(size_t q = 0; q < cur_p; q++)
                    if(m1[i][j + q])
                        cur_bits |= static_cast<int64_t>(1) << q;

                compute_row(cur_bits);

                res_bits[i] |= cur_block_unions[cur_bits];
            }
        }

        for(size_t i = 0; i < res.get_height(); i++)
            for(size_t j = 0; j < res.get_width(); j++)
                res[i][j] = res_bits[i][j];
    }

    size_t estimate_multiplications(size_t n, size_t k, size_t m) override {
        // TODO: What to do with this?
        return 0;
    }
};
