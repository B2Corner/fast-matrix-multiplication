#pragma once

#include "../core/mm.h"
#include "multiplier_naive.h"
#include <limits>

/// An algorithm based on bilinear aggregation (see Kaporin et al. 2004)
///
/// Complexity: O(n^{log_12(1080)}) ~ O(2.8109)
/// Extra memory: TODO: Optimize the extra memory usage and calculate this
template<typename T>
class MatrixMultiplierBilinearAggregation : public MatrixMultiplier<T> {
public:
    MatrixMultiplierBilinearAggregation(size_t fallback_size = 3, size_t max_split_size = 12)
        : fallback_size_(fallback_size), max_split_size_(max_split_size) {
    }

    void multiply_tiles(MatrixTile<T> res, const MatrixTile<T>& m1, const MatrixTile<T>& m2) override {
        size_t n = m1.get_height();
        size_t k = m1.get_width();
        size_t m = m2.get_width();

        if(n == 1 || k == 1 || m == 1) {
            multiplier_naive_.multiply_tiles(res, m1, m2);
            return;
        }

        for(size_t i = 0; i < 2; i++)
            for(size_t j = 0; j < 2; j++)
                x2_disjoint_multiply_tiles<true, true>(
                    res.get_tile(i * (n / 2), j * (m / 2), n / 2, m / 2), m1.get_tile(i * (n / 2), 0, n / 2, k / 2),
                    m2.get_tile(0, j * (m / 2), k / 2, m / 2), res.get_tile(i * (n / 2), j * (m / 2), n / 2, m / 2),
                    m1.get_tile(i * (n / 2), k / 2, n / 2, k / 2), m2.get_tile(k / 2, j * (m / 2), k / 2, m / 2));

        if(n % 2 != 0)
            multiplier_naive_.multiply_tiles(res.get_tile(n - n % 2, 0, n % 2, m - m % 2), m1.get_tile(n - n % 2, 0, n % 2, k),
                                             m2.get_tile(0, 0, k, m - m % 2));
        if(m % 2 != 0)
            multiplier_naive_.multiply_tiles(res.get_tile(0, m - m % 2, n, m % 2), m1.get_tile(0, 0, n, k),
                                             m2.get_tile(0, m - m % 2, k, m % 2));
        if(k % 2 != 0) {
            multiplier_naive_.multiply_tiles_add(res.get_tile(0, 0, n - n % 2, m - m % 2), m1.get_tile(0, k - k % 2, n - n % 2, k % 2),
                                                 m2.get_tile(k - k % 2, 0, k % 2, m - m % 2));
        }
    }

    size_t estimate_multiplications(size_t n, size_t k, size_t m) override {
        if(n == 1 || k == 1 || m == 1)
            return n * k * m;
        size_t res = find_best_d(n / 2, k / 2, m / 2).num_multiplications * 4;
        res += (n % 2) * k * (m - m % 2);
        res += n * k * (m % 2);
        res += (n - n % 2) * (k % 2) * (m - m % 2);
        return res;
    }

private:
    struct RecursiveStepDecision {
        size_t d;
        size_t num_multiplications;
    };

    RecursiveStepDecision find_best_d(size_t n, size_t k, size_t m) {
        if(n * k * m <= fallback_size_ * fallback_size_ * fallback_size_)
            return {0, 2 * n * k * m};

        size_t best_ops = 2 * n * k * m;
        size_t best_d = 0;
        size_t max_d = std::min(static_cast<size_t>(max_split_size_), std::min(n, std::min(k, m)));
        for(size_t d = 4; d <= max_d; d += 2) {
            size_t cur_ops = find_best_d(n / d, k / d, m / d).num_multiplications * ((d * d * d + 3 * d * d) / 2);
            cur_ops += 2 * (n % d) * k * (m - m % d);
            cur_ops += 2 * n * k * (m % d);
            cur_ops += 2 * (n - n % d) * (k % d) * (m - m % d);
            if(cur_ops < best_ops) {
                best_ops = cur_ops;
                best_d = d;
            }
        }
        return {best_d, best_ops};
    }

    /// Assumes the two multiplications are of the same size
    template<bool zero_result, bool add_to_result>
    void x2_disjoint_multiply_tiles(MatrixTile<T> res1, const MatrixTile<T>& m11, const MatrixTile<T>& m12, MatrixTile<T> res2,
                                    const MatrixTile<T>& m21, const MatrixTile<T>& m22) {
        assert(m11.get_width() == m12.get_height() && m11.get_height() == res1.get_height() && m12.get_width() == res1.get_width());
        assert(m11.get_height() == m21.get_height() && m11.get_width() == m21.get_width() && m12.get_width() == m22.get_width());
        size_t n = m11.get_height();
        size_t k = m11.get_width();
        size_t m = m12.get_width();

        if constexpr(zero_result) {
            for(size_t i = 0; i < n; i++)
                for(size_t j = 0; j < m; j++) {
                    res1[i][j] = 0;
                    res2[i][j] = 0;
                }
        }

        // Partition all input matrices in d x d blocks for some d, the blocks are not necessarily square
        auto [d, _] = find_best_d(n, k, m);

        if(d == 0) {
            if constexpr(add_to_result) {
                multiplier_naive_.multiply_tiles_add(res1, m11, m12);
                multiplier_naive_.multiply_tiles_add(res2, m21, m22);
            } else {
                multiplier_naive_.multiply_tiles_subtract(res1, m11, m12);
                multiplier_naive_.multiply_tiles_subtract(res2, m21, m22);
            }
            return;
        }

        // Storage for temporary stuff
        Matrix<T> tmp_res(n / d, m / d);
        Matrix<T> tmp_res2(n / d, m / d);

        // Compute c_j and r_k
        Matrix<T> r_k(n - n % d, m / d);
        for(size_t i = 0; i < n - n % d; i++)
            for(size_t j = 0; j < m / d; j++)
                r_k[i][j] = 0;
        for(size_t j = 0; j < d; j++) {
            // Clear tmp_res(2)
            Matrix<T> c_j1(n / d, m / d);
            Matrix<T> c_j2(n / d, m / d);
            for(size_t i = 0; i < n / d; i++)
                for(size_t j = 0; j < m / d; j++) {
                    c_j1[i][j] = 0;
                    c_j2[i][j] = 0;
                }

            for(size_t p = 0; p < d / 2; p++) {
                MatrixTile<T> u1 = m21.get_tile(p * (n / d), j * (k / d), n / d, k / d);
                MatrixTile<T> y1 = m12.get_tile(p * (k / d), j * (m / d), k / d, m / d);
                MatrixTile<T> u2 = m21.get_tile((p + d / 2) * (n / d), j * (k / d), n / d, k / d);
                MatrixTile<T> y2 = m12.get_tile((p + d / 2) * (k / d), j * (m / d), k / d, m / d);

                x2_disjoint_multiply_tiles<true, true>(tmp_res, u1, y1, tmp_res2, u2, y2);

                c_j1 += tmp_res;
                c_j2 += tmp_res2;
                add(r_k.get_tile(p * (n / d), 0, n / d, m / d), tmp_res);
                add(r_k.get_tile((p + d / 2) * (n / d), 0, n / d, m / d), tmp_res2);
            }

            for(size_t i = 0; i < d; i++) {
                if constexpr(add_to_result) {
                    subtract(res1.get_tile((n / d) * i, (m / d) * j, n / d, m / d), c_j1);
                    subtract(res1.get_tile((n / d) * i, (m / d) * j, n / d, m / d), c_j2);
                } else {
                    add(res1.get_tile((n / d) * i, (m / d) * j, n / d, m / d), c_j1);
                    add(res1.get_tile((n / d) * i, (m / d) * j, n / d, m / d), c_j2);
                }
            }
        }

        // Factor in r_k
        for(size_t i = 0; i < d; i++) {
            if constexpr(add_to_result)
                subtract(res2.get_tile(0, (m / d) * i, n - n % d, m / d), r_k);
            else
                add(res2.get_tile(0, (m / d) * i, n - n % d, m / d), r_k);
        }

        // Compute correctional terms for the first pair
        Matrix<T> sum_u = m21.get_tile(0, 0, n / d, k - k % d);
        for(size_t p = 1; p < d; p++)
            sum_u += m21.get_tile(p * (n / d), 0, n / d, k - k % d);
        for(size_t i = 0; i < d; i++) {
            Matrix<T> sum_x(n / d, k / d);
            for(size_t j = 0; j < n / d; j++)
                for(size_t p = 0; p < k / d; p++)
                    sum_x[j][p] = 0;

            for(size_t p = 0; p < d; p++)
                sum_x += m11.get_tile(i * (n / d), p * (k / d), n / d, k / d);

            Matrix<T> x_plus_u1(n / d, k / d);
            Matrix<T> x_plus_u2(n / d, k / d);
            for(size_t j = 0; j < d; j += 2) {
                MatrixTile<T> v1 = m22.get_tile(j * (k / d), i * (m / d), k / d, m / d);
                MatrixTile<T> v2 = m22.get_tile((j + 1) * (k / d), i * (m / d), k / d, m / d);

                add(x_plus_u1.get_tile(), sum_x, sum_u.get_tile(0, j * (k / d), n / d, k / d));
                add(x_plus_u2.get_tile(), sum_x, sum_u.get_tile(0, (j + 1) * (k / d), n / d, k / d));
                x2_disjoint_multiply_tiles<false, !add_to_result>(res1.get_tile((n / d) * i, (m / d) * j, n / d, m / d), x_plus_u1, v1,
                                                                  res1.get_tile((n / d) * i, (m / d) * (j + 1), n / d, m / d), x_plus_u2,
                                                                  v2);
            }
        }

        // Compute correctional terms for the second pair
        Matrix<T> sum_y = m12.get_tile(0, 0, k - k % d, m / d);
        for(size_t p = 1; p < d; p++)
            sum_y += m12.get_tile(0, p * (m / d), k - k % d, m / d);
        for(size_t i = 0; i < d; i++) {
            Matrix<T> sum_v(k / d, m / d);
            for(size_t j = 0; j < k / d; j++)
                for(size_t p = 0; p < m / d; p++)
                    sum_v[j][p] = 0;

            for(size_t j = 0; j < d; j++)
                sum_v += m22.get_tile(j * (k / d), i * (m / d), k / d, m / d);

            Matrix<T> y_plus_v1(k / d, m / d);
            Matrix<T> y_plus_v2(k / d, m / d);
            for(size_t p = 0; p < d; p += 2) {
                MatrixTile<T> x1 = m11.get_tile(i * (n / d), p * (k / d), n / d, k / d);
                MatrixTile<T> x2 = m11.get_tile(i * (n / d), (p + 1) * (k / d), n / d, k / d);

                add(y_plus_v1.get_tile(), sum_y.get_tile(p * (k / d), 0, k / d, m / d), sum_v);
                add(y_plus_v2.get_tile(), sum_y.get_tile((p + 1) * (k / d), 0, k / d, m / d), sum_v);
                x2_disjoint_multiply_tiles<false, !add_to_result>(res2.get_tile((n / d) * p, (m / d) * i, n / d, m / d), x1, y_plus_v1,
                                                                  res2.get_tile((n / d) * (p + 1), (m / d) * i, n / d, m / d), x2,
                                                                  y_plus_v2);
            }
        }

        // Compute m_{ijk}
        Matrix<T> x_plus_u_1(n / d, k / d);
        Matrix<T> x_plus_u_2(n / d, k / d);
        Matrix<T> y_plus_v_1(k / d, m / d);
        Matrix<T> y_plus_v_2(k / d, m / d);
        size_t prev_pair[3] = {std::numeric_limits<size_t>::max(), 0, 0};
        for(size_t i = 0; i < d; i++)
            for(size_t j = 0; j < d; j++) {
                MatrixTile<T> v = m22.get_tile(j * (k / d), i * (m / d), k / d, m / d);
                for(size_t p = 0; p < d; p++) {
                    MatrixTile<T> x = m11.get_tile(i * (n / d), p * (k / d), n / d, k / d);
                    MatrixTile<T> y = m12.get_tile(p * (k / d), j * (m / d), k / d, m / d);
                    MatrixTile<T> u = m21.get_tile(p * (n / d), j * (k / d), n / d, k / d);

                    if(prev_pair[0] == std::numeric_limits<size_t>::max()) {
                        add_tiles(x_plus_u_1.get_tile(), x, u);
                        add_tiles(y_plus_v_1.get_tile(), y, v);

                        prev_pair[0] = i;
                        prev_pair[1] = j;
                        prev_pair[2] = p;
                    } else {
                        add_tiles(x_plus_u_2.get_tile(), x, u);
                        add_tiles(y_plus_v_2.get_tile(), y, v);

                        x2_disjoint_multiply_tiles<true, true>(tmp_res, x_plus_u_1, y_plus_v_1, tmp_res2, x_plus_u_2, y_plus_v_2);

                        if constexpr(add_to_result) {
                            add(res1.get_tile((n / d) * prev_pair[0], (m / d) * prev_pair[1], n / d, m / d), tmp_res);
                            add(res1.get_tile((n / d) * i, (m / d) * j, n / d, m / d), tmp_res2);

                            add(res2.get_tile((n / d) * prev_pair[2], (m / d) * prev_pair[0], n / d, m / d), tmp_res);
                            add(res2.get_tile((n / d) * p, (m / d) * i, n / d, m / d), tmp_res2);
                        } else {
                            subtract(res1.get_tile((n / d) * prev_pair[0], (m / d) * prev_pair[1], n / d, m / d), tmp_res);
                            subtract(res1.get_tile((n / d) * i, (m / d) * j, n / d, m / d), tmp_res2);

                            subtract(res2.get_tile((n / d) * prev_pair[2], (m / d) * prev_pair[0], n / d, m / d), tmp_res);
                            subtract(res2.get_tile((n / d) * p, (m / d) * i, n / d, m / d), tmp_res2);
                        }

                        prev_pair[0] = std::numeric_limits<size_t>::max();
                    }
                }
            }

        // Account for the stuff that did not make it to the recursive calls
        if(n % d != 0) {
            if constexpr(add_to_result) {
                multiplier_naive_.multiply_tiles_add(res1.get_tile(n - n % d, 0, n % d, m - m % d), m11.get_tile(n - n % d, 0, n % d, k),
                                                     m12.get_tile(0, 0, k, m - m % d));
                multiplier_naive_.multiply_tiles_add(res2.get_tile(n - n % d, 0, n % d, m - m % d), m21.get_tile(n - n % d, 0, n % d, k),
                                                     m22.get_tile(0, 0, k, m - m % d));
            } else {
                multiplier_naive_.multiply_tiles_subtract(res1.get_tile(n - n % d, 0, n % d, m - m % d),
                                                          m11.get_tile(n - n % d, 0, n % d, k), m12.get_tile(0, 0, k, m - m % d));
                multiplier_naive_.multiply_tiles_subtract(res2.get_tile(n - n % d, 0, n % d, m - m % d),
                                                          m21.get_tile(n - n % d, 0, n % d, k), m22.get_tile(0, 0, k, m - m % d));
            }
        }
        if(m % d != 0) {
            if constexpr(add_to_result) {
                multiplier_naive_.multiply_tiles_add(res1.get_tile(0, m - m % d, n, m % d), m11.get_tile(0, 0, n, k),
                                                     m12.get_tile(0, m - m % d, k, m % d));
                multiplier_naive_.multiply_tiles_add(res2.get_tile(0, m - m % d, n, m % d), m21.get_tile(0, 0, n, k),
                                                     m22.get_tile(0, m - m % d, k, m % d));
            } else {
                multiplier_naive_.multiply_tiles_subtract(res1.get_tile(0, m - m % d, n, m % d), m11.get_tile(0, 0, n, k),
                                                          m12.get_tile(0, m - m % d, k, m % d));
                multiplier_naive_.multiply_tiles_subtract(res2.get_tile(0, m - m % d, n, m % d), m21.get_tile(0, 0, n, k),
                                                          m22.get_tile(0, m - m % d, k, m % d));
            }
        }
        if(k % d != 0) {
            if constexpr(add_to_result) {
                multiplier_naive_.multiply_tiles_add(res1.get_tile(0, 0, n - n % d, m - m % d),
                                                     m11.get_tile(0, k - k % d, n - n % d, k % d),
                                                     m12.get_tile(k - k % d, 0, k % d, m - m % d));
                multiplier_naive_.multiply_tiles_add(res2.get_tile(0, 0, n - n % d, m - m % d),
                                                     m21.get_tile(0, k - k % d, n - n % d, k % d),
                                                     m22.get_tile(k - k % d, 0, k % d, m - m % d));
            } else {
                multiplier_naive_.multiply_tiles_subtract(res1.get_tile(0, 0, n - n % d, m - m % d),
                                                          m11.get_tile(0, k - k % d, n - n % d, k % d),
                                                          m12.get_tile(k - k % d, 0, k % d, m - m % d));
                multiplier_naive_.multiply_tiles_subtract(res2.get_tile(0, 0, n - n % d, m - m % d),
                                                          m21.get_tile(0, k - k % d, n - n % d, k % d),
                                                          m22.get_tile(k - k % d, 0, k % d, m - m % d));
            }
        }
    }

    size_t fallback_size_;
    size_t max_split_size_;
    MatrixMultiplierNaive<T> multiplier_naive_;
};
