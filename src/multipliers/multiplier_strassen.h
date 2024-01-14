#pragma once

#include "../core/mm.h"
#include <algorithm>
#include <vector>
#include "multiplier_naive.h"

/// Classic Strassen algorithm
/// This implementation follows the formulas from Winograd from, but due to extra memory constraints makes
/// slightly more additions than what Winograd needs in theory.
///
/// Complexity: O(n^{log_2(7)}) ~ O(n^2.8074)
/// Extra memory: (n * k + k * m + n * m) / 3
template<typename T>
class MatrixMultiplierStrassen : public MatrixMultiplier<T> {
public:
    MatrixMultiplierStrassen() : fallback_enabled_(false), fallback_size_(0) {
    }

    MatrixMultiplierStrassen(size_t fallback_size) : fallback_enabled_(true), fallback_size_(fallback_size) {
    }

    size_t estimate_multiplications(size_t n, size_t k, size_t m) override {
        if(n == 0 || k == 0 || m == 0)
            return 0;
        if(fallback_enabled_ && std::max(n, std::max(k, m)) <= fallback_size_) {
            return multiplier_naive_.estimate_multiplications(n, k, m);
        }

        size_t res = 0;

        size_t height_part_left = n / 2;
        size_t width_part_left = k / 2;
        size_t width_part_right = m / 2;

        res += 7 * estimate_multiplications(height_part_left, width_part_left, width_part_right);

        // Account for odd dimensions
        if(k % 2 == 1)
            res += 4 * height_part_left * width_part_right;
        if(m % 2 == 1)
            res += 2 * height_part_left * k;
        if(n % 2 == 1)
            res += m * k;
        return res;
    }

    void multiply_tiles(MatrixTile<T> res, const MatrixTile<T>& m1, const MatrixTile<T>& m2) override {
        std::vector<Matrix<T>> scratchpads_left;
        std::vector<Matrix<T>> scratchpads_right;
        std::vector<Matrix<T>> scratchpads_res;
        generate_scratchpads(m1.get_height(), m1.get_width(), m2.get_width(), scratchpads_left, scratchpads_right, scratchpads_res);
        multiply_tiles_overwrite(res, m1, m2, scratchpads_left.data(), scratchpads_right.data(), scratchpads_res.data());
    }

private:
    bool fallback_enabled_;
    size_t fallback_size_;
    MatrixMultiplierNaive<T> multiplier_naive_;

    void generate_scratchpads(size_t n, size_t k, size_t m, std::vector<Matrix<T>>& scratchpads_left,
                              std::vector<Matrix<T>>& scratchpads_right, std::vector<Matrix<T>>& scratchpads_res) {
        while(n > 1 && k > 1 && m > 1) {
            scratchpads_left.emplace_back(n / 2, k / 2);
            scratchpads_right.emplace_back(k / 2, m / 2);
            scratchpads_res.emplace_back(n / 2, m / 2);
            n /= 2;
            k /= 2;
            m /= 2;
        }
    }

    // res2 = v
    // res3 = u
    // res4 = wr
    // res1 = t
    // res4 += res1
    // res3 += res4
    //   t     v
    //  w + u  w
    // s = res4
    // res4 = res3 + res2
    // res2 += s
    //   t     w + v
    //  w + u  w + u + v

    void multiply_tiles_overwrite(MatrixTile<T> res, const MatrixTile<T>& m1, const MatrixTile<T>& m2, Matrix<T>* scratchpads_left,
                                  Matrix<T>* scratchpads_right, Matrix<T>* scratchpads_res) {
        if(m1.get_height() == 0 || m1.get_width() == 0 || m2.get_width() == 0)
            return;
        if(fallback_enabled_ && std::max(m1.get_height(), std::max(m1.get_width(), m2.get_width())) <= fallback_size_) {
            multiplier_naive_.multiply_tiles(res, m1, m2);
            return;
        }

        size_t height_part_left = m1.get_height() / 2;
        size_t width_part_left = m1.get_width() / 2;
        size_t height_part_right = m2.get_height() / 2;
        size_t width_part_right = m2.get_width() / 2;

        if(m1.get_height() > 1 && m1.get_width() > 1 && m2.get_width() > 1) {
            auto a = m1.get_tile(0, 0, height_part_left, width_part_left);
            auto b = m1.get_tile(0, width_part_left, height_part_left, width_part_left);
            auto c = m1.get_tile(height_part_left, 0, height_part_left, width_part_left);
            auto d = m1.get_tile(height_part_left, width_part_left, height_part_left, width_part_left);

            auto A = m2.get_tile(0, 0, height_part_right, width_part_right);
            auto B = m2.get_tile(height_part_right, 0, height_part_right, width_part_right);
            auto C = m2.get_tile(0, width_part_right, height_part_right, width_part_right);
            auto D = m2.get_tile(height_part_right, width_part_right, height_part_right, width_part_right);

            auto res1 = res.get_tile(0, 0, height_part_left, width_part_right);
            auto res2 = res.get_tile(0, width_part_right, height_part_left, width_part_right);
            auto res3 = res.get_tile(height_part_left, 0, height_part_left, width_part_right);
            auto res4 = res.get_tile(height_part_left, width_part_right, height_part_left, width_part_right);

            auto scratchpad_left = static_cast<MatrixTile<T>>(*scratchpads_left);
            auto scratchpad_right = static_cast<MatrixTile<T>>(*scratchpads_right);
            auto scratchpad_res = static_cast<MatrixTile<T>>(*scratchpads_res);

            // Compute v
            add_tiles(scratchpad_left, c, d);
            subtract_tiles(scratchpad_right, C, A);
            multiply_tiles_overwrite(res2, scratchpad_left, scratchpad_right, scratchpads_left + 1, scratchpads_right + 1,
                                     scratchpads_res + 1);

            // Compute u
            subtract_tiles(scratchpad_left, c, a);
            subtract_tiles(scratchpad_right, C, D);
            multiply_tiles_overwrite(res3, scratchpad_left, scratchpad_right, scratchpads_left + 1, scratchpads_right + 1,
                                     scratchpads_res + 1);

            // Compute right part of w
            add_tiles(scratchpad_left, d);
            subtract_tiles(scratchpad_right, A, scratchpad_right);
            multiply_tiles_overwrite(res4, scratchpad_left, scratchpad_right, scratchpads_left + 1, scratchpads_right + 1,
                                     scratchpads_res + 1);

            // Compute t
            multiply_tiles_overwrite(res1, a, A, scratchpads_left + 1, scratchpads_right + 1, scratchpads_res + 1);

            add_tiles(res4, res1);
            add_tiles(res3, res4);
            copy_tile(scratchpad_res, res4);
            add_tiles(res4, res3, res2);
            add_tiles(res2, scratchpad_res);

            // Finish res1
            multiply_tiles_add(res1, b, B, scratchpads_left + 1, scratchpads_right + 1, scratchpads_res + 1);

            // Finish res2
            subtract_tiles(scratchpad_left, b, scratchpad_left);
            multiply_tiles_add(res2, scratchpad_left, D, scratchpads_left + 1, scratchpads_right + 1, scratchpads_res + 1);

            // Finish res3
            add_tiles(scratchpad_right, B, C);
            subtract_tiles(scratchpad_right, A);
            subtract_tiles(scratchpad_right, D);
            multiply_tiles_add(res3, d, scratchpad_right, scratchpads_left + 1, scratchpads_right + 1, scratchpads_res + 1);
        }

        // Account for odd dimensions
        if(m1.get_width() % 2 == 1) {
            if(m1.get_width() > 1) {
                for(size_t i = 0; i < 2 * height_part_left; i++)
                    for(size_t j = 0; j < 2 * width_part_right; j++)
                        res[i][j] += m1[i][m1.get_width() - 1] * m2[m2.get_height() - 1][j];
            } else {
                for(size_t i = 0; i < 2 * height_part_left; i++)
                    for(size_t j = 0; j < 2 * width_part_right; j++)
                        res[i][j] = m1[i][m1.get_width() - 1] * m2[m2.get_height() - 1][j];
            }
        }

        if(m2.get_width() % 2 == 1) {
            for(size_t i = 0; i < 2 * height_part_left; i++) {
                res[i][m2.get_width() - 1] = static_cast<T>(0);
                for(size_t j = 0; j < m1.get_width(); j++)
                    res[i][m2.get_width() - 1] += m1[i][j] * m2[j][m2.get_width() - 1];
            }
        }

        if(m1.get_height() % 2 == 1) {
            for(size_t j = 0; j < m2.get_width(); j++) {
                res[m1.get_height() - 1][j] = static_cast<T>(0);
                for(size_t i = 0; i < m1.get_width(); i++)
                    res[m1.get_height() - 1][j] += m1[m1.get_height() - 1][i] * m2[i][j];
            }
        }
    }

    void multiply_tiles_add(MatrixTile<T> res, const MatrixTile<T>& m1, const MatrixTile<T>& m2, Matrix<T>* scratchpads_left,
                            Matrix<T>* scratchpads_right, Matrix<T>* scratchpads_res) {
        if(m1.get_height() == 0 || m1.get_width() == 0 || m2.get_width() == 0)
            return;
        if(fallback_enabled_ && std::max(m1.get_height(), std::max(m1.get_width(), m2.get_width())) <= fallback_size_) {
            multiplier_naive_.multiply_tiles_add(res, m1, m2);
            return;
        }

        size_t height_part_left = m1.get_height() / 2;
        size_t width_part_left = m1.get_width() / 2;
        size_t height_part_right = m2.get_height() / 2;
        size_t width_part_right = m2.get_width() / 2;

        if(m1.get_height() > 1 && m1.get_width() > 1 && m2.get_width() > 1) {
            auto a = m1.get_tile(0, 0, height_part_left, width_part_left);
            auto b = m1.get_tile(0, width_part_left, height_part_left, width_part_left);
            auto c = m1.get_tile(height_part_left, 0, height_part_left, width_part_left);
            auto d = m1.get_tile(height_part_left, width_part_left, height_part_left, width_part_left);

            auto A = m2.get_tile(0, 0, height_part_right, width_part_right);
            auto B = m2.get_tile(height_part_right, 0, height_part_right, width_part_right);
            auto C = m2.get_tile(0, width_part_right, height_part_right, width_part_right);
            auto D = m2.get_tile(height_part_right, width_part_right, height_part_right, width_part_right);

            auto res1 = res.get_tile(0, 0, height_part_left, width_part_right);
            auto res2 = res.get_tile(0, width_part_right, height_part_left, width_part_right);
            auto res3 = res.get_tile(height_part_left, 0, height_part_left, width_part_right);
            auto res4 = res.get_tile(height_part_left, width_part_right, height_part_left, width_part_right);

            auto scratchpad_left = static_cast<MatrixTile<T>>(*scratchpads_left);
            auto scratchpad_right = static_cast<MatrixTile<T>>(*scratchpads_right);
            auto scratchpad_res = static_cast<MatrixTile<T>>(*scratchpads_res);

            // Compute v
            add_tiles(scratchpad_left, c, d);
            subtract_tiles(scratchpad_right, C, A);
            multiply_tiles_overwrite(scratchpad_res, scratchpad_left, scratchpad_right, scratchpads_left + 1, scratchpads_right + 1,
                                     scratchpads_res + 1);
            add_tiles(res2, scratchpad_res);
            add_tiles(res4, scratchpad_res);

            // Compute u
            subtract_tiles(scratchpad_left, c, a);
            subtract_tiles(scratchpad_right, C, D);
            multiply_tiles_overwrite(scratchpad_res, scratchpad_left, scratchpad_right, scratchpads_left + 1, scratchpads_right + 1,
                                     scratchpads_res + 1);
            add_tiles(res3, scratchpad_res);
            add_tiles(res4, scratchpad_res);

            // Compute t
            multiply_tiles_overwrite(scratchpad_res, a, A, scratchpads_left + 1, scratchpads_right + 1, scratchpads_res + 1);
            add_tiles(res1, scratchpad_res);

            // Compute right part of w
            add_tiles(scratchpad_left, d);
            subtract_tiles(scratchpad_right, A, scratchpad_right);
            multiply_tiles_add(scratchpad_res, scratchpad_left, scratchpad_right, scratchpads_left + 1, scratchpads_right + 1,
                               scratchpads_res + 1);
            add_tiles(res2, scratchpad_res);
            add_tiles(res3, scratchpad_res);
            add_tiles(res4, scratchpad_res);

            // Finish res1
            multiply_tiles_add(res1, b, B, scratchpads_left + 1, scratchpads_right + 1, scratchpads_res + 1);

            // Finish res2
            subtract_tiles(scratchpad_left, b, scratchpad_left);
            multiply_tiles_add(res2, scratchpad_left, D, scratchpads_left + 1, scratchpads_right + 1, scratchpads_res + 1);

            // Finish res3
            add_tiles(scratchpad_right, B, C);
            subtract_tiles(scratchpad_right, A);
            subtract_tiles(scratchpad_right, D);
            multiply_tiles_add(res3, d, scratchpad_right, scratchpads_left + 1, scratchpads_right + 1, scratchpads_res + 1);
        }

        // Account for odd dimensions
        if(m1.get_width() % 2 == 1) {
            for(size_t i = 0; i < 2 * height_part_left; i++)
                for(size_t j = 0; j < 2 * width_part_right; j++)
                    res[i][j] += m1[i][m1.get_width() - 1] * m2[m2.get_height() - 1][j];
        }

        if(m2.get_width() % 2 == 1) {
            for(size_t i = 0; i < 2 * height_part_left; i++) {
                for(size_t j = 0; j < m1.get_width(); j++)
                    res[i][m2.get_width() - 1] += m1[i][j] * m2[j][m2.get_width() - 1];
            }
        }

        if(m1.get_height() % 2 == 1) {
            for(size_t j = 0; j < m2.get_width(); j++) {
                for(size_t i = 0; i < m1.get_width(); i++)
                    res[m1.get_height() - 1][j] += m1[m1.get_height() - 1][i] * m2[i][j];
            }
        }
    }
};
