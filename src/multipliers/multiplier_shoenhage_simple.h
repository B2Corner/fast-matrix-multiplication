#pragma once

#include <cstdint>
#include "../core/mm.h"
#include "../core/poly.h"
#include "../core/poly_matix_utils.h"

/// An APA-type algorithm, based on the fact that the border rank of 3x3x3 multiplicaiton is <= 21
/// See "PARTIAL AND TOTAL MATRIX MULTIPLICATION" By A. Shoenhage(1981), example 2.3
///
/// Complexity: O(n^{log_3^21} (log_n)^{log_2^3}) ~ O(n^2.772 (log_n)^1.585) (usually performs better than that)
/// Extra memory: TODO
template<typename T>
class MatrixMultiplierShoenhageSimple : public MatrixMultiplier<T> {
public:
    void multiply_tiles(MatrixTile<T> res, const MatrixTile<T>& m1, const MatrixTile<T>& m2) override {
        std::vector<Matrix<Poly<T>>> scratchpads_left;
        std::vector<Matrix<Poly<T>>> scratchpads_right;
        std::vector<Matrix<Poly<T>>> scratchpads_res;
        generate_scratchpads(m1.get_height(), m1.get_width(), m2.get_width(), scratchpads_left, scratchpads_right, scratchpads_res);

        Matrix<Poly<T>> m1_poly = m1;
        Matrix<Poly<T>> m2_poly = m2;
        Matrix<Poly<T>> res_poly(m1.get_height(), m2.get_width());
        multiply_tiles_overwrite(res_poly, m1_poly, m2_poly, scratchpads_left.data(), scratchpads_right.data(), scratchpads_res.data());

        for(size_t i = 0; i < res.get_height(); i++)
            for(size_t j = 0; j < res.get_width(); j++)
                res[i][j] = res_poly[i][j].evaluate_at_0();
    }

    size_t estimate_multiplications(size_t n, size_t k, size_t m) override {
        return estimate_impl(n, k, m, 0, 0);
    }

private:
    void generate_scratchpads(size_t n, size_t k, size_t m, std::vector<Matrix<Poly<T>>>& scratchpads_left,
                              std::vector<Matrix<Poly<T>>>& scratchpads_right, std::vector<Matrix<Poly<T>>>& scratchpads_res) {
        if(!(n > 2 && k > 2 && m > 2))
            return;

        scratchpads_left.push_back(Matrix<Poly<T>>(n / 3, k / 3));
        scratchpads_right.push_back(Matrix<Poly<T>>(k / 3, m / 3));
        scratchpads_res.push_back(Matrix<Poly<T>>(n / 3, m / 3));
        n /= 3;
        k /= 3;
        m /= 3;

        generate_scratchpads(n, k, m, scratchpads_left, scratchpads_right, scratchpads_res);
    }

    void multiply_tiles_add(MatrixTile<Poly<T>> res, int64_t shift, const MatrixTile<Poly<T>>& m1, const MatrixTile<Poly<T>>& m2,
                            Matrix<Poly<T>>* scratchpads_left, Matrix<Poly<T>>* scratchpads_right, Matrix<Poly<T>>* scratchpads_res) {
        if(m1.get_height() == 0 || m1.get_width() == 0 || m2.get_width() == 0)
            return;

        size_t height_part_left = m1.get_height() / 3;
        size_t width_part_left = m1.get_width() / 3;
        size_t height_part_right = m2.get_height() / 3;
        size_t width_part_right = m2.get_width() / 3;

        if(m1.get_height() > 2 && m1.get_width() > 2 && m2.get_width() > 2) {
            const MatrixTile<Poly<T>> a[3][3] = {
                {m1.get_tile(0 * height_part_left, 0 * width_part_left, height_part_left, width_part_left),
                 m1.get_tile(0 * height_part_left, 1 * width_part_left, height_part_left, width_part_left),
                 m1.get_tile(0 * height_part_left, 2 * width_part_left, height_part_left, width_part_left)},
                {m1.get_tile(1 * height_part_left, 0 * width_part_left, height_part_left, width_part_left),
                 m1.get_tile(1 * height_part_left, 1 * width_part_left, height_part_left, width_part_left),
                 m1.get_tile(1 * height_part_left, 2 * width_part_left, height_part_left, width_part_left)},
                {m1.get_tile(2 * height_part_left, 0 * width_part_left, height_part_left, width_part_left),
                 m1.get_tile(2 * height_part_left, 1 * width_part_left, height_part_left, width_part_left),
                 m1.get_tile(2 * height_part_left, 2 * width_part_left, height_part_left, width_part_left)}};

            const MatrixTile<Poly<T>> b[3][3] = {
                {m2.get_tile(0 * height_part_right, 0 * width_part_right, height_part_right, width_part_right),
                 m2.get_tile(0 * height_part_right, 1 * width_part_right, height_part_right, width_part_right),
                 m2.get_tile(0 * height_part_right, 2 * width_part_right, height_part_right, width_part_right)},
                {m2.get_tile(1 * height_part_right, 0 * width_part_right, height_part_right, width_part_right),
                 m2.get_tile(1 * height_part_right, 1 * width_part_right, height_part_right, width_part_right),
                 m2.get_tile(1 * height_part_right, 2 * width_part_right, height_part_right, width_part_right)},
                {m2.get_tile(2 * height_part_right, 0 * width_part_right, height_part_right, width_part_right),
                 m2.get_tile(2 * height_part_right, 1 * width_part_right, height_part_right, width_part_right),
                 m2.get_tile(2 * height_part_right, 2 * width_part_right, height_part_right, width_part_right)}};

            MatrixTile<Poly<T>> res_tiles[3][3];
            for(int32_t i = 0; i < 3; i++)
                for(int32_t j = 0; j < 3; j++)
                    res_tiles[i][j] = res.get_tile(i * height_part_left, j * width_part_right, height_part_left, width_part_right);

            auto scratchpad_left = static_cast<MatrixTile<Poly<T>>>(*scratchpads_left);
            auto scratchpad_right = static_cast<MatrixTile<Poly<T>>>(*scratchpads_right);
            auto scratchpad_res = static_cast<MatrixTile<Poly<T>>>(*scratchpads_res);

            // Compute p_{ij}
            for(int32_t i = 0; i < 3; i++) {
                for(int32_t j = 0; j < 3; j++) {
                    if(i == j)
                        continue;

                    add_poly_tiles<2, 0>(scratchpad_left, a[i][0], a[j][2]);
                    add_poly_tiles<0, 1>(scratchpad_right, b[0][j], b[2][i]);
                    multiply_tiles_overwrite(scratchpad_res, scratchpad_left, scratchpad_right, scratchpads_left + 1, scratchpads_right + 1,
                                             scratchpads_res + 1);
                    add_poly_tiles(res_tiles[i][j], -2 + shift, scratchpad_res);
                    add_poly_tiles(res_tiles[j][i], -1 + shift, scratchpad_res);
                }
            }

            // Compute q_{ij}
            for(int32_t i = 0; i < 3; i++) {
                for(int32_t j = 0; j < 3; j++) {
                    if(i == j)
                        continue;

                    add_poly_tiles<2, 0>(scratchpad_left, a[i][1], a[j][2]);
                    subtract_poly_tiles<0, 1>(scratchpad_right, b[1][j], b[2][i]);
                    multiply_tiles_add(res_tiles[i][j], -2 + shift, scratchpad_left, scratchpad_right, scratchpads_left + 1,
                                       scratchpads_right + 1, scratchpads_res + 1);
                }
            }

            // Compute r_j
            for(int32_t j = 0; j < 3; j++) {
                add_tiles(scratchpad_right, b[0][j], b[1][j]);
                multiply_tiles_overwrite(scratchpad_res, a[j][2], scratchpad_right, scratchpads_left + 1, scratchpads_right + 1,
                                         scratchpads_res + 1);
                for(int32_t i = 0; i < 3; i++)
                    subtract_poly_tiles(res_tiles[i][j], -2 + shift, scratchpad_res);
            }

            // Compute p_{ii}
            for(int32_t i = 0; i < 3; i++) {
                add_poly_tiles<2, 0>(scratchpad_left, a[i][0], a[i][2]);
                multiply_tiles_overwrite(scratchpad_res, scratchpad_left, b[0][i], scratchpads_left + 1, scratchpads_right + 1,
                                         scratchpads_res + 1);

                for(int32_t j = 0; j < 3; j++)
                    if(i != j)
                        subtract_poly_tiles(res_tiles[i][j], -1 + shift, scratchpad_res);
                    else
                        add_poly_tiles(res_tiles[i][j], -2 + shift, scratchpad_res);
            }

            // Compute q_{ii}
            for(int32_t i = 0; i < 3; i++) {
                add_poly_tiles<2, 0>(scratchpad_left, a[i][1], a[i][2]);
                add_poly_tiles<0, 2>(scratchpad_right, b[1][i], b[2][i]);
                multiply_tiles_add(res_tiles[i][i], -2 + shift, scratchpad_left, scratchpad_right, scratchpads_left + 1,
                                   scratchpads_right + 1, scratchpads_res + 1);
            }
        }

        // Account for dimensions that are not a multiple of 3
        if(m1.get_width() % 3 != 0) {
            naive_multiply_add(
                res.get_tile(0, 0, m1.get_height() - m1.get_height() % 3, m2.get_width() - m2.get_width() % 3), shift,
                m1.get_tile(0, m1.get_width() - m1.get_width() % 3, m1.get_height() - m1.get_height() % 3, m1.get_width() % 3),
                m2.get_tile(m2.get_height() - m2.get_height() % 3, 0, m2.get_height() % 3, m2.get_width() - m2.get_width() % 3));
        }
        if(m2.get_width() % 3 != 0) {
            naive_multiply_add(
                res.get_tile(0, res.get_width() - res.get_width() % 3, res.get_height() - res.get_height() % 3, res.get_width() % 3), shift,
                m1.get_tile(0, 0, m1.get_height() - m1.get_height() % 3, m1.get_width()),
                m2.get_tile(0, m2.get_width() - m2.get_width() % 3, m2.get_height(), m2.get_width() % 3));
        }
        if(m1.get_height() % 3 != 0) {
            naive_multiply_add(res.get_tile(res.get_height() - res.get_height() % 3, 0, res.get_height() % 3, res.get_width()), shift,
                               m1.get_tile(m1.get_height() - m1.get_height() % 3, 0, m1.get_height() % 3, m1.get_width()), m2);
        }
    }

    void multiply_tiles_overwrite(MatrixTile<Poly<T>> res, const MatrixTile<Poly<T>>& m1, const MatrixTile<Poly<T>>& m2,
                                  Matrix<Poly<T>>* scratchpads_left, Matrix<Poly<T>>* scratchpads_right, Matrix<Poly<T>>* scratchpads_res) {
        for(size_t i = 0; i < m1.get_height(); i++)
            for(size_t j = 0; j < m2.get_width(); j++)
                res[i][j] = static_cast<Poly<T>>(0);
        multiply_tiles_add(res, 0, m1, m2, scratchpads_left, scratchpads_right, scratchpads_res);
    }

private:
    void naive_multiply_add(MatrixTile<Poly<T>> res, int64_t shift, const MatrixTile<Poly<T>>& m1, const MatrixTile<Poly<T>>& m2) {
        for(size_t i = 0; i < m1.get_height(); i++)
            for(size_t j = 0; j < m1.get_width(); j++)
                for(size_t k = 0; k < m2.get_width(); k++)
                    res[i][k] += (m1[i][j] * m2[j][k]).power_shift(shift);
    }

    size_t estimate_impl(size_t n, size_t k, size_t m, size_t deg1, size_t deg2) {
        if(n == 0 || k == 0 || m == 0)
            return 0;

        size_t res = 0;

        size_t height_part_left = n / 3;
        size_t width_part_left = k / 3;
        size_t width_part_right = m / 3;

        if(n > 2 && k > 2 && m > 2) {
            // Compute p_{ij} and q_{ij}
            res += 12 * estimate_impl(height_part_left, width_part_left, width_part_right, deg1 + 2, deg2 + 1);

            // Compute r_j
            res += 3 * estimate_impl(height_part_left, width_part_left, width_part_right, deg1, deg2);

            // Compute p_{ii}
            res += 3 * estimate_impl(height_part_left, width_part_left, width_part_right, deg1 + 2, deg2);

            // Compute q_{ii}
            res += 3 * estimate_impl(height_part_left, width_part_left, width_part_right, deg1 + 2, deg2 + 2);
        }

        // Account for dimensions that are not a multiple of 3
        if(k % 3 != 0)
            res += ((n - n % 3) * (k % 3) * (m - m % 3)) * Poly<T>::estimate_multiplications_in_mul_operator(deg1, deg2);
        if(m % 3 != 0)
            res += ((n - n % 3) * k * (m % 3)) * Poly<T>::estimate_multiplications_in_mul_operator(deg1, deg2);
        if(n % 3 != 0)
            res += ((n % 3) * k * m) * Poly<T>::estimate_multiplications_in_mul_operator(deg1, deg2);
        return res;
    }
};
