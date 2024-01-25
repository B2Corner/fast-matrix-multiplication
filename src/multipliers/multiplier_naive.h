#pragma once

#include "../core/mm.h"

/// Naive matrix multiplication
///
/// Complexity: O(n^3)
/// Extra memory: O(1)
template<typename T>
class MatrixMultiplierNaive : public MatrixMultiplier<T> {
public:
    void multiply_tiles(MatrixTile<T> res, const MatrixTile<T>& m1, const MatrixTile<T>& m2) override {
        for(size_t i = 0; i < m1.get_height(); i++)
            for(size_t j = 0; j < m2.get_width(); j++)
                res[i][j] = T(0);

        for(size_t i = 0; i < m1.get_height(); i++)
            for(size_t j = 0; j < m1.get_width(); j++) {
                for(size_t k = 0; k < m2.get_width(); k++)
                    res[i][k] += m1[i][j] * m2[j][k];
            }
    }

    void multiply_tiles_add(MatrixTile<T> res, const MatrixTile<T>& m1, const MatrixTile<T>& m2) {
        for(size_t i = 0; i < m1.get_height(); i++)
            for(size_t j = 0; j < m1.get_width(); j++) {
                for(size_t k = 0; k < m2.get_width(); k++)
                    res[i][k] += m1[i][j] * m2[j][k];
            }
    }

    size_t estimate_multiplications(size_t n, size_t k, size_t m) override {
        return n * k * m;
    }
};
