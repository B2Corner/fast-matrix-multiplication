#pragma once

#include "matrix.h"

template<typename T>
class MatrixMultiplier {
public:
    Matrix<T> multiply(const Matrix<T>& m1, const Matrix<T>& m2) {
        assert(m1.get_width() == m2.get_height());

        Matrix<T> res(m1.get_height(), m2.get_width());
        multiply_tiles(res, m1, m2);

        return res;
    }

    virtual void multiply_tiles(MatrixTile<T> res, const MatrixTile<T>& m1, const MatrixTile<T>& m2) = 0;

    virtual size_t estimate_multiplications(size_t n, size_t k, size_t m) = 0;
};
