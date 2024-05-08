#pragma once

#include "poly.h"
#include "matrix.h"

template<int64_t shift1, int64_t shift2, typename T>
void add_poly_tiles(MatrixTile<Poly<T>>& res, const MatrixTile<Poly<T>>& t1, const MatrixTile<Poly<T>>& t2) {
    assert(t1.get_width() == t2.get_width());
    assert(t1.get_height() == t2.get_height());
    assert(t1.get_width() == res.get_width());
    assert(t1.get_height() == res.get_height());

    for(size_t i = 0; i < t1.get_height(); i++)
        for(size_t j = 0; j < t1.get_width(); j++)
            res[i][j] = t1[i][j].power_shift(shift1) + t2[i][j].power_shift(shift2);
}

template<int64_t shift1, int64_t shift2, typename T>
void subtract_poly_tiles(MatrixTile<Poly<T>>& res, const MatrixTile<Poly<T>>& t1, const MatrixTile<Poly<T>>& t2) {
    assert(t1.get_width() == t2.get_width());
    assert(t1.get_height() == t2.get_height());
    assert(t1.get_width() == res.get_width());
    assert(t1.get_height() == res.get_height());

    for(size_t i = 0; i < t1.get_height(); i++)
        for(size_t j = 0; j < t1.get_width(); j++)
            res[i][j] = t1[i][j].power_shift(shift1) - t2[i][j].power_shift(shift2);
}

template<typename T>
void add_poly_tiles(MatrixTile<Poly<T>>& res, int64_t shift, const MatrixTile<Poly<T>>& t) {
    assert(res.get_width() == t.get_width());
    assert(res.get_height() == t.get_height());

    for(size_t i = 0; i < t.get_height(); i++)
        for(size_t j = 0; j < t.get_width(); j++)
            res[i][j] += t[i][j].power_shift(shift);
}

template<typename T>
void subtract_poly_tiles(MatrixTile<Poly<T>>& res, int64_t shift, const MatrixTile<Poly<T>>& t) {
    assert(res.get_width() == t.get_width());
    assert(res.get_height() == t.get_height());

    for(size_t i = 0; i < t.get_height(); i++)
        for(size_t j = 0; j < t.get_width(); j++)
            res[i][j] -= t[i][j].power_shift(shift);
}

template<typename T>
void copy_poly_tiles(MatrixTile<Poly<T>>& res, int64_t shift, const MatrixTile<Poly<T>>& t) {
    assert(res.get_width() == t.get_width());
    assert(res.get_height() == t.get_height());

    for(size_t i = 0; i < t.get_height(); i++)
        for(size_t j = 0; j < t.get_width(); j++)
            res[i][j] = t[i][j].power_shift(shift);
}
