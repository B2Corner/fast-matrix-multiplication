#pragma once

#include <cstddef>
#include <cassert>
#include <initializer_list>
#include <algorithm>

// Forward declarations, needed for some helper classes
template<typename T>
class Matrix;

template<typename T>
class MatrixTile;

// Helper class for double indexing of matrices and matrix tiles, should not be used directly
template<typename T>
class MatrixTileRow {
    friend class Matrix<T>;
    friend class MatrixTile<T>;

public:
    T& operator[](size_t ind) {
        assert(ind < width_);
        return data_[ind];
    }

    const T& operator[](size_t ind) const {
        assert(ind < width_);
        return data_[ind];
    }

private:
    T* data_;
    size_t width_;

    // Matrix row cannot be constructed directly, only by operator[] of Matrix
    MatrixTileRow(T* data, size_t width) : data_(data), width_(width) {
    }
};

// A lighter version of matrix, in the sense it doesn't own the underlying memory
template<typename T>
class MatrixTile {
    friend class Matrix<T>;

public:
    MatrixTile() : height_(0), width_(0), data_(nullptr), hor_shift_(0) {
    }

    // We are OK with default rule-of-5 implementation

    // ---------- Operators ----------

    MatrixTileRow<T> operator[](size_t ind) {
        assert(ind < height_);
        return {data_[ind] + hor_shift_, width_};
    }

    const MatrixTileRow<T> operator[](size_t ind) const {
        assert(ind < height_);
        return {data_[ind] + hor_shift_, width_};
    }

    // ---------- Tile getting  ----------

    MatrixTile<T> get_tile(size_t row0, size_t col0, size_t tile_height, size_t tile_width) const {
        return {tile_height, tile_width, data_ + row0, hor_shift_ + col0};
    }

    // ---------- Getters ----------

    size_t get_width() const {
        return width_;
    }

    size_t get_height() const {
        return height_;
    }

    // ---------- Destructor ----------

    ~MatrixTile() = default;

private:
    size_t height_, width_;
    T** data_;
    size_t hor_shift_;

    MatrixTile(size_t height, size_t width, T** data, size_t hor_shift)
        : height_(height), width_(width), data_(data), hor_shift_(hor_shift) {
    }
};

template<typename T>
class Matrix {
public:
    // ---------- Constructors ----------

    Matrix() : height_(0), width_(0), data_(nullptr) {
    }

    Matrix(size_t height, size_t width) : height_(height), width_(width) {
        allocate_data();
    }

    Matrix(std::initializer_list<std::initializer_list<T>> initializer_list) {
        height_ = initializer_list.size();
        width_ = (height_ == 0 ? 0 : initializer_list.begin()->size());

        // In debug mode, check that all rows have the same length
#ifndef NDEBUG
        for(const auto& row : initializer_list)
            assert(row.size() == width_);
#endif

        // Allocate the memory
        allocate_data();

        // Fill the data
        size_t i = 0;
        for(const auto& row : initializer_list) {
            size_t j = 0;
            for(const auto& elem : row) {
                data_[i][j] = std::move(elem);
                j++;
            }
            i++;
        }
    }

    // ---------- Rule of 5 ----------

    Matrix& operator=(const Matrix& other) {
        if(this == &other)
            return *this;

        height_ = other.height_;
        width_ = other.width_;

        allocate_data();

        for(size_t i = 0; i < height_; i++)
            for(size_t j = 0; j < width_; j++)
                data_[i][j] == other.data_[i][j];

        return *this;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if(this == &other)
            return *this;

        std::swap(height_, other.height_);
        std::swap(width_, other.width_);
        std::swap(data_, other.data_);

        return *this;
    }

    Matrix(const Matrix& other) {
        height_ = other.height_;
        width_ = other.width_;

        allocate_data();

        for(size_t i = 0; i < height_; i++)
            for(size_t j = 0; j < width_; j++)
                data_[i][j] == other.data_[i][j];
    }

    Matrix(Matrix&& other) noexcept : Matrix() {
        *this = std::move(other);
    }

    // ---------- Operators ----------

    MatrixTileRow<T> operator[](size_t ind) {
        assert(ind < height_);
        return {data_[ind], width_};
    }

    const MatrixTileRow<T> operator[](size_t ind) const {
        assert(ind < height_);
        return {data_[ind], width_};
    }

    bool operator==(const Matrix& other) const {
        if(height_ != other.height_ || width_ != other.width_)
            return false;
        for(size_t i = 0; i < height_; i++)
            for(size_t j = 0; j < width_; j++)
                if(data_[i][j] != other[i][j])
                    return false;
        return true;
    }

    // ---------- Tile getting  ----------

    MatrixTile<T> get_tile(size_t row0, size_t col0, size_t tile_height, size_t tile_width) const {
        return {tile_height, tile_width, data_ + row0, col0};
    }

    operator MatrixTile<T>() const {
        return get_tile(0, 0, height_, width_);
    }

    // ---------- Getters ----------

    size_t get_width() const {
        return width_;
    }

    size_t get_height() const {
        return height_;
    }

    // ---------- Destructor ----------

    ~Matrix() {
        if(data_ != nullptr)
            delete[] data_[0];
        delete[] data_;
    }

private:
    size_t height_, width_;
    T** data_;

    // We need to handle OOM errors gracefully, so this is a separate function
    void allocate_data() {
        data_ = new T*[height_];

        T* buf;
        try {
            buf = new T[width_ * height_];
        } catch(...) {
            delete[] data_;
            throw;
        }

        for(size_t i = 0; i < height_; i++)
            data_[i] = buf + i * width_;
    }
};

template<typename T>
void add_tiles(MatrixTile<T>& t1, const MatrixTile<T>& t2) {
    assert(t1.get_width() == t2.get_width());
    assert(t1.get_height() == t2.get_height());

    for(size_t i = 0; i < t1.get_height(); i++)
        for(size_t j = 0; j < t1.get_width(); j++)
            t1[i][j] += t2[i][j];
}

template<typename T>
void add_tiles(MatrixTile<T>& res, const MatrixTile<T>& t1, const MatrixTile<T>& t2) {
    assert(t1.get_width() == t2.get_width());
    assert(t1.get_height() == t2.get_height());
    assert(t1.get_width() == res.get_width());
    assert(t1.get_height() == res.get_height());

    for(size_t i = 0; i < t1.get_height(); i++)
        for(size_t j = 0; j < t1.get_width(); j++)
            res[i][j] = t1[i][j] + t2[i][j];
}

template<typename T>
void subtract_tiles(MatrixTile<T>& t1, const MatrixTile<T>& t2) {
    assert(t1.get_width() == t2.get_width());
    assert(t1.get_height() == t2.get_height());

    for(size_t i = 0; i < t1.get_height(); i++)
        for(size_t j = 0; j < t1.get_width(); j++)
            t1[i][j] -= t2[i][j];
}

template<typename T>
void subtract_tiles(MatrixTile<T>& res, const MatrixTile<T>& t1, const MatrixTile<T>& t2) {
    assert(t1.get_width() == t2.get_width());
    assert(t1.get_height() == t2.get_height());
    assert(t1.get_width() == res.get_width());
    assert(t1.get_height() == res.get_height());

    for(size_t i = 0; i < t1.get_height(); i++)
        for(size_t j = 0; j < t1.get_width(); j++)
            res[i][j] = t1[i][j] - t2[i][j];
}

template<typename T>
void copy_tile(MatrixTile<T>& res, const MatrixTile<T>& t) {
    assert(t.get_width() == res.get_width());
    assert(t.get_height() == res.get_height());

    for(size_t i = 0; i < t.get_height(); i++)
        for(size_t j = 0; j < t.get_width(); j++)
            res[i][j] = t[i][j];
}