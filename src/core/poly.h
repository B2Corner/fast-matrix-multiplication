#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

template<typename T>
class Poly {
public:
    Poly() : data_{0} {
    }

    Poly(T x) {
        data_.reserve(1);
        data_.push_back(std::move(x));
    }

    size_t degree() const {
        return data_.size() - 1;
    }

    Poly<T> power_shift(int64_t shift) const {
        if(shift > 0) {
            Poly<T> res = *this;
            for(int64_t i = 0; i < shift; i++)
                res.data_.insert(res.data_.begin(), T(0));
            return res;
        } else if(-shift < data_.size()) {
            Poly<T> res;
            res.data_ = {data_.begin() + (-shift), data_.end()};
            return res;
        } else {
            return {};
        }
    }

    Poly<T> operator+(const Poly<T>& other) const {
        Poly<T> res;
        res.data_.resize(std::max(data_.size(), other.data_.size()), T(0));
        for(size_t i = 0; i < data_.size(); i++)
            res.data_[i] += data_[i];
        for(size_t i = 0; i < other.data_.size(); i++)
            res.data_[i] += other.data_[i];
        return res;
    }

    Poly<T>& operator+=(const Poly& other) {
        *this = *this + other;
        return *this;
    }

    Poly<T> operator-(const Poly<T>& other) const {
        Poly<T> res;
        res.data_.resize(std::max(data_.size(), other.data_.size()), T(0));
        for(size_t i = 0; i < data_.size(); i++)
            res.data_[i] += data_[i];
        for(size_t i = 0; i < other.data_.size(); i++)
            res.data_[i] -= other.data_[i];
        return res;
    }

    Poly<T>& operator-=(const Poly& other) {
        *this = *this - other;
        return *this;
    }

    /// Uses Karatsuba multiplication; O(n^{log_2^3}) ~ O(n^1.585) time for two polynomials of degree n
    Poly<T> operator*(const Poly<T>& other) const {
        if(data_.size() == 1 || other.data_.size() == 1) {
            Poly<T> res;
            res.data_.resize(data_.size() + other.data_.size() - 1, T(0));
            for(size_t i = 0; i < data_.size(); i++)
                for(size_t j = 0; j < other.data_.size(); j++)
                    res.data_[i + j] += data_[i] * other.data_[j];
            return res;
        } else {
            size_t left_size = std::min(data_.size(), other.data_.size()) / 2;
            Poly x0 = Poly{data_.begin(), data_.begin() + left_size};
            Poly x1 = Poly{data_.begin() + left_size, data_.end()};
            Poly y0 = Poly{other.data_.begin(), other.data_.begin() + left_size};
            Poly y1 = Poly{other.data_.begin() + left_size, other.data_.end()};

            Poly<T> r0 = x0 * y0;
            Poly<T> r2 = x1 * y1;
            Poly<T> r3 = (x0 + x1) * (y0 + y1);
            Poly<T> r1 = r3 - r0 - r2;

            Poly<T> res;
            res.data_.resize(data_.size() + other.data_.size() - 1, T(0));
            res += r0;
            res += r1.power_shift(left_size);
            res += r2.power_shift(2 * left_size);

            return res;
        }
    }

    T evaluate_at_0() const {
        return data_[0];
    }

    static size_t estimate_multiplications_in_mul_operator(size_t deg1, size_t deg2) {
        if(deg1 == 0 || deg2 == 0)
            return (deg1 + 1) * (deg2 + 1);

        size_t left_size = std::min(deg1 + 1, deg2 + 1) / 2;
        size_t res = 0;
        res += estimate_multiplications_in_mul_operator(left_size - 1, left_size - 1);
        res += estimate_multiplications_in_mul_operator(deg1 - left_size, deg2 - left_size);
        res +=
            estimate_multiplications_in_mul_operator(std::max(left_size - 1, deg1 - left_size), std::max(left_size - 1, deg2 - left_size));
        return res;
    }

private:
    Poly(const std::vector<T>::const_iterator& it1, const std::vector<T>::const_iterator& it2) : data_(it1, it2) {
    }

    std::vector<T> data_;
};
