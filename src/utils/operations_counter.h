#pragma once

#include <cstddef>

template<typename T>
struct OperationsCounter {
private:
    T val_;

public:
    static size_t mul_counter;

    OperationsCounter() = default;

    OperationsCounter(const T& val) : val_(val) {
    }

    OperationsCounter operator-() const {
        return {-val_};
    }

    OperationsCounter operator+(const OperationsCounter& other) const {
        return {val_ + other.val_};
    }

    OperationsCounter& operator+=(const OperationsCounter& other) {
        val_ += other.val_;
        return *this;
    }

    OperationsCounter operator-(const OperationsCounter& other) const {
        return {val_ - other.val_};
    }

    OperationsCounter& operator-=(const OperationsCounter& other) {
        val_ -= other.val_;
        return *this;
    }

    OperationsCounter operator*(const OperationsCounter& other) const {
        mul_counter++;
        return {val_ * other.val_};
    }

    OperationsCounter operator*=(const OperationsCounter& other) {
        mul_counter++;
        val_ *= other.val_;
    }
};

template<typename T>
size_t OperationsCounter<T>::mul_counter = 0;
