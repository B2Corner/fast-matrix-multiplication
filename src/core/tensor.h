#pragma once

#include <vector>

struct Size3D {
    size_t n, k, m;

    inline bool operator==(const Size3D& other) const {
        return n == other.n && k == other.k && m == other.m;
    }
};

// 3D tensor, stored as its sparse canonical decomposition
template<typename T>
class Tensor {
public:
    struct DecompositionElement {
        std::vector<std::pair<size_t, T>> coeffs_1;
        std::vector<std::pair<size_t, T>> coeffs_2;
        std::vector<std::pair<size_t, T>> coeffs_res;
    };

    Tensor(Size3D size, std::vector<DecompositionElement> decomposition)
        : n_(size.n), k_(size.k), m_(size.m), decomposition_(std::move(decomposition)) {
    }

    Size3D sizes() const {
        return {n_, k_, m_};
    }

    size_t rank() const {
        return decomposition_.size();
    }

    std::vector<DecompositionElement>& get_decomposition() {
        return decomposition_;
    }

private:
    size_t n_, k_, m_;
    std::vector<DecompositionElement> decomposition_;
};
