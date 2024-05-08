#pragma once

#include "../core/mm.h"
#include "multiplier_naive.h"
#include "../core/tensor.h"
#include <unordered_map>
#include <iostream>

/// An algorithm that uses known MM tensor decompositions to construct a close-to-optimal scheme for a given problem size.
///
/// Complexity: depends on the provided tensors; approx. O(n^{2.75}) - O(n^{2.8}) for reasonable decompositions
/// Extra memory: depends on the provided tensors
template<typename T, typename U>
class MatrixMultiplierTensors : public MatrixMultiplier<T> {
public:
    MatrixMultiplierTensors(const std::vector<Tensor<U>> tensors, size_t fallback_size = 0)
        : fallback_size_(fallback_size), tensors_(std::move(tensors)) {
    }

    void multiply_tiles(MatrixTile<T> res, const MatrixTile<T>& m1, const MatrixTile<T>& m2) override {
        DecisionCache cache;
        multiply_tiles_impl(res, m1, m2, cache);
    }

    size_t estimate_multiplications(size_t n, size_t k, size_t m) override {
        DecisionCache cache;
        auto [tensor_index, num_multiplications] = find_best_tensor(n, k, m, cache);
        return num_multiplications;
    }

private:
    struct RecursiveStepDecision {
        size_t tensor_index;
        size_t num_multiplications;
    };

    struct SizeHasher {
    public:
        size_t operator()(const Size3D& size) const {
            return size.n * 1'000'000'007 + size.k * 1'000'000'009 + size.m * 10'000'007;
        }
    };

    using DecisionCache = std::unordered_map<Size3D, RecursiveStepDecision, SizeHasher>;

    RecursiveStepDecision find_best_tensor(size_t n, size_t k, size_t m, DecisionCache& cache) {
        if(n * k * m <= fallback_size_ * fallback_size_ * fallback_size_)
            return {tensors_.size(), n * k * m};
        auto it = cache.find({n, k, m});
        if(it != cache.end())
            return it->second;

        size_t best_ops = n * k * m;
        size_t best_index = tensors_.size();
        for(size_t i = 0; i < tensors_.size(); i++) {
            auto [tensor_n, tensor_m, tensor_k] = tensors_[i].sizes();
            if(tensor_n > n || tensor_m > m || tensor_k > k)
                continue;

            size_t cur_ops = find_best_tensor(n / tensor_n, k / tensor_k, m / tensor_m, cache).num_multiplications * tensors_[i].rank();
            cur_ops += (n % tensor_n) * k * (m - m % tensor_m);
            cur_ops += n * k * (m % tensor_m);
            cur_ops += (n - n % tensor_n) * (k % tensor_k) * (m - m % tensor_m);
            if(cur_ops < best_ops) {
                best_ops = cur_ops;
                best_index = i;
            }
        }
        cache[{n, k, m}] = {best_index, best_ops};
        return {best_index, best_ops};
    }

    void multiply_tiles_impl(MatrixTile<T> res, const MatrixTile<T>& m1, const MatrixTile<T>& m2, DecisionCache& cache) {
        size_t n = m1.get_height();
        size_t k = m1.get_width();
        size_t m = m2.get_width();

        auto [tensor_index, num_multiplications] = find_best_tensor(n, k, m, cache);
        if(tensor_index == tensors_.size()) {
            multiplier_naive_.multiply_tiles(res, m1, m2);
            return;
        }
        Tensor<U>& best_tensor = tensors_[tensor_index];
        auto [tensor_n, tensor_k, tensor_m] = best_tensor.sizes();

        auto get_tiles = [](MatrixTile<T> src, size_t parts_ver, size_t parts_hor) {
            std::vector<std::vector<MatrixTile<T>>> tiles(parts_ver);
            size_t tile_height = src.get_height() / parts_ver;
            size_t tile_width = src.get_width() / parts_hor;
            for(size_t i = 0; i < parts_ver; i++) {
                tiles[i].resize(parts_hor);
                for(size_t j = 0; j < parts_hor; j++)
                    tiles[i][j] = src.get_tile(tile_height * i, tile_width * j, tile_height, tile_width);
            }
            return tiles;
        };

        std::vector<std::vector<MatrixTile<T>>> res_tiles = get_tiles(res, tensor_n, tensor_m);
        std::vector<std::vector<MatrixTile<T>>> m1_tiles = get_tiles(m1, tensor_n, tensor_k);
        std::vector<std::vector<MatrixTile<T>>> m2_tiles = get_tiles(m2, tensor_k, tensor_m);

        evaluate_tensor(res_tiles, m1_tiles, m2_tiles, best_tensor, cache);

        // Account for the stuff that did not make it to the recursive calls
        if(n % tensor_n != 0) {
            multiplier_naive_.multiply_tiles(res.get_tile(n - n % tensor_n, 0, n % tensor_n, m - m % tensor_m),
                                             m1.get_tile(n - n % tensor_n, 0, n % tensor_n, k), m2.get_tile(0, 0, k, m - m % tensor_m));
        }
        if(m % tensor_m != 0) {
            multiplier_naive_.multiply_tiles(res.get_tile(0, m - m % tensor_m, n, m % tensor_m), m1.get_tile(0, 0, n, k),
                                             m2.get_tile(0, m - m % tensor_m, k, m % tensor_m));
        }
        if(k % tensor_k != 0) {
            multiplier_naive_.multiply_tiles_add(res.get_tile(0, 0, n - n % tensor_n, m - m % tensor_m),
                                                 m1.get_tile(0, k - k % tensor_k, n - n % tensor_n, k % tensor_k),
                                                 m2.get_tile(k - k % tensor_k, 0, k % tensor_k, m - m % tensor_m));
        }
    }

    void copy_with_coeff(MatrixTile<T> to, const MatrixTile<T> from, U coeff) {
        assert(to.get_height() == from.get_height() && to.get_width() == from.get_width());
        if(coeff == 1) {
            for(size_t i = 0; i < to.get_height(); i++)
                for(size_t j = 0; j < to.get_width(); j++)
                    to[i][j] = from[i][j];
        } else if(coeff == -1) {
            for(size_t i = 0; i < to.get_height(); i++)
                for(size_t j = 0; j < to.get_width(); j++)
                    to[i][j] = -from[i][j];
        } else {
            for(size_t i = 0; i < to.get_height(); i++)
                for(size_t j = 0; j < to.get_width(); j++)
                    to[i][j] = from[i][j] * coeff;
        }
    }

    void add_with_coeff(MatrixTile<T> to, const MatrixTile<T> from, U coeff) {
        assert(to.get_height() == from.get_height() && to.get_width() == from.get_width());
        if(coeff == 1) {
            for(size_t i = 0; i < to.get_height(); i++)
                for(size_t j = 0; j < to.get_width(); j++)
                    to[i][j] += from[i][j];
        } else if(coeff == -1) {
            for(size_t i = 0; i < to.get_height(); i++)
                for(size_t j = 0; j < to.get_width(); j++)
                    to[i][j] -= from[i][j];
        } else {
            for(size_t i = 0; i < to.get_height(); i++)
                for(size_t j = 0; j < to.get_width(); j++)
                    to[i][j] += from[i][j] * coeff;
        }
    }

    void add_with_coeffs(MatrixTile<T> to, const MatrixTile<T> m1, U coeff1, const MatrixTile<T> m2, U coeff2) {
        /*assert(to.get_height() == m1.get_height() && to.get_width() == m1.get_width());
        assert(to.get_height() == m2.get_height() && to.get_width() == m2.get_width());
        for(size_t i = 0; i < to.get_height(); i++)
            for(size_t j = 0; j < to.get_width(); j++)
                to[i][j] = m1[i][j] * coeff1 + m2[i][j] * coeff2;*/
        copy_with_coeff(to, m1, coeff1);
        add_with_coeff(to, m2, coeff2);
    }

    void evaluate_tensor(std::vector<std::vector<MatrixTile<T>>>& res_tiles, const std::vector<std::vector<MatrixTile<T>>>& m1_tiles,
                         const std::vector<std::vector<MatrixTile<T>>>& m2_tiles, Tensor<U>& tensor, DecisionCache& cache) {
        std::vector<std::vector<bool>> first_write(res_tiles.size());
        for(size_t i = 0; i < res_tiles.size(); i++)
            first_write[i].resize(res_tiles[i].size(), true);

        auto sum_all = [&](Matrix<T>& res, const std::vector<std::vector<MatrixTile<T>>>& tiles,
                           const std::vector<std::pair<size_t, U>>& coeffs) {
            if(coeffs.size() == 1)
                copy_with_coeff(res.get_tile(), tiles[coeffs[0].first / tiles[0].size()][coeffs[0].first % tiles[0].size()],
                                coeffs[0].second);
            else {
                add_with_coeffs(res.get_tile(), tiles[coeffs[0].first / tiles[0].size()][coeffs[0].first % tiles[0].size()],
                                coeffs[0].second, tiles[coeffs[1].first / tiles[0].size()][coeffs[1].first % tiles[0].size()],
                                coeffs[1].second);
                for(size_t i = 2; i < coeffs.size(); i++)
                    add_with_coeff(res.get_tile(), tiles[coeffs[i].first / tiles[0].size()][coeffs[i].first % tiles[0].size()],
                                   coeffs[i].second);
            }
        };

        auto add_to_all = [&](const Matrix<T>& res, std::vector<std::vector<MatrixTile<T>>>& tiles,
                              const std::vector<std::pair<size_t, U>>& coeffs) {
            for(size_t i = 0; i < coeffs.size(); i++) {
                MatrixTile<T>& target_tile = tiles[coeffs[i].first / tiles[0].size()][coeffs[i].first % tiles[0].size()];
                if(first_write[coeffs[i].first / tiles[0].size()][coeffs[i].first % tiles[0].size()]) {
                    copy_with_coeff(target_tile, res.get_tile(), coeffs[i].second);
                    first_write[coeffs[i].first / tiles[0].size()][coeffs[i].first % tiles[0].size()] = false;
                } else
                    add_with_coeff(target_tile, res.get_tile(), coeffs[i].second);
            }
        };

        Matrix<T> tmp_res(res_tiles[0][0].get_height(), res_tiles[0][0].get_width());
        Matrix<T> tmp_m1(m1_tiles[0][0].get_height(), m1_tiles[0][0].get_width());
        Matrix<T> tmp_m2(m2_tiles[0][0].get_height(), m2_tiles[0][0].get_width());
        for(size_t t = 0; t < tensor.rank(); t++) {
            typename Tensor<U>::DecompositionElement& element = tensor.get_decomposition()[t];

            sum_all(tmp_m1, m1_tiles, element.coeffs_1);
            sum_all(tmp_m2, m2_tiles, element.coeffs_2);
            multiply_tiles_impl(tmp_res, tmp_m1, tmp_m2, cache);
            add_to_all(tmp_res, res_tiles, element.coeffs_res);
        }
    }

    size_t fallback_size_;
    std::vector<Tensor<U>> tensors_;
    MatrixMultiplierNaive<T> multiplier_naive_;
};
