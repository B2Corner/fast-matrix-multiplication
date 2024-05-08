#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <random>
#include "../src/multipliers/multiplier_naive.h"
#include "../src/multipliers/multiplier_four_russians.h"

TEST_CASE("Four russians multiplication stress") {
    MatrixMultiplierNaive<bool> multiplier_naive;
    MatrixMultiplierFourRussians<bool> multiplier_four_russians;

    std::uniform_int_distribution<size_t> dimensions_distrib(1, 100);
    std::uniform_int_distribution<int32_t> elems_distrib(0, 99);
    const int32_t percent_of_ones = 10;
    std::mt19937 rng;
    for(size_t t = 0; t < 100; t++) {
        auto n = dimensions_distrib(rng);
        auto k = dimensions_distrib(rng);
        auto m = dimensions_distrib(rng);

        Matrix<bool> m1(n, k);
        for(size_t i = 0; i < n; i++)
            for(size_t j = 0; j < k; j++)
                m1[i][j] = elems_distrib(rng) < percent_of_ones;

        Matrix<int32_t> m2(k, m);
        for(size_t i = 0; i < k; i++)
            for(size_t j = 0; j < m; j++)
                m2[i][j] = elems_distrib(rng) < percent_of_ones;

        Matrix<bool> expected = multiplier_naive.multiply(m1, m2);
        Matrix<bool> actual = multiplier_four_russians.multiply(m1, m2);
        REQUIRE(expected == actual);
    }
}

TEST_CASE("Four russians multiplication benchmark") {
    std::mt19937 rng;
    std::uniform_int_distribution<int32_t> elems_distrib(0, 99);
    const int32_t percent_of_ones = 50;

    size_t n, k, m;
    Matrix<int32_t> m1, m2;

    auto setup_square = [&](size_t x) {
        n = x;
        k = x;
        m = x;

        m1 = Matrix<int32_t>(n, k);
        for(size_t i = 0; i < n; i++)
            for(size_t j = 0; j < k; j++)
                m1[i][j] = elems_distrib(rng) < percent_of_ones;

        m2 = Matrix<int32_t>(k, m);
        for(size_t i = 0; i < k; i++)
            for(size_t j = 0; j < m; j++)
                m2[i][j] = elems_distrib(rng) < percent_of_ones;
    };
    MatrixMultiplierFourRussians<bool> multiplier_four_russians;

    std::array<size_t, 8> sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    for(size_t size : sizes) {
        std::string benchmark_name = std::to_string(size);
        benchmark_name = benchmark_name + " x " + benchmark_name + " x " + benchmark_name;

        setup_square(size);
        BENCHMARK(benchmark_name.c_str()) {
            return multiplier_four_russians.multiply(m1, m2);
        };
    }
}
