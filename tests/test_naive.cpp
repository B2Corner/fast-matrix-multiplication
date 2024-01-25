#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <random>
#include "../src/utils/operations_counter.h"
#include "../src/multipliers/multiplier_naive.h"

TEST_CASE("Naive multiplication correctness") {
    Matrix<int32_t> m1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    Matrix<int32_t> m2 = {{7, -2, 5}, {5, 4, 9}, {1, 10, -3}};

    MatrixMultiplierNaive<int32_t> multiplier;
    Matrix<int32_t> actual = multiplier.multiply(m1, m2);

    Matrix<int32_t> expected = {{20, 36, 14}, {59, 72, 47}, {98, 108, 80}};
    REQUIRE(actual == expected);
}

TEST_CASE("Naive multiplication estimation correctness") {
    MatrixMultiplierNaive<OperationsCounter<int32_t>> multiplier_naive;

    std::uniform_int_distribution<size_t> dimensions_distrib(1, 100);
    std::uniform_int_distribution<int32_t> elems_distrib(-100, 100);
    std::mt19937 rng;
    for(size_t t = 0; t < 100; t++) {
        auto n = dimensions_distrib(rng);
        auto k = dimensions_distrib(rng);
        auto m = dimensions_distrib(rng);

        Matrix<OperationsCounter<int32_t>> m1(n, k);
        for(size_t i = 0; i < n; i++)
            for(size_t j = 0; j < k; j++)
                m1[i][j] = elems_distrib(rng);

        Matrix<OperationsCounter<int32_t>> m2(k, m);
        for(size_t i = 0; i < k; i++)
            for(size_t j = 0; j < m; j++)
                m2[i][j] = elems_distrib(rng);

        size_t before = OperationsCounter<int32_t>::mul_counter;
        auto m_res = multiplier_naive.multiply(m1, m2);
        size_t after = OperationsCounter<int32_t>::mul_counter;
        size_t expected = after - before;

        size_t actual = multiplier_naive.estimate_multiplications(n, k, m);
        REQUIRE(expected == actual);
    }
}

TEST_CASE("Naive multiplication benchmark") {
    std::mt19937 rng;
    std::uniform_int_distribution<size_t> elems_distrib(-10, 10);

    size_t n, k, m;
    Matrix<int32_t> m1, m2;

    auto setup_square = [&](size_t x) {
        n = x;
        k = x;
        m = x;

        m1 = Matrix<int32_t>(n, k);
        for(size_t i = 0; i < n; i++)
            for(size_t j = 0; j < k; j++)
                m1[i][j] = elems_distrib(rng);

        m2 = Matrix<int32_t>(k, m);
        for(size_t i = 0; i < k; i++)
            for(size_t j = 0; j < m; j++)
                m2[i][j] = elems_distrib(rng);
    };
    MatrixMultiplierNaive<int32_t> multiplier_naive;

    std::array<size_t, 5> sizes = {64, 128, 256, 512, 1024};
    for(size_t size : sizes) {
        std::string benchmark_name = std::to_string(size);
        benchmark_name = benchmark_name + " x " + benchmark_name + " x " + benchmark_name;

        setup_square(size);
        BENCHMARK(benchmark_name.c_str()) {
            return multiplier_naive.multiply(m1, m2);
        };
    }
}
