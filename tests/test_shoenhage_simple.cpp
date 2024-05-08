#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <random>
#include "../src/utils/operations_counter.h"
#include "../src/multipliers/multiplier_naive.h"
#include "../src/multipliers/multiplier_shoenhage_simple.h"

TEST_CASE("Shoenhage simple multiplication stress") {
    MatrixMultiplierNaive<int32_t> multiplier_naive;
    MatrixMultiplierShoenhageSimple<int32_t> multiplier_shoenhage_simple;

    std::uniform_int_distribution<size_t> dimensions_distrib(1, 100);
    std::uniform_int_distribution<int32_t> elems_distrib(-100, 100);
    std::mt19937 rng;
    for(size_t t = 0; t < 100; t++) {
        auto n = dimensions_distrib(rng);
        auto k = dimensions_distrib(rng);
        auto m = dimensions_distrib(rng);

        Matrix<int32_t> m1(n, k);
        for(size_t i = 0; i < n; i++)
            for(size_t j = 0; j < k; j++)
                m1[i][j] = elems_distrib(rng);

        Matrix<int32_t> m2(k, m);
        for(size_t i = 0; i < k; i++)
            for(size_t j = 0; j < m; j++)
                m2[i][j] = elems_distrib(rng);

        Matrix<int32_t> expected = multiplier_naive.multiply(m1, m2);
        Matrix<int32_t> actual = multiplier_shoenhage_simple.multiply(m1, m2);
        REQUIRE(expected == actual);
    }
}

TEST_CASE("Shoenhage multiplication estimation correctness") {
    MatrixMultiplierShoenhageSimple<OperationsCounter<int32_t>> multiplier_shoenhage_simple;

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
        auto m_res = multiplier_shoenhage_simple.multiply(m1, m2);
        size_t after = OperationsCounter<int32_t>::mul_counter;
        size_t expected = after - before;

        size_t actual = multiplier_shoenhage_simple.estimate_multiplications(n, k, m);
        REQUIRE(expected == actual);
    }
}

TEST_CASE("Shoenhage simple multiplication benchmark") {
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
    MatrixMultiplierShoenhageSimple<int32_t> multiplier_shoenhage_simple;

    setup_square(64);
    BENCHMARK("64 x 64 x 64") {
        return multiplier_shoenhage_simple.multiply(m1, m2);
    };
}
