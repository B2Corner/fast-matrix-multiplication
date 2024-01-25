#include <iostream>
#include <iomanip>
#include "src/multipliers/multiplier_naive.h"
#include "src/multipliers/multiplier_shoenhage_simple.h"
#include "src/multipliers/multiplier_strassen.h"

namespace {

std::string nice_str(size_t num) {
    std::stringstream stream;
    stream << std::setprecision(6);
    stream << static_cast<double>(num);

    return stream.str();
}

template<typename T>
double get_ratio_with_naive(MatrixMultiplier<T>& multuplier, size_t size) {
    MatrixMultiplierNaive<T> multiplier_naive;
    return static_cast<double>(multuplier.estimate_multiplications(size, size, size)) /
           static_cast<double>(multiplier_naive.estimate_multiplications(size, size, size));
}

template<typename T>
void output_estimation_table_row(MatrixMultiplier<T>& multuplier, size_t size, size_t max_size_len) {
    std::cout << size << std::string(max_size_len - std::to_string(size).size(), ' ') << " | ";
    std::string operations_cnt_str = nice_str(multuplier.estimate_multiplications(size, size, size));
    std::cout << operations_cnt_str << std::string(33 - operations_cnt_str.size(), ' ') << " | ";
    std::cout << get_ratio_with_naive(multuplier, size) << std::endl;
}

}  // namespace

int main() {
    MatrixMultiplierNaive<int32_t> multiplier_naive;
    MatrixMultiplierStrassen<int32_t> multiplier_strassen;
    MatrixMultiplierShoenhageSimple<int32_t> multiplier_shoenhage_simple;

    std::cout << std::setprecision(6);

    // Run estimations of required number of operations for different multiplication algorithms
    std::cout << "Estimations for naive multiplier (square matrices)" << std::endl;
    std::cout << "n       | number of element multiplications | ratio w/ naive" << std::endl;
    output_estimation_table_row(multiplier_naive, 100, 7);
    output_estimation_table_row(multiplier_naive, 1000, 7);
    output_estimation_table_row(multiplier_naive, 10000, 7);
    output_estimation_table_row(multiplier_naive, 100000, 7);
    output_estimation_table_row(multiplier_naive, 1000000, 7);
    std::cout << std::endl;

    std::cout << "Estimations for strassen multiplier (square matrices)" << std::endl;
    std::cout << "n       | number of element multiplications | ratio w/ naive" << std::endl;
    output_estimation_table_row(multiplier_strassen, 100, 7);
    output_estimation_table_row(multiplier_strassen, 1000, 7);
    output_estimation_table_row(multiplier_strassen, 10000, 7);
    output_estimation_table_row(multiplier_strassen, 100000, 7);
    output_estimation_table_row(multiplier_strassen, 1000000, 7);
    std::cout << std::endl;

    std::cout << "Estimations for shoenhage simple multiplier (square matrices)" << std::endl;
    std::cout << "n       | number of element multiplications | ratio w/ naive" << std::endl;
    output_estimation_table_row(multiplier_shoenhage_simple, 100, 7);
    output_estimation_table_row(multiplier_shoenhage_simple, 1000, 7);
    output_estimation_table_row(multiplier_shoenhage_simple, 10000, 7);
    output_estimation_table_row(multiplier_shoenhage_simple, 100000, 7);
    output_estimation_table_row(multiplier_shoenhage_simple, 1000000, 7);
    std::cout << std::endl;
}
