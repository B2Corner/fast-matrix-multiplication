project('fast_matrix_multuplucation', 'cpp')
# Setting the C++ standard should be in default_options,
# but a Meson bug open since 2019 prevents us from putting the setting there
if meson.get_compiler('cpp').get_id() == 'gcc' or meson.get_compiler('cpp').get_id() == 'clang'
  add_project_arguments('-std=c++20', language : 'cpp')
  add_project_arguments('-march=native', language : 'cpp')
endif

dependencies = []

dependencies += subproject('catch2').get_variable('catch2_with_main_dep')

test_files = [
    'tests/test_naive.cpp',
    'tests/test_strassen.cpp',
    'tests/test_shoenhage_simple.cpp',
    'tests/test_bitpacking.cpp',
    'tests/test_four_russians.cpp',
    'tests/test_bilinear_aggregation.cpp',
    'tests/test_tensors.cpp'
]
executable('test_fmm', test_files, dependencies : dependencies)

executable('collect_stats', 'collect_stats.cpp')
