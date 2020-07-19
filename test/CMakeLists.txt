message(STATUS ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(DFE_MLIR_TEST_DEPENDS
  FileCheck count not
  dfe-opt
  dfe-translate
  )

add_lit_testsuite(check-dfe-mlir "Running the dfe-mlir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DFE_MLIR_TEST_DEPENDS}
  )
set_target_properties(check-dfe-mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(DFE_MLIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${DFE_MLIR_TEST_DEPS}
)