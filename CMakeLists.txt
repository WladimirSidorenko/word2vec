CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11)

PROJECT(word2vec)
SET(word2vec_VERSION_MAJOR 0)
SET(word2vec_VERSION_MINOR 1)

FIND_PACKAGE (Threads)
FIND_PACKAGE(Doxygen QUIET)
IF(DOXYGEN_FOUND)
  SET(W2V_ASSET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/doc/static)
  CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  ADD_CUSTOM_TARGET(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
ENDIF(DOXYGEN_FOUND)

ADD_DEFINITIONS(-Wall -Wextra -funroll-loops -march=native -funroll-loops -Ofast)
IF(OPENMP_FOUND)
  ADD_DEFINITIONS(OpenMP_C_FLAGS)
ENDIF(OPENMP_FOUND)

# specify output directory
SET(EXECUTABLE_OUTPUT_PATH bin)
SET(W2V_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src
  CACHE FILEPATH "Default directory containing word2vec source files")

# define targets

## word2vec
FILE(GLOB W2V_SOURCES
  "${W2V_SRC_DIR}/*.h"
  "${W2V_SRC_DIR}/*.c"
  )
ADD_EXECUTABLE(word2vec ${W2V_SOURCES})
TARGET_COMPILE_OPTIONS(word2vec PRIVATE -pthread -O3 -march=native)
TARGET_INCLUDE_DIRECTORIES(word2vec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
TARGET_LINK_LIBRARIES(word2vec m pthread)

## tests
ENABLE_TESTING()
SET(W2V_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests
  CACHE FILEPATH "Default directory containing test files")
SET(W2V_BIN_DIR ${CMAKE_BINARY_DIR}/${EXECUTABLE_OUTPUT_PATH}
  CACHE FILEPATH "Default directory containing built executables")

ADD_TEST(NAME word2vec_compatibility
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
  COMMAND tap-driver.sh --test-name word2vec_compatibility
  --trs-file /dev/stdout --log-file /dev/stderr
  --color-tests yes -- ${W2V_TEST_DIR}/test_0.test ${W2V_BIN_DIR}/word2vec)

ADD_TEST(NAME task_specific
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
  COMMAND tap-driver.sh --test-name word2vec_compatibility
  --trs-file /dev/stdout --log-file /dev/stderr
  --color-tests yes -- ${W2V_TEST_DIR}/test_1.test ${W2V_BIN_DIR}/word2vec)

ADD_TEST(NAME task_specific_w2v
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
  COMMAND tap-driver.sh --test-name word2vec_compatibility
  --trs-file /dev/stdout --log-file /dev/stderr
  --color-tests yes -- ${W2V_TEST_DIR}/test_2.test ${W2V_BIN_DIR}/word2vec)

# ADD_TEST(NAME task_specific_least_sq
#   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
#   COMMAND tap-driver.sh --test-name word2vec_compatibility
#   --trs-file /dev/stdout --log-file /dev/stderr
#   --color-tests yes -- ${W2V_TEST_DIR}/test_3.test ${W2V_BIN_DIR}/word2vec)
