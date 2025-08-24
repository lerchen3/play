#!/bin/bash

# Build script for Active Inference MCTS C++ module
# This script compiles the C++ code into a Python extension module

set -e  # Exit on any error

echo "Building Active Inference MCTS C++ Module..."

# Check if we're on Windows (Git Bash/MSYS2/Cygwin)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    IS_WINDOWS=true
    echo "Detected Windows environment"
else
    IS_WINDOWS=false
    echo "Detected Unix-like environment"
fi

# Configuration
PROJECT_NAME="mcts_cpp_active"
OUTPUT_DIR="build_active"
SOURCES="mcts_active_inference.cpp"

# Check for required files
required_files=("mcts_active_inference.cpp" "mcts_active_inference.h")
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: Required file $file not found!"
        echo "Make sure you have all the necessary source files."
        exit 1
    fi
done

# Check for dependencies
echo "Checking dependencies..."

# Check for Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.7+."
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Check for pybind11
echo "Checking for pybind11..."
if ! $PYTHON_CMD -c "import pybind11" 2>/dev/null; then
    echo "pybind11 not found. Installing..."
    $PYTHON_CMD -m pip install pybind11
fi

# Check for numpy
echo "Checking for numpy..."
if ! $PYTHON_CMD -c "import numpy" 2>/dev/null; then
    echo "numpy not found. Installing..."
    $PYTHON_CMD -m pip install numpy
fi

# Create build directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Get Python and pybind11 configuration
echo "Getting build configuration..."

PYTHON_INCLUDE=$($PYTHON_CMD -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB=$($PYTHON_CMD -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYBIND11_INCLUDE=$($PYTHON_CMD -c "import pybind11; print(pybind11.get_cmake_dir())")

if [[ -z "$PYTHON_INCLUDE" ]]; then
    echo "Error: Could not find Python include directory"
    exit 1
fi

echo "Python include: $PYTHON_INCLUDE"
echo "Python lib: $PYTHON_LIB"

# Set compiler flags
if [[ "$IS_WINDOWS" == true ]]; then
    # Windows-specific flags
    CXX_FLAGS="-std=c++17 -O3 -Wall -shared -fPIC"
    PYTHON_FLAGS="-I$PYTHON_INCLUDE"
    PYBIND11_FLAGS=$($PYTHON_CMD -m pybind11 --includes)
    LINK_FLAGS=""
    
    # Windows library extension
    if [[ "$PYTHON_CMD" == *"python3"* ]]; then
        PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
        LIB_EXT=".cp${PYTHON_VERSION}-win_amd64.pyd"
    else
        LIB_EXT=".pyd"
    fi
else
    # Unix-like flags
    CXX_FLAGS="-std=c++17 -O3 -Wall -shared -fPIC"
    PYTHON_FLAGS="-I$PYTHON_INCLUDE"
    PYBIND11_FLAGS=$($PYTHON_CMD -m pybind11 --includes)
    
    # Platform-specific library extension
    if [[ "$OSTYPE" == "darwin"* ]]; then
        LIB_EXT=".so"
        LINK_FLAGS="-undefined dynamic_lookup"
    else
        LIB_EXT=".so"
        LINK_FLAGS=""
    fi
fi

echo "Compiler flags: $CXX_FLAGS"
echo "Python flags: $PYTHON_FLAGS"
echo "Pybind11 flags: $PYBIND11_FLAGS"
echo "Link flags: $LINK_FLAGS"
echo "Library extension: $LIB_EXT"

# Check for C++ compiler
if command -v g++ &> /dev/null; then
    CXX="g++"
elif command -v clang++ &> /dev/null; then
    CXX="clang++"
elif command -v cl &> /dev/null; then
    CXX="cl"
else
    echo "Error: No C++ compiler found. Please install g++, clang++, or MSVC."
    exit 1
fi

echo "Using C++ compiler: $CXX"

# Create pybind11 wrapper (if it doesn't exist)
WRAPPER_FILE="../mcts_active_wrapper.cpp"
if [[ ! -f "$WRAPPER_FILE" ]]; then
    echo "Creating pybind11 wrapper..."
    cat > "$WRAPPER_FILE" << 'EOF'
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "mcts_active_inference.h"

namespace py = pybind11;

PYBIND11_MODULE(mcts_cpp_active, m) {
    m.doc() = "Active Inference MCTS for Chess";
    
    // TrainingData struct
    py::class_<TrainingData>(m, "TrainingData")
        .def(py::init<>())
        .def_readwrite("fen", &TrainingData::fen)
        .def_readwrite("policy_target", &TrainingData::policy_target)
        .def_readwrite("value_target", &TrainingData::value_target);
    
    // MCTSNodeActive class
    py::class_<MCTSNodeActive>(m, "MCTSNodeActive")
        .def(py::init<const Position&, MCTSNodeActive*, double>(),
             py::arg("game_state"), py::arg("parent") = nullptr, py::arg("prior") = 0.0)
        .def("collectTrainingData", &MCTSNodeActive::collectTrainingData)
        .def("getQValue", &MCTSNodeActive::getQValue);
    
    // MCTSActive class
    py::class_<MCTSActive>(m, "MCTSActive")
        .def(py::init<Net*>())
        .def("search", &MCTSActive::search)
        .def("trainNetwork", &MCTSActive::trainNetwork)
        .def("clearTrainingBuffer", &MCTSActive::clearTrainingBuffer)
        .def("getTrainingBufferSize", &MCTSActive::getTrainingBufferSize);
    
    // Helper functions
    m.def("initMoveEncodingActive", &initMoveEncodingActive);
    m.def("moveToUCIActive", &moveToUCIActive);
    m.def("uciToMoveActive", &uciToMoveActive);
}
EOF
fi

# Compile the module
echo "Compiling C++ module..."

# Copy source files to build directory
cp "../$SOURCES" .
cp "../mcts_active_inference.h" .
cp "../mcts.h" .
cp "../position.h" .
cp "../config.h" .
cp "../mcts.cpp" .
cp "../position.cpp" .
cp "../config.cpp" .
cp "../tables.h" .
cp "../tables.cpp" .
cp "../c_api.h" .
cp "../c_api.cpp" .
cp "../types.h" .
cp "$WRAPPER_FILE" .

# Build command
OUTPUT_FILE="${PROJECT_NAME}${LIB_EXT}"

if [[ "$CXX" == "cl" ]]; then
    # MSVC compilation (Windows)
    echo "Using MSVC compiler..."
    cl /EHsc /std:c++17 /O2 /LD /Fe:"$OUTPUT_FILE" \
       mcts_active_inference.cpp mcts_active_wrapper.cpp \
       mcts.cpp position.cpp config.cpp tables.cpp c_api.cpp \
       /I"$PYTHON_INCLUDE" $PYBIND11_FLAGS $LINK_FLAGS
else
    # GCC/Clang compilation
    $CXX $CXX_FLAGS $PYTHON_FLAGS $PYBIND11_FLAGS \
         -o "$OUTPUT_FILE" \
         mcts_active_inference.cpp mcts_active_wrapper.cpp \
         mcts.cpp position.cpp config.cpp tables.cpp c_api.cpp \
         $LINK_FLAGS
fi

if [[ $? -eq 0 ]]; then
    echo "✓ Compilation successful!"
    
    # Copy the compiled module to the parent directory
    cp "$OUTPUT_FILE" ../
    
    echo "✓ Module copied to parent directory: $OUTPUT_FILE"
    
    # Test the module
    echo "Testing the compiled module..."
    cd ..
    if $PYTHON_CMD -c "import $PROJECT_NAME; print('✓ Module import successful!')"; then
        echo "✓ Active Inference MCTS module is ready to use!"
        echo ""
        echo "You can now run:"
        echo "  python train_active.py"
        echo ""
        echo "Or import in Python:"
        echo "  import $PROJECT_NAME"
    else
        echo "✗ Module import failed. Check for missing dependencies."
        exit 1
    fi
    
else
    echo "✗ Compilation failed!"
    echo "Check the error messages above for details."
    exit 1
fi

# Cleanup option
echo ""
# Non-interactive (e.g., Kaggle notebook) - clean up automatically
echo "Non-interactive environment detected. Cleaning up build directory automatically..."
cd ..
rm -rf "$OUTPUT_DIR"
echo "✓ Build directory cleaned up."


echo "Build process completed!" 