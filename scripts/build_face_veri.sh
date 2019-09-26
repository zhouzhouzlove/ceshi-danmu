#!/usr/bin/env sh
set -e

if [ -z "$NDK_ROOT" ] && [ "$#" -eq 0 ]; then
    echo 'Either $NDK_ROOT should be set or provided as argument'
    echo "e.g., 'export NDK_ROOT=/path/to/ndk' or"
    echo "      '${0} /path/to/ndk'"
    exit 1
else
    NDK_ROOT="${1:-${NDK_ROOT}}"
fi

ANDROID_ABI="arm64-v8a"
#ANDROID_ABI="armeabi-v7a"

WD=$(readlink -f "`dirname $0`/..")
N_JOBS=${N_JOBS:-4}
INFERXLITE_DIR=${WD}
BUILD_DIR=${INFERXLITE_DIR}/build

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

OPENCV_ROOT="/home/bzhang/work/Git/caffe-android-lib/android_lib/opencv/sdk/native/jni"

cmake -DCMAKE_TOOLCHAIN_FILE="${WD}/android-cmake/android.toolchain.cmake" \
      -DANDROID_NDK="${NDK_ROOT}" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DOpenCV_DIR="${OPENCV_ROOT}"\
      -DANDROID_ABI="${ANDROID_ABI}"\
      -DANDROID_NATIVE_API_LEVEL=24 \
      ..

make -j${N_JOBS}

make install/strip

cd "${WD}"
rm -rf "${BUILD_DIR}"

