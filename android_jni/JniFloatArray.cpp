#include "JniFloatArray.h"
#include <cstddef>
JniFloatArray::JniFloatArray(JNIEnv *env, jfloatArray jniPtr) {
    this->env = env;
    this->jniPtr = jniPtr;
    this->len = 0;
    this->pArray = NULL;
    if (NULL != jniPtr) {
        len = env->GetArrayLength(jniPtr);
        pArray = env->GetFloatArrayElements(jniPtr, NULL);
    }
}

JniFloatArray::~JniFloatArray() {
    if (pArray != NULL && jniPtr != NULL) {
        env->ReleaseFloatArrayElements(jniPtr, pArray, 0);
    }
}

int JniFloatArray::size() {
    return len;
}

float *JniFloatArray::get() {
    return this->pArray;
}

float &JniFloatArray::get(int k) {
    return this->pArray[k];
}

float &JniFloatArray::operator[](int k) {
    return get(k);
}
