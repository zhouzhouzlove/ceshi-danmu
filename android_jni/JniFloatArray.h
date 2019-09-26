#ifndef JNIFLOATARRAY_H
#define JNIFLOATARRAY_H

#include <jni.h>

class JniFloatArray {
private:
    JNIEnv *env;
    jfloatArray jniPtr;
    float *pArray;
    int len;

public:
    JniFloatArray(JNIEnv *env, jfloatArray jniPtr);

    ~JniFloatArray();

    int size();

    float *get();

    float &get(int k);

    float &operator[](int k);
};

#endif // JNIFLOATARRAY_H
