#include "JniBitmap.h"

JniBitmap::JniBitmap(JNIEnv *env, jobject image) {
    this->env = env;
    this->image = image;
    ret = -1;
    imagePixels = NULL;
    if (NULL != image) {
        do {
            if ((ret = AndroidBitmap_getInfo(env, image, &imageInfo)) < 0) {
                break;
            }
            if (imageInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
                ret = -2;
                break;
            }
            if ((ret = AndroidBitmap_lockPixels(env, image, &imagePixels)) < 0) {
                break;
            }
            ret = 0;
        } while (false);
    }
}

JniBitmap::~JniBitmap() {
    if (0 == ret && image != NULL && imagePixels != NULL) {
        imagePixels = NULL;
        AndroidBitmap_unlockPixels(env, image);
        image = NULL;
        env = NULL;
    }
}