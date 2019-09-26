#include <jni.h>
#include <string>
#include "FaceManager.h"
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include "JniBitmap.h"
#include "JniFloatArray.h"
#include <android/log.h>

#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, "perfxlab_native", __VA_ARGS__);

using namespace std;

#ifdef __ANDROID__
jobject getGlobalContext(JNIEnv *env)
{
    //获取Activity Thread的实例对象
    jclass activityThread = env->FindClass("android/app/ActivityThread");
    jmethodID currentActivityThread = env->GetStaticMethodID(activityThread, "currentActivityThread", "()Landroid/app/ActivityThread;");
    jobject at = env->CallStaticObjectMethod(activityThread, currentActivityThread);
    //获取Application，也就是全局的Context
    jmethodID getApplication = env->GetMethodID(activityThread, "getApplication", "()Landroid/app/Application;");
    jobject context = env->CallObjectMethod(at, getApplication);
    return context;
}

char *get_android_id(JNIEnv *env)
{
    char *szDevId = NULL;
    jobject mContext = getGlobalContext(env);

    if (!mContext) {
        return szDevId;
    }

    jclass resCls = env->FindClass("android/content/Context");
    jmethodID getMethod = env->GetMethodID(resCls, "getContentResolver", "()Landroid/content/ContentResolver;");
    jobject resolver = env->CallObjectMethod(mContext, getMethod);
    if (resolver == NULL) {
        LOGE("Invalid resolver!");
    }

    jclass cls_context = env->FindClass("android/provider/Settings$Secure");
    if (cls_context == NULL) {
        LOGE("Invalid cls_context!");
    }

    jmethodID getStringMethod = env->GetStaticMethodID(cls_context, "getString", "(Landroid/content/ContentResolver;Ljava/lang/String;)Ljava/lang/String;");
    if (getStringMethod == NULL) {
        LOGE("Invalid getStringMethod!");
    }

    jfieldID ANDROID_ID = env->GetStaticFieldID(cls_context, "ANDROID_ID", "Ljava/lang/String;");
    jstring str = (jstring)(env->GetStaticObjectField(cls_context, ANDROID_ID));

    jstring jId = (jstring)(env->CallStaticObjectMethod(cls_context, getStringMethod, resolver, str));
    szDevId = (char *)(env->GetStringUTFChars(jId, 0));

    return szDevId;
}
#endif

void processMat(cv::Mat &mat, cv::Mat &resMat, int rotation) {
    int w = mat.cols;
    int h = mat.rows;
    switch (rotation) {
        case 90:
            cv::transpose(mat, resMat);
            cv::flip(resMat, resMat, 0);
            break;
        case 180:
            cv::flip(mat, resMat, -1);
            break;
        case 270:
            cv::transpose(mat, resMat);
            cv::flip(resMat, resMat, 1);
            break;
        default:
            resMat = mat;
            break;
    }
}

static void getRoatePointsXy(vector<vector<float>> &points, int pointRoate, int width, int height) {

    vector<vector<float>> pointsArr = points;
    for (int j = 0; j < points.size(); ++j) {
        vector<float> vectorSrc = pointsArr[j];
        switch (pointRoate) {
            case 180:
                for (int i = 0; i < vectorSrc.size() / 2; i++) {
                    points[j][i * 2 + 0] = width - vectorSrc[i * 2 + 0];
                    points[j][i * 2 + 1] = height - vectorSrc[i * 2 + 1];
                }
                break;
            case 0:
                for (int i = 0; i < vectorSrc.size() / 2; i++) {
                    points[j][i * 2 + 0] = vectorSrc[i * 2 + 0];
                    points[j][i * 2 + 1] = vectorSrc[i * 2 + 1];
                }
                break;
            case 270:
                for (int i = 0; i < vectorSrc.size() / 2; i++) {
                    points[j][i * 2 + 0] = vectorSrc[i * 2 + 1];
                    points[j][i * 2 + 1] = height - vectorSrc[i * 2 + 0];
                }
                break;
            case 90:
                for (int i = 0; i < vectorSrc.size() / 2; i++) {
                    points[j][i * 2 + 0] = width - vectorSrc[i * 2 + 1];
                    points[j][i * 2 + 1] = vectorSrc[i * 2 + 0];
                }
                break;
        }
    }
}

static void points2rect(vector<vector<float>> &rects) {
    for (int j = 0; j < rects.size(); ++j) {
        int maxValue = 60000;
        float left = maxValue, right = -maxValue
        , top = maxValue
        , bottom = -maxValue;
        for (int i = 0; i < rects[j].size() / 2; i++) {
            float x = rects[j][i * 2];
            float y = rects[j][i * 2 + 1];
            left = min(left, x);
            top = min(top, y);
            right = max(right, x);
            bottom = max(bottom, y);
        }
        rects[j][0] = left;
        rects[j][1] = top;
        rects[j][2] = right;
        rects[j][3] = bottom;
    }
}

static void getRoateRect(vector<vector<float>> &rects, int pointRoate, int width, int height) {

    //先旋转点
    getRoatePointsXy(rects, pointRoate, width, height);
    //计算rect
    points2rect(rects);
}

cv::Mat getBgrMatCV(jbyte *nv21, int width, int height) {
    cv::Mat matNv21(height * 1.5, width, CV_8UC1, nv21);
    cv::Mat bgrMat;
    cv::cvtColor(matNv21, bgrMat, CV_YUV2BGR_NV21);
    return bgrMat;
}

jobjectArray features2FloatArray2D(JNIEnv *env, const vector<vector<float>> &features) {
    jobjectArray array2D = env->NewObjectArray(features.size(), env->FindClass("[F"), NULL);;

    // features -> featureArray
    for(int i = 0; i < features.size(); i++) {
        const vector<float>& feature_tmp = features[i];
        jfloatArray array1D = env->NewFloatArray(feature_tmp.size());
        if (array1D != NULL) {
            env->SetFloatArrayRegion(array1D, 0, feature_tmp.size(), &feature_tmp[0]);
            env->SetObjectArrayElement(array2D, i, array1D);
            env->DeleteLocalRef(array1D);
        }
    }

    return array2D;
}

jobjectArray handleResult(JNIEnv *env, const vector<vector<float>> &rects, const vector<vector<float>> &features, const vector<int> gender_results) {
    jobjectArray retInfos = NULL;
    if (rects.size() == 0) {
        return retInfos;
    }
    const char *pClassNameFaceInfo = "com/perfxlab/faceregnition/sdkfacerecognition/FaceInfo";
    jclass jclsFaceInfo = env->FindClass(pClassNameFaceInfo);
    jclass jclsRectF = env->FindClass("android/graphics/RectF");
    jfieldID jfleft = env->GetFieldID(jclsRectF, "left", "F");
    jfieldID jfright = env->GetFieldID(jclsRectF, "right", "F");
    jfieldID jftop = env->GetFieldID(jclsRectF, "top", "F");
    jfieldID jfbottom = env->GetFieldID(jclsRectF, "bottom", "F");
    jfieldID jfrect = env->GetFieldID(jclsFaceInfo, "rect", "Landroid/graphics/RectF;");
    jfieldID jffeatures = env->GetFieldID(jclsFaceInfo, "feature", "[F");
    jfieldID jfgender = env->GetFieldID(jclsFaceInfo, "gender", "I");

    retInfos = env->NewObjectArray(rects.size(), jclsFaceInfo, NULL);
    for (int i = 0; i < rects.size(); i++) {

        jobject jobjInfo = env->AllocObject(jclsFaceInfo);

        jobject jobjRect = env->AllocObject(jclsRectF);
        env->SetFloatField(jobjRect, jfleft, rects[i][0]);
        env->SetFloatField(jobjRect, jftop, rects[i][1]);
        env->SetFloatField(jobjRect, jfright, rects[i][2]);
        env->SetFloatField(jobjRect, jfbottom, rects[i][3]);
        env->SetObjectField(jobjInfo, jfrect, jobjRect);

        if(i < features.size()) {
            jfloatArray jffeaturesarr = env->NewFloatArray(features[i].size());
            env->SetFloatArrayRegion(jffeaturesarr, 0, features[i].size(), &features[i][0]);
            env->SetObjectField(jobjInfo, jffeatures, jffeaturesarr);
            env->DeleteLocalRef(jffeaturesarr);
        }

        if(i < gender_results.size()) {
            env->SetIntField(jobjInfo, jfgender, gender_results[i]);
        }

        env->SetObjectArrayElement(retInfos, i, jobjInfo);
    }

    return retInfos;
}

void floatArray2VectorFloat(JNIEnv *env, jfloatArray floatArray, vector<float>& vectorFloat) {
//    jfloat *floats = env->GetFloatArrayElements(floatArray, NULL);
//    int len = env->GetArrayLength(floatArray);
    JniFloatArray floats(env, floatArray);
    int len = floats.size();
    for(int i = 0; i < len; i++) {
        vectorFloat.push_back(floats[i]);
    }
}

extern "C"
JNIEXPORT jlong
JNICALL
Java_com_perfxlab_faceregnition_sdkfacerecognition_FaceRecognition_nativeInit(
        JNIEnv *env, jobject /* this */, jstring modelDir) {
    const char* szModelDir = env->GetStringUTFChars(modelDir, 0);

#ifdef __ANDROID__
    const char* androidId = get_android_id(env);
    LOGE("androidId=%s", androidId);
#endif

    LOGE("nativeInit 0");
    FaceManager* pFaceManager = new FaceManager(szModelDir);
    LOGE("nativeInit 1");

    env->ReleaseStringUTFChars(modelDir, szModelDir);

    return (long) pFaceManager;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_perfxlab_faceregnition_sdkfacerecognition_FaceRecognition_nativeRelease(JNIEnv *env,
                                                                                 jobject instance,
                                                                                 jlong handle) {

    FaceManager* pFaceManager = (FaceManager*)handle;
    delete pFaceManager;
    pFaceManager = nullptr;
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_perfxlab_faceregnition_sdkfacerecognition_FaceRecognition_nativeGetFeature(JNIEnv *env,
                                                                                    jobject instance,
                                                                                    jlong handle,
                                                                                    jbyteArray data_,
                                                                                    jint width,
                                                                                    jint height,
                                                                                    jint imageRotate) {
    jbyte *data = env->GetByteArrayElements(data_, NULL);

    FaceManager* pFaceManager = (FaceManager*)handle;

    cv::Mat bgrMat = getBgrMatCV(data, width, height);

    cv::Mat input;
    processMat(bgrMat, input, imageRotate);

    vector<vector<float>> features;
    vector<vector<float>> face_rects;
    pFaceManager->get_feature(input, face_rects, features);

    env->ReleaseByteArrayElements(data_, data, 0);

    return handleResult(env, face_rects, features, vector<int>());
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_perfxlab_faceregnition_sdkfacerecognition_FaceRecognition_nativeGetFeatureBitmap(
        JNIEnv *env, jobject instance, jlong handle, jobject bitmap) {

    FaceManager* pFaceManager = (FaceManager*)handle;

    JniBitmap jniBitmapSrc(env, bitmap);
    // 转化为Mat
    cv::Mat input(jniBitmapSrc.getHeight(), jniBitmapSrc.getWidth(), CV_8UC4,
                   jniBitmapSrc.imagePixels);
    cv::cvtColor(input, input, cv::COLOR_RGBA2BGR);

    vector<vector<float>> features;
    vector<vector<float>> face_rects;
    pFaceManager->get_feature(input, face_rects, features);
    jobjectArray ret = handleResult(env, face_rects, features, vector<int>());

    return ret;
}

extern "C"
JNIEXPORT jfloat JNICALL
Java_com_perfxlab_faceregnition_sdkfacerecognition_FaceRecognition_nativeGetScore(JNIEnv *env,
                                                                                  jobject instance,
                                                                                  jlong handle,
                                                                                  jfloatArray feature1_,
                                                                                  jfloatArray feature2_) {
    FaceManager* pFaceManager = (FaceManager*)handle;
    vector<float> vecFeature1;
    vector<float> vecFeature2;
    floatArray2VectorFloat(env, feature1_, vecFeature1);
    floatArray2VectorFloat(env, feature2_, vecFeature2);
    float score = pFaceManager->get_score(vecFeature1, vecFeature2);
    return score;
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_perfxlab_faceregnition_sdkfacerecognition_FaceRecognition_nativeFaceAttr(JNIEnv *env,
                                                                                    jobject instance,
                                                                                    jlong handle,
                                                                                    jbyteArray data_,
                                                                                    jint width,
                                                                                    jint height,
                                                                                    jint imageRotate) {
    jbyte *data = env->GetByteArrayElements(data_, NULL);

    FaceManager* pFaceManager = (FaceManager*)handle;

    cv::Mat bgrMat = getBgrMatCV(data, width, height);

    cv::Mat input;
    processMat(bgrMat, input, imageRotate);

    vector<vector<float>> face_rects;
    std::vector<int> gender_results;
    pFaceManager->get_gender(input, face_rects, gender_results);

//    if(face_rects.size() > 0) {
//        cv::Mat saveMat = input.clone();
//        vector<float> face_rect = face_rects[0];
//        cv::rectangle(saveMat, cvRect(face_rect[0], face_rect[1], face_rect[2] - face_rect[0], face_rect[3] - face_rect[1]), cvScalar(0, 255, 0, 255));
//        cv::imwrite("/sdcard/test.jpg", saveMat);
//    }

    getRoateRect(face_rects, imageRotate, width, height);

    env->ReleaseByteArrayElements(data_, data, 0);

    return handleResult(env, face_rects, vector<vector<float>>(), gender_results);
}