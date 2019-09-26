#ifndef QUANTIZATION_H
    #define QUANTIZATION_H

    #include "inferxlite_common.h"

    #ifndef MAX_STRING_LENGTH
        #define MAX_STRING_LENGTH 1000
    #endif // MAX_STRING_LENGTH

    #ifndef NAME_STRING_LENGTH
        #define NAME_STRING_LENGTH 100
    #endif // NAME_STRING_LENGTH

    #ifndef OUT
        #define OUT
    #endif // OUT

    #ifndef IN
        #define IN
    #endif // IN

    #ifndef uchar
        #define uchar unsigned char
    #endif // uchar

    typedef enum tagESignBitMark{
        sbm_HAVE_SIGN_BIT = 0,
        sbm_NONE_SIGN_BIT = 1
    }ESignBitMark;

    typedef enum tagEBitsPerFloat{
        bpf_8BITS_PER_FLOAT  = 0,
        bpf_16BITS_PER_FLOAT = 1
    }EBitsPerFloat;

    typedef enum tagEOptimizeType{
        ot_NONE_WEIGHT = 0,
        ot_HAVE_WEIGHT = 1,
        ot_Max_Value   = 2
    }EOptimizeType;

    typedef enum tagERoundingMethod{
        rm_ROUND = 0,
        rm_FLOOR = 1,
        rm_FIX   = 2
    }ERoundingMethod;

    typedef struct tagQuantScheme{
        ESignBitMark    m_eSignBitMark;
        EBitsPerFloat   m_eBitsPerFloat;
        EOptimizeType   m_eOptimizeType;
        ERoundingMethod m_eRoundingMethod;
                    int m_iExponentBitBias;
    }QuantScheme;

    typedef struct tagInferxQuantLayers{
        /*max number pipeline, the number upper limit of modelflow and dataflow */
        unsigned int   m_iMaxPipelineNumber;
        unsigned int   m_iLayerTotal;
                char** m_ppcTop;
                char** m_ppcBottom;
                char** m_ppcModel;
    }InferxQuantLayers;

    // 目标是对 inferx_data_tensor 结构体做扩展
    // 实际上是对 inferx_data_pipeline 和 inferx_model_pipeline 做了内存扩展
    typedef struct tagInferxDataTensorQuantExt{
        QuantScheme m_QuantScheme;
    }InferxDataTensorQuantExt;

    // inferx_context_t 对此结构体做扩展
    typedef struct tagInferxContextQuantExt{
        /* quantlize data_pipeline arrary */
        struct inferx_data_pipeline*  m_pQuantDataFlow;
        /* quantlize model_pipeline arrary */
        struct inferx_model_pipeline* m_pQuantModelFlow;

        /* layers name of prototxt */
        InferxQuantLayers             m_QuantLayers;

        /* inferxlite quantization scheme*/
        QuantScheme                   m_QuantScheme;
    }InferxContextQuantExt;

    #ifdef __cplusplus
    extern "C"
    {
    #endif // __cplusplus

    void
    inferx_create_quant_context(IN OUT inferx_context* ppInferxContext);

    void
    inferx_destroy_quant_context(IN OUT inferx_context* ppInferxContext);

    void
    inferx_quant_variance_analysis(IN const char* pcVariAnalyFile);

    void
    inferx_use_quant_flow(IN struct inferx_handler* pstInferxHandler);

    void
    inferx_disuse_quant_flow(IN struct inferx_handler* pstInferxHandler);

    void
    inferx_set_current_image_index(IN const int  iCurrentImageIndex, 
                                   IN const bool bIsOutputBinInfo);

	void
	inferx_save_bin_info(IN struct inferx_handler* pstInferxHandler);

    void
    inferx_input_after( int* nchw_l,    void* pdata,     char* top_pre,
                       char* iname_pre, char* model_pre, char* data_pre,
                       struct inferx_handler* pstInferxHandler);

    void
    inferx_convolution_after(int   input_c,    int   output_c,
                             int   kernel_h,   int   kernel_w,
                             int   str_h,      int   str_w,
                             int   pad_h,      int   pad_w,
                             int   group,      int   dilation, int axis, bool bias, bool force_nd_im2col,
                             char* bottom_pre, char* top_pre,
                             char* iname_pre,  char* model_pre, char* data_pre,
                             int   activation_type, float negative_slope,
                             struct inferx_handler * pstInferxHandler);

    void
    inferx_pooling_after(int kernel_h, int kernel_w,
                         int str_h,    int str_w,
                         int pad_h,    int pad_w,
                         char* pool_mode,
                         char* bottom_pre, char* top_pre,
                         char* iname_pre,  char* model_pre, char* data_pre,
                         struct inferx_handler * pstInferxHandler);

    void
    inferx_concat_after(int    num_output,  int   axis,    int   concat_dim,  int  bottom_num,
                        char** bottoms_pre, char* top_pre, char* iname_pre,  char* model_pre, char* data_pre,
                        struct inferx_handler* pstInferxHandler);

    void
    inferx_elem_wise_operate_after(int   coeffs_num, float* coeffs,
                                   char* op_mode,
                                   bool  stabel_prod_grad,
                                   int   bottom_num, char** bottoms_pre,
                                   char* top_pre, char* iname_pre, char* model_pre, char* data_pre,
                                   struct inferx_handler* pstInferxHandler);

    void
    inferx_permute_after( int  blob_dim1,   int  blob_dim2,  int  blob_dim3,  int  blob_dim4,  int  axis_num,
                         char* bottom_pre, char* top_pre,   char* iname_pre, char* model_pre, char* data_pre,
                         struct inferx_handler * pstInferxHandler);

    void
    inferx_flatten_after(char* bottom_pre, char* top_pre, char* iname_pre, char* model_pre, char* data_pre,
                         struct inferx_handler * pstInferxHandler);

    void
    inferx_reshape_after(int n, int c, int h, int w,
                         char* bottom_pre, char* top_pre,
                         char* iname_pre,  char* model_pre, char* data_pre,
                         struct inferx_handler * pstInferxHandler);

    #ifdef __cplusplus
    }
    #endif // __cplusplus

#endif //QUANTIZATION_H
