#ifndef MAT_H__
#define MAT_H__

#define PERFDNN_8U   0
#define PERFDNN_8S   1
#define PERFDNN_16U  2
#define PERFDNN_16S  3
#define PERFDNN_32S  4
#define PERFDNN_32F  5
#define PERFDNN_64F  6
#define PERFDNN_16F  (7+2)

#define PERF_DNN_ALLOC_BYTE_ALIGNMENT 64
#define perf_dnn_assert(exp, description) \
{ \               
    if(exp) \      
        ; \         
    else \
    { \
        printf("PERF_DNN_ERROR: "); \
            printf(description);   \
            printf("\n");   \
            exit(1);  \
    } \
} 
#define PERF_DNN_BYTE_ALIGNMENT(address, alignment) \
{ \
    (address) = (((address) + ((alignment) - 1)) & (-alignment)); \
}

#define PERFDNN_CN_SHIFT              3
#define PERFDNN_DEPTH_MAX             (1 << PERFDNN_CN_SHIFT)
#define PERFDNN_MAT_DEPTH_MASK       (PERFDNN_DEPTH_MAX - 1)
#define PERFDNN_MAT_DEPTH(flags)     ((flags % 7) & PERFDNN_MAT_DEPTH_MASK)
// Size of each channel item,
// 0x124489 = 1000 0100 0100 0010 0010 0001 0001 ~ array of sizeof(arr_type_elem) */
#define PERFDNN_ELEM_SIZE(type) \
    ((((sizeof(size_t)<<28)|0x8442211) >> PERFDNN_MAT_DEPTH(type)*4) & 15)


/////////////////////////////////////perfDNN Mat//////////////////////////////////////////////////////////
/**
 * @brief   User-callable function to create a PerfDNN mat.
 * @param[in]  rows                the rows of mat
 * @param[in]  cols                the cols of mat
 * @param[in]  step                the number of elements that one row contains 
 * @param[in]  channels            the number of channels 
 * @param[in]  batch_size          the number of images that processed simultaneously 
 * @param[in]  depth               the data type of mat
 * @retva     perfdnn_mat         PerfDNN mat 
 */
perfdnn_mat_t perf_dnn_init_mat(const size_t rows, 
        const size_t cols, 
        const size_t step, 
        const size_t channels,
        const size_t batch_size,
        const size_t depth);


/**
 * @brief   User-callable function to create a PerfDNN mat.
 * @param[in]  data                the data of mat
 * @param[in]  rows                the rows of mat
 * @param[in]  cols                the cols of mat
 * @param[in]  step                the number of elements that one row contains 
 * @param[in]  channels            the number of channels 
 * @param[in]  batch_size          the number of images that processed simultaneously 
 * @param[in]  depth               the data type of mat which is defined in  type.h
 * @retva     perfdnn_mat         PerfDNN mat 
 */
perfdnn_mat_t perf_dnn_init_mat_with_data( void* data,
        const size_t rows, 
        const size_t cols, 
        const size_t step, 
        const size_t channels,
        const size_t batch_size,
        const size_t depth);


/**
 * @brief   User-callable function to create a PerfDNN mat.
 * @param[in]  src_dst_mat         the mat to be reshaped
 * @param[in]  rows                the rows of mat
 * @param[in]  cols                the cols of mat
 * @param[in]  channels            the number of channels 
 */
void  perf_dnn_reshape_mat(perfdnn_mat_t src_dst_mat,
        const int rows,
        const int cols,
        const int channels);

/**
 * @brief   User-callable function to free a PerfDNN mat.
 * @param[in]   mat             the mat will be freed 
 */
void perf_dnn_free_mat(const perfdnn_mat_t mat);
#endif
