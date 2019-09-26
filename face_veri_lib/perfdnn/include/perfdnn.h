#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status code for any PERFDNN function call.
 */
enum perfdnn_status {
	/** The call succeeded, and all output arguments now contain valid data. */
	perfdnn_status_success = 0,
	/** PERFDNN function was called with batch_size == 0. */
	perfdnn_status_invalid_batch_size = 1,
	/** PERFDNN function was called with channels == 0. */
	perfdnn_status_invalid_channels = 2,
	/** PERFDNN function was called with input_channels == 0. */
	perfdnn_status_invalid_input_channels = 3,
	/** PERFDNN function was called with output_channels == 0. */
	perfdnn_status_invalid_output_channels = 4,
	/** PERFDNN function was called with input_size.height == 0 or input_size.width == 0 */
	perfdnn_status_invalid_input_size = 5,
	/** PERFDNN function was called with input_stride.height == 0 or input_stride.width == 0 */
	perfdnn_status_invalid_input_stride = 6,
	/** PERFDNN function was called with input_padding not less than respective kernel (or pooling) size, i.e.:
	 *
	 *  - input_padding.left   >= kernel_size.width  (>= pooling_size.width)
	 *  - input_padding.right  >= kernel_size.width  (>= pooling_size.width)
	 *  - input_padding.top    >= kernel_size.height (>= pooling_size.height)
	 *  - input_padding.bottom >= kernel_size.height (>= pooling_size.height)
	 */
	perfdnn_status_invalid_input_padding = 7,
	/** PERFDNN function was called with kernel_size.height == 0 or kernel_size.width == 0 */
	perfdnn_status_invalid_kernel_size = 8,
	/** PERFDNN function was called with pooling_size.height == 0 or pooling_size.width == 0 */
	perfdnn_status_invalid_pooling_size = 9,
	/** PERFDNN function was called with pooling_stride.height == 0 or pooling_stride.width == 0 */
	perfdnn_status_invalid_pooling_stride = 10,
	perfdnn_status_invalid_pooling_type = 11,
	/** PERFDNN function was called with convolution algorithm not in perfdnn_convolution_algorithm enumeration */
	perfdnn_status_invalid_algorithm = 12,
	/** PERFDNN function was called with convolution transform strategy not in perfdnn_convolution_transform_strategy enum */
	perfdnn_status_invalid_transform_strategy = 13,
	/** PERFDNN function was called with output_subsampling.height == 0 or output_subsampling.width == 0 */
	perfdnn_status_invalid_output_subsampling = 14,
	/** PERFDNN function was called with dilation.height == 0 or dilation.width == 0 */
	perfdnn_status_invalid_dilation = 15,
	/** PERFDNN function was called with group == 0. */
	perfdnn_status_invalid_group = 16,
	/** PERFDNN function was called with activation not in perfdnn_activation enum */
	perfdnn_status_invalid_activation = 17,
	/** PERFDNN function was called with invalid activation parameters */
	perfdnn_status_invalid_activation_parameters = 18,

	/** PERFDNN does not support the particular input size for the function */
	perfdnn_status_unsupported_input_size = 20,
	/** PERFDNN does not support the particular input stride for the function */
	perfdnn_status_unsupported_input_stride = 21,
	/** PERFDNN does not support the particular input padding for the function */
	perfdnn_status_unsupported_input_padding = 22,
	/** PERFDNN does not support the particular kernel size for the function */
	perfdnn_status_unsupported_kernel_size = 23,
	/** PERFDNN does not support the particular pooling size for the function */
	perfdnn_status_unsupported_pooling_size = 24,
	/** PERFDNN does not support the particular pooling stride for the function */
	perfdnn_status_unsupported_pooling_stride = 25,
	/** PERFDNN does not support the particular convolution algorithm for the function */
	perfdnn_status_unsupported_algorithm = 26,
	/** PERFDNN does not support the particular convolution transform strategy for the algorithm */
	perfdnn_status_unsupported_transform_strategy = 27,
	/** PERFDNN does not support the particular activation function for the function */
	perfdnn_status_unsupported_activation = 28,
	/** PERFDNN does not support the particular activation function parameters for the function */
	perfdnn_status_unsupported_activation_parameters = 29, 

	/** PERFDNN function was called before the library was initialized */
	perfdnn_status_uninitialized = 50,
	/** PERFDNN does not implement this function for the host CPU */
	perfdnn_status_unsupported_hardware = 51,
	/** PERFDNN failed to allocate memory for temporary buffers */
	perfdnn_status_out_of_memory = 52,
	/** Scratch space buffer is too small */
	perfdnn_status_insufficient_buffer = 53,
	/** Scratch space buffer is not properly aligned */
	perfdnn_status_misaligned_buffer = 54,
  perfdnn_status_invalid_tiling_dim = 55
};

/**
 * @brief Activation applied applied after a convolutional or fully-connected layer.
 */
enum perfdnn_activation {
	/** Identity activation f(x) := x, i.e. no transformation */
	perfdnn_activation_identity = 0,

	/** ReLU activation f(x) := max(0, x) */
	perfdnn_activation_relu = 1,

	/** TANH activation f(x) := (1 - e^-2x)/(1 + e^-2x) */
  perfdnn_activation_tanh=2,

	/** POW activation f(x) := exp(x) */
  perfdnn_activation_pow=3,

	/** SIGMOID activation f(x) := 1/(1+exp(-x)) */
  perfdnn_activation_sigmoid=4,

	/** SOFTMAX activation f(x) := exp(x)/sum(exp(j)) */
  perfdnn_activation_softmax=5,

	/** LOGSOFTMAX activation f(x) := log(1+exp(x)) */
  perfdnn_activation_logsoftmax=6
};
enum perfdnn_pooling {
	/** max pooling */
	perfdnn_pooling_max = 0,
	/** average pooling */
	perfdnn_pooling_average = 1,
	/** stochastic pooling */
	perfdnn_pooling_stochastic = 2

};

/**
 * @brief Algorithm for computing convolutional layers.
 */
enum perfdnn_convolution_algorithm {
	/** Let PERFDNN choose the algorithm depending on layer parameters */
	perfdnn_convolution_algorithm_auto = 0,
  /** Im2col gemm*/
	perfdnn_convolution_algorithm_im2col_gemm = 1,
	/** Tiled convolution based on 2D Winograd transform F(3x3, 6x6) with 8x8 blocks. Supports only 3x3 kernels. */
	perfdnn_convolution_algorithm_wt8x8 = 2,
	/** Direct convolution implementation.kernel_size >=3 */
	perfdnn_convolution_algorithm_direct = 3,
	/** interleave convolution implementation. dilation==1; kernel_size==3 padding<>2 */
	perfdnn_convolution_algorithm_interleave = 4,
	/** interleave convolution implementation. dilation==1 ; kernel_size==3 ;padding==1 ;stride==1 || stride==2*/
	perfdnn_convolution_algorithm_depthwise = 5,
};

enum perfdnn_convolution_transform_strategy {
	perfdnn_convolution_transform_strategy_compute = 1,
	perfdnn_convolution_transform_strategy_precompute = 2,
	perfdnn_convolution_transform_strategy_reuse = 3
};

/* For backward compatibility */
#define perfdnn_convolution_transform_strategy_block_based perfdnn_convolution_transform_strategy_compute
#define perfdnn_convolution_transform_strategy_tuple_based perfdnn_convolution_transform_strategy_compute

/**
 * @brief Size of images, kernels, and pooling filters in PERFDNN.
 */
struct perfdnn_size {
	/** Width (horizontal size) of an image, kernel, or pooling filter. */
	size_t width;
	/** Height (vertical size) of an image, kernel, or pooling filter. */
	size_t height;
};

/**
 * @brief Padding of images in PERFDNN.
 */
struct perfdnn_padding {
	/** Padding above the image data */
	size_t top;
	/** Padding on the right of image data */
	size_t right;
	/** Padding below the image data */
	size_t bottom;
	/** Padding on the left of image data */
	size_t left;
};

/**
 * @brief Profiling information about time spent in different phases of a function call.
 */
struct perfdnn_profile {
	/** Time spent inside the function call, in seconds. */
	double total;
	/** Time spend on transformation of the input or input gradient tensor, in seconds. */
	double input_transform;
	/** Time spend on transformation of the kernel or kernel gradient tensor, in seconds. */
	double kernel_transform;
	/** Time spend on transformation of the output or output gradient tensor, in seconds. */
	double output_transform;
	/** Time spend on multiplication-accumulation of transformed coefficients, in seconds. */
	double block_multiplication;
};

enum perfdnn_status perfdnn_initialize(void);

enum perfdnn_status perfdnn_deinitialize(void);
/**
 * @brief Computes output of a 2D convolutional layer for a single input image and a kernel tensor.
 * @details This function targets prediction with convolutional neural networks and performs forward propagation.
 * @param algorithm The type of algorithm to use for convolution. Possible values are:
 *
 *    - perfdnn_convolution_algorithm_auto    -- let the function choose the algorithm.
 *    - perfdnn_convolution_algorithm_im2col_gemm  --first use im2col then use gemm   
 *    - perfdnn_convolution_algorithm_wt8x8   -- tiled convolution based on 2D Winograd transform F(3x3, 6x6). Supports only 3x3 kernels.
 *    - perfdnn_convolution_algorithm_direct   -- direct convolution
 *
 * @param input_channels The number of channels (AKA features, dimensions) in the input image.
 * @param output_channels The number of channels (AKA features, dimensions) in the output image.
 * @param input_size Size of input image, excluding implicit zero-padding.
 * @param input_padding Implicit zero-padding of input image.
 * @param kernel_size Kernel size.
 * @param output_subsampling Subsample region for output, also known as convolution stride.
 * @param dilation Dilation is used to extern receptive fileld.
 * @param[in]  input  A 3D tensor input[input_channels][input_size.height][input_size.width].
 * @param[in]  kernel A 4D tensor kernel[output_channels][input_channels][kernel_size.height][kernel_size.width].
 * @param[in]  bias   A 1D array bias[output_channels].
 * @param[out] output A 3D tensor output[output_channels][output_size.height][output_size.width] where
 *                        output_size.height = (input_padding.top + input_size.height + input_padding.bottom) -
 *                                             (kernel_size.height - 1)
 *                        output_size.width  = (input_padding.left + input_size.width + input_padding.right) -
 *                                             (kernel_size.width - 1)
 * @param group  Group is used for group convolution.
 * @param bias_term Bias_term indicates whether bias is used.
 */

enum perfdnn_status perfdnn_convolution_inference(
	enum perfdnn_convolution_algorithm algorithm,
	size_t input_channels,
	size_t output_channels,
	struct perfdnn_size input_size,
	struct perfdnn_padding input_padding,
	struct perfdnn_size kernel_size,
	struct perfdnn_size output_subsampling,
  struct perfdnn_size dilation,
	const float* input,
	const float* kernel,
	float* bias,
	float* output,
  size_t group,
  bool bias_term,
  enum perfdnn_activation activation,
  void** workspace_buffer,
  size_t* workspace_size
  );

/**
 * @brief Computes output of a rectified linear unit (ReLU) layer for an input matrix.
 * @details This function targets both prediction and training of convolutional neural networks and performs forward
 *          propagation. Is is optimized for both large and small minibatch sizes.
 * @param channels   The number of channels (AKA features, dimensions) in both input and output matrices.
 * @param[in]  input  A 2D matrix input[channels][input_height*input_width].
 * @param[out] output A 2D matrix output[channels][input_height*input_width].
 */
enum perfdnn_status perfdnn_relu_inference(
        size_t elements,
        const float input[],
        float output[],
        float negative_slope
        );
enum perfdnn_status perfdnn_prelu_inference(
        size_t batchsize,
        size_t input_channels,
        size_t input_height,
        size_t input_width,
        const float input[],
        float output[],
        float slope[],
       bool channel_shared 
        );
/**
 * @brief Computes output of a rectified linear unit (Tanh) layer for an input matrix.
 * @details This function targets both prediction and training of convolutional neural networks and performs forward
 *          propagation. Is is optimized for both large and small minibatch sizes.
 * @param channels   The number of channels (AKA features, dimensions) in both input and output matrices.
 * @param[in]  input  A 2D matrix input[channels][input_height*input_width].
 * @param[out] output A 2D matrix output[channels][input_height*input_width].
 */
enum perfdnn_status perfdnn_tanh_inference(
        size_t elements,
        float input[],
        float output[]
        );
enum perfdnn_status perfdnn_pow_inference(
        size_t elements,
        float input[],
        float output[],
        float power,
        float scale,
        float shift
        );
enum perfdnn_status perfdnn_softmax_inference(
        size_t elements,
        float input[],
        float output[]
        );
enum perfdnn_status perfdnn_logsoftmax_inference(
        size_t elements,
        float input[],
        float output[]
        );
enum perfdnn_status perfdnn_sigmoid_inference(
        size_t elements,
        float input[],
        float output[]);

enum perfdnn_status perfdnn_pooling_inference(
        bool isglobal,
        size_t batch_size,
        size_t channels,
        struct perfdnn_size input_size,
        struct perfdnn_padding input_padding,
        struct perfdnn_size pooling_size,
        struct perfdnn_size pooling_stride,
        const float input[],
        float output[],
        enum perfdnn_pooling pooling_type
        );
/*
 im2col = (kernel ^T) * output
 im2col->image
 * */
enum perfdnn_status perfdnn_deconvolution_inference(
        size_t input_channels,
        size_t output_channels,
        struct perfdnn_size input_size,
        struct perfdnn_padding input_padding,
        struct perfdnn_size kernel_size,
        struct perfdnn_size stride,
        struct perfdnn_size dilation,
        const float* input,
        const float* kernel,
        float* output,
        size_t group,
        bool bias_term,
        float* bias,
        enum perfdnn_activation activation
        );

typedef struct {
    size_t rows;
    size_t cols;
    size_t step;
    size_t channels;
    size_t batch_size;
    size_t depth;
    size_t elem_size;
    void* data;
}perfdnn_mat;
typedef perfdnn_mat* perfdnn_mat_t;
void proposal_cpu(perfdnn_mat_t score_mat,
                  perfdnn_mat_t bbox_deltas_mat,
                  perfdnn_mat_t out_mat,
                  perfdnn_mat_t oscore_mat,
                  perfdnn_mat_t ratio_mat,
                  perfdnn_mat_t scale_mat,
                  float *        im_info,
                  float          threshold,
                  int            feature_stride,
                  int            rpn_pre_nms_top_n,
                  int            rpn_post_nms_top_n,
                  int            rpn_min_size);

enum perfdnn_status perfdnn_tiling_inference(
        size_t output_channels, 
        size_t tiling_dim, 
        struct perfdnn_size input_size,
        float scale,  
        const float input[],  
        float* output);

#ifdef __cplusplus
} /* extern "C" */
#endif


