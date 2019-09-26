#pragma once

#include <perfdnn.h>
#include <perfdnn/hwinfo.h>
#define min(a,b) ((a)<(b)?(a):(b))

static inline enum perfdnn_status validate_convolution_arguments(
	size_t batch_size, size_t input_channels, size_t output_channels,
	struct perfdnn_size input_size, struct perfdnn_padding input_padding,
	struct perfdnn_size kernel_size, struct perfdnn_size output_subsampling,struct perfdnn_size dilation,size_t group,enum perfdnn_activation activation)
{
	if (!perfdnn_hwinfo.initialized) {
		return perfdnn_status_uninitialized;
	}

	if (!perfdnn_hwinfo.supported) {
		return perfdnn_status_unsupported_hardware;
	}

	if (batch_size == 0) {
		return perfdnn_status_invalid_batch_size;
	}

	if (input_channels == 0) {
		return perfdnn_status_invalid_input_channels;
	}

	if (output_channels == 0) {
		return perfdnn_status_invalid_output_channels;
	}

	if (min(input_size.height, input_size.width) == 0) {
		return perfdnn_status_invalid_input_size;
	}


	if (min(kernel_size.height, kernel_size.width) == 0) {
		return perfdnn_status_invalid_kernel_size;
	}

	if (min(output_subsampling.height, output_subsampling.width) == 0) {
		return perfdnn_status_invalid_output_subsampling;
	}
	if (min(dilation.height, dilation.width) == 0) {
		return perfdnn_status_invalid_dilation;
	}
	if (group == 0) {
		return perfdnn_status_invalid_group;
  }
  if(activation != perfdnn_activation_identity && 
          activation != perfdnn_activation_relu && 
          activation != perfdnn_activation_tanh && 
          activation != perfdnn_activation_pow &&
          activation != perfdnn_activation_sigmoid &&
          activation != perfdnn_activation_softmax &&
          activation != perfdnn_activation_logsoftmax)
  {
      return perfdnn_status_invalid_activation;
  }


	return perfdnn_status_success;
}
static inline enum perfdnn_status validate_pooling_arguments(
	size_t batch_size, size_t input_channels, 
	struct perfdnn_size input_size, struct perfdnn_padding input_padding,
	struct perfdnn_size pooling_size, struct perfdnn_size pooling_stride, enum perfdnn_pooling pooling_type)
{
	if (!perfdnn_hwinfo.initialized) {
		return perfdnn_status_uninitialized;
	}

	if (!perfdnn_hwinfo.supported) {
		return perfdnn_status_unsupported_hardware;
	}

	if (batch_size == 0) {
		return perfdnn_status_invalid_batch_size;
	}

	if (input_channels == 0) {
		return perfdnn_status_invalid_input_channels;
	}

	if (min(input_size.height, input_size.width) == 0) {
		return perfdnn_status_invalid_input_size;
	}

	if (min(pooling_size.height, pooling_size.width) == 0) {
		return perfdnn_status_invalid_pooling_size;
	}

	if (min(pooling_stride.height, pooling_stride.width) == 0) {
		return perfdnn_status_invalid_pooling_stride;
	}
  if(pooling_type != perfdnn_pooling_average && pooling_type != perfdnn_pooling_max && pooling_type != perfdnn_pooling_stochastic )
  {
      return perfdnn_status_invalid_pooling_type;
  }


	return perfdnn_status_success;
}

static inline enum perfdnn_status validate_deconvolution_arguments(
	size_t batch_size, size_t input_channels, size_t output_channels,
	struct perfdnn_size input_size, struct perfdnn_padding input_padding,
	struct perfdnn_size kernel_size, struct perfdnn_size output_subsampling,struct perfdnn_size dilation,size_t group,enum perfdnn_activation activation)
{
	if (!perfdnn_hwinfo.initialized) {
		return perfdnn_status_uninitialized;
	}

	if (!perfdnn_hwinfo.supported) {
		return perfdnn_status_unsupported_hardware;
	}

	if (batch_size == 0) {
		return perfdnn_status_invalid_batch_size;
	}

	if (input_channels == 0) {
		return perfdnn_status_invalid_input_channels;
	}

	if (output_channels == 0) {
		return perfdnn_status_invalid_output_channels;
	}

	if (min(input_size.height, input_size.width) == 0) {
		return perfdnn_status_invalid_input_size;
	}


	if (min(kernel_size.height, kernel_size.width) == 0) {
		return perfdnn_status_invalid_kernel_size;
	}

	if (min(output_subsampling.height, output_subsampling.width) == 0) {
		return perfdnn_status_invalid_output_subsampling;
	}
	if (min(dilation.height, dilation.width) == 0) {
		return perfdnn_status_invalid_dilation;
	}
	if (group == 0) {
		return perfdnn_status_invalid_group;
  }
  if(activation != perfdnn_activation_identity && 
          activation != perfdnn_activation_relu && 
          activation != perfdnn_activation_tanh && 
          activation != perfdnn_activation_pow &&
          activation != perfdnn_activation_sigmoid &&
          activation != perfdnn_activation_softmax &&
          activation != perfdnn_activation_logsoftmax)
  {
      return perfdnn_status_invalid_activation;
  }


	return perfdnn_status_success;
}
static inline enum perfdnn_status validate_tiling_arguments(
   size_t output_channels,
   size_t tiling_dim,
   struct perfdnn_size input_size)

{

	if (output_channels == 0) {
		return perfdnn_status_invalid_output_channels;
	}
	if (tiling_dim == 0) {
		return perfdnn_status_invalid_tiling_dim;
	}

	if (min(input_size.height, input_size.width) == 0) {
		return perfdnn_status_invalid_input_size;
	}

	return perfdnn_status_success;
}
