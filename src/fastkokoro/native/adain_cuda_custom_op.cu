#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

#include "onnxruntime_c_api.h"

#ifndef FASTKOKORO_ORT_API_VERSION
#define FASTKOKORO_ORT_API_VERSION ORT_API_VERSION
#endif

static const OrtApi* g_ort = NULL;

typedef struct {
  int unused;
} AdaInCudaKernel;

typedef struct {
  int unused;
} AdaInSnakeCudaKernel;

typedef struct {
  int unused;
} Atan2CudaKernel;

static __global__ void AdaInFp16Kernel(
    const half* x,
    const half* norm_weight,
    const half* norm_bias,
    const half* scale,
    const half* shift,
    half* y,
    int channels,
    int length) {
  const int c = blockIdx.x;
  if (c >= channels) return;

  extern __shared__ float scratch[];
  float* sum_scratch = scratch;
  float* sq_scratch = scratch + blockDim.x;

  const int offset = c * length;
  float sum = 0.0f;
  float sumsq = 0.0f;
  for (int pos = threadIdx.x; pos < length; pos += blockDim.x) {
    const float value = __half2float(x[offset + pos]);
    sum += value;
    sumsq += value * value;
  }
  sum_scratch[threadIdx.x] = sum;
  sq_scratch[threadIdx.x] = sumsq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sum_scratch[threadIdx.x] += sum_scratch[threadIdx.x + stride];
      sq_scratch[threadIdx.x] += sq_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float mean = sum_scratch[0] / (float)length;
  float variance = sq_scratch[0] / (float)length - mean * mean;
  if (variance < 0.0f) variance = 0.0f;
  const float inv_std = rsqrtf(variance + 1e-5f);
  const float norm_w = __half2float(norm_weight[c]);
  const float norm_b = __half2float(norm_bias[c]);
  const float scale_v = __half2float(scale[c]);
  const float shift_v = __half2float(shift[c]);

  for (int pos = threadIdx.x; pos < length; pos += blockDim.x) {
    const float value = __half2float(x[offset + pos]);
    const float normalized = (value - mean) * inv_std * norm_w + norm_b;
    y[offset + pos] = __float2half(normalized * scale_v + shift_v);
  }
}

static __global__ void AdaInSnakeFp16Kernel(
    const half* x,
    const half* norm_weight,
    const half* norm_bias,
    const half* scale,
    const half* shift,
    const half* alpha,
    half* y,
    int channels,
    int length) {
  const int c = blockIdx.x;
  if (c >= channels) return;

  extern __shared__ float scratch[];
  float* sum_scratch = scratch;
  float* sq_scratch = scratch + blockDim.x;

  const int offset = c * length;
  float sum = 0.0f;
  float sumsq = 0.0f;
  for (int pos = threadIdx.x; pos < length; pos += blockDim.x) {
    const float value = __half2float(x[offset + pos]);
    sum += value;
    sumsq += value * value;
  }
  sum_scratch[threadIdx.x] = sum;
  sq_scratch[threadIdx.x] = sumsq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sum_scratch[threadIdx.x] += sum_scratch[threadIdx.x + stride];
      sq_scratch[threadIdx.x] += sq_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float mean = sum_scratch[0] / (float)length;
  float variance = sq_scratch[0] / (float)length - mean * mean;
  if (variance < 0.0f) variance = 0.0f;
  const float inv_std = rsqrtf(variance + 1e-5f);
  const float norm_w = __half2float(norm_weight[c]);
  const float norm_b = __half2float(norm_bias[c]);
  const float scale_v = __half2float(scale[c]);
  const float shift_v = __half2float(shift[c]);
  const float alpha_v = __half2float(alpha[c]);
  const float inv_alpha = 1.0f / alpha_v;

  for (int pos = threadIdx.x; pos < length; pos += blockDim.x) {
    const float value = __half2float(x[offset + pos]);
    const float normalized = (value - mean) * inv_std * norm_w + norm_b;
    const float adain = normalized * scale_v + shift_v;
    const float sine = sinf(alpha_v * adain);
    y[offset + pos] = __float2half(adain + inv_alpha * sine * sine);
  }
}

static void* ORT_API_CALL CreateKernel(const OrtCustomOp* op, const OrtApi* api,
                                       const OrtKernelInfo* info) {
  (void)op;
  (void)info;
  g_ort = api;
  return calloc(1, sizeof(AdaInCudaKernel));
}

static const char* ORT_API_CALL GetName(const OrtCustomOp* op) {
  (void)op;
  return "AdaInCudaFp16";
}

static const char* ORT_API_CALL GetAdaInSnakeName(const OrtCustomOp* op) {
  (void)op;
  return "AdaInSnakeCudaFp16";
}

static const char* ORT_API_CALL GetExecutionProviderType(const OrtCustomOp* op) {
  (void)op;
  return "CUDAExecutionProvider";
}

static ONNXTensorElementDataType ORT_API_CALL GetInputType(const OrtCustomOp* op,
                                                          size_t index) {
  (void)op;
  (void)index;
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

static size_t ORT_API_CALL GetInputTypeCount(const OrtCustomOp* op) {
  (void)op;
  return 5;
}

static size_t ORT_API_CALL GetAdaInSnakeInputTypeCount(const OrtCustomOp* op) {
  (void)op;
  return 6;
}

static ONNXTensorElementDataType ORT_API_CALL GetOutputType(const OrtCustomOp* op,
                                                           size_t index) {
  (void)op;
  (void)index;
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

static size_t ORT_API_CALL GetOutputTypeCount(const OrtCustomOp* op) {
  (void)op;
  return 1;
}

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL GetInputCharacteristic(
    const OrtCustomOp* op, size_t index) {
  (void)op;
  (void)index;
  return INPUT_OUTPUT_REQUIRED;
}

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL GetOutputCharacteristic(
    const OrtCustomOp* op, size_t index) {
  (void)op;
  (void)index;
  return INPUT_OUTPUT_REQUIRED;
}

static OrtMemType ORT_API_CALL GetInputMemoryType(const OrtCustomOp* op, size_t index) {
  (void)op;
  (void)index;
  return OrtMemTypeDefault;
}

static void ORT_API_CALL KernelDestroy(void* op_kernel) {
  free(op_kernel);
}

static void ORT_API_CALL KernelCompute(void* op_kernel, OrtKernelContext* context) {
  (void)op_kernel;
  const OrtValue *input = NULL, *norm_weight = NULL, *norm_bias = NULL;
  const OrtValue *scale = NULL, *shift = NULL;
  OrtTensorTypeAndShapeInfo* input_info = NULL;
  int64_t dims64[3] = {0, 0, 0};
  const half *x = NULL, *norm_w = NULL, *norm_b = NULL, *s = NULL, *t = NULL;
  half* y = NULL;
  OrtValue* output = NULL;
  void* stream = NULL;
  int channels = 0;
  int length = 0;
  const int threads = 256;
  size_t shared_bytes = 0;

  if (g_ort->KernelContext_GetInput(context, 0, &input) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 1, &norm_weight) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 2, &norm_bias) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 3, &scale) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 4, &shift) != NULL) return;
  if (g_ort->GetTensorTypeAndShape(input, &input_info) != NULL) return;
  if (g_ort->GetDimensions(input_info, dims64, 3) != NULL) goto cleanup;
  if (dims64[0] != 1 || dims64[1] <= 0 || dims64[2] <= 0) goto cleanup;
  if (g_ort->KernelContext_GetOutput(context, 0, dims64, 3, &output) != NULL) {
    goto cleanup;
  }
  if (g_ort->GetTensorMutableData((OrtValue*)input, (void**)&x) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData((OrtValue*)norm_weight, (void**)&norm_w) != NULL) {
    goto cleanup;
  }
  if (g_ort->GetTensorMutableData((OrtValue*)norm_bias, (void**)&norm_b) != NULL) {
    goto cleanup;
  }
  if (g_ort->GetTensorMutableData((OrtValue*)scale, (void**)&s) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData((OrtValue*)shift, (void**)&t) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData(output, (void**)&y) != NULL) goto cleanup;

  channels = (int)dims64[1];
  length = (int)dims64[2];
  shared_bytes = (size_t)threads * 2 * sizeof(float);
  if (g_ort->KernelContext_GetGPUComputeStream(context, &stream) != NULL) {
    goto cleanup;
  }
  AdaInFp16Kernel<<<channels, threads, shared_bytes, (cudaStream_t)stream>>>(
      x, norm_w, norm_b, s, t, y, channels, length);

cleanup:
  if (input_info) g_ort->ReleaseTensorTypeAndShapeInfo(input_info);
}

static void ORT_API_CALL AdaInSnakeKernelCompute(void* op_kernel,
                                                OrtKernelContext* context) {
  (void)op_kernel;
  const OrtValue *input = NULL, *norm_weight = NULL, *norm_bias = NULL;
  const OrtValue *scale = NULL, *shift = NULL, *alpha = NULL;
  OrtTensorTypeAndShapeInfo* input_info = NULL;
  int64_t dims64[3] = {0, 0, 0};
  const half *x = NULL, *norm_w = NULL, *norm_b = NULL;
  const half *s = NULL, *t = NULL, *a = NULL;
  half* y = NULL;
  OrtValue* output = NULL;
  void* stream = NULL;
  int channels = 0;
  int length = 0;
  const int threads = 256;
  size_t shared_bytes = 0;

  if (g_ort->KernelContext_GetInput(context, 0, &input) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 1, &norm_weight) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 2, &norm_bias) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 3, &scale) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 4, &shift) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 5, &alpha) != NULL) return;
  if (g_ort->GetTensorTypeAndShape(input, &input_info) != NULL) return;
  if (g_ort->GetDimensions(input_info, dims64, 3) != NULL) goto cleanup;
  if (dims64[0] != 1 || dims64[1] <= 0 || dims64[2] <= 0) goto cleanup;
  if (g_ort->KernelContext_GetOutput(context, 0, dims64, 3, &output) != NULL) {
    goto cleanup;
  }
  if (g_ort->GetTensorMutableData((OrtValue*)input, (void**)&x) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData((OrtValue*)norm_weight, (void**)&norm_w) != NULL) {
    goto cleanup;
  }
  if (g_ort->GetTensorMutableData((OrtValue*)norm_bias, (void**)&norm_b) != NULL) {
    goto cleanup;
  }
  if (g_ort->GetTensorMutableData((OrtValue*)scale, (void**)&s) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData((OrtValue*)shift, (void**)&t) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData((OrtValue*)alpha, (void**)&a) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData(output, (void**)&y) != NULL) goto cleanup;

  channels = (int)dims64[1];
  length = (int)dims64[2];
  shared_bytes = (size_t)threads * 2 * sizeof(float);
  if (g_ort->KernelContext_GetGPUComputeStream(context, &stream) != NULL) {
    goto cleanup;
  }
  AdaInSnakeFp16Kernel<<<channels, threads, shared_bytes, (cudaStream_t)stream>>>(
      x, norm_w, norm_b, s, t, a, y, channels, length);

cleanup:
  if (input_info) g_ort->ReleaseTensorTypeAndShapeInfo(input_info);
}

static __global__ void Atan2Fp16Kernel(
    const half* imag,
    const half* real,
    half* phase,
    int64_t size) {
  const int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;
  const float y = __half2float(imag[index]);
  const float x = __half2float(real[index]);
  phase[index] = __float2half(atan2f(y, x));
}

static void* ORT_API_CALL CreateAtan2Kernel(const OrtCustomOp* op, const OrtApi* api,
                                            const OrtKernelInfo* info) {
  (void)op;
  (void)info;
  g_ort = api;
  return calloc(1, sizeof(Atan2CudaKernel));
}

static const char* ORT_API_CALL GetAtan2Name(const OrtCustomOp* op) {
  (void)op;
  return "Atan2CudaFp16";
}

static size_t ORT_API_CALL GetAtan2InputTypeCount(const OrtCustomOp* op) {
  (void)op;
  return 2;
}

static void ORT_API_CALL Atan2KernelCompute(void* op_kernel,
                                            OrtKernelContext* context) {
  (void)op_kernel;
  const OrtValue *imag_value = NULL, *real_value = NULL;
  OrtTensorTypeAndShapeInfo* input_info = NULL;
  size_t dim_count = 0;
  int64_t* dims = NULL;
  int64_t size = 1;
  const half *imag = NULL, *real = NULL;
  half* phase = NULL;
  OrtValue* output = NULL;
  void* stream = NULL;
  const int threads = 256;

  if (g_ort->KernelContext_GetInput(context, 0, &imag_value) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 1, &real_value) != NULL) return;
  if (g_ort->GetTensorTypeAndShape(imag_value, &input_info) != NULL) return;
  if (g_ort->GetDimensionsCount(input_info, &dim_count) != NULL) goto cleanup;
  dims = (int64_t*)calloc(dim_count, sizeof(int64_t));
  if (dims == NULL) goto cleanup;
  if (g_ort->GetDimensions(input_info, dims, dim_count) != NULL) goto cleanup;
  for (size_t index = 0; index < dim_count; ++index) {
    if (dims[index] <= 0) goto cleanup;
    size *= dims[index];
  }
  if (g_ort->KernelContext_GetOutput(context, 0, dims, dim_count, &output) != NULL) {
    goto cleanup;
  }
  if (g_ort->GetTensorMutableData((OrtValue*)imag_value, (void**)&imag) != NULL) {
    goto cleanup;
  }
  if (g_ort->GetTensorMutableData((OrtValue*)real_value, (void**)&real) != NULL) {
    goto cleanup;
  }
  if (g_ort->GetTensorMutableData(output, (void**)&phase) != NULL) goto cleanup;
  if (g_ort->KernelContext_GetGPUComputeStream(context, &stream) != NULL) {
    goto cleanup;
  }

  Atan2Fp16Kernel<<<
      (int)((size + threads - 1) / threads), threads, 0, (cudaStream_t)stream>>>(
      imag, real, phase, size);

cleanup:
  free(dims);
  if (input_info) g_ort->ReleaseTensorTypeAndShapeInfo(input_info);
}

static OrtCustomOp c_AdaInCudaOp = {
    FASTKOKORO_ORT_API_VERSION,
    CreateKernel,
    GetName,
    GetExecutionProviderType,
    GetInputType,
    GetInputTypeCount,
    GetOutputType,
    GetOutputTypeCount,
    KernelCompute,
    KernelDestroy,
    GetInputCharacteristic,
    GetOutputCharacteristic,
    GetInputMemoryType,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
};

static OrtCustomOp c_AdaInSnakeCudaOp = {
    FASTKOKORO_ORT_API_VERSION,
    CreateKernel,
    GetAdaInSnakeName,
    GetExecutionProviderType,
    GetInputType,
    GetAdaInSnakeInputTypeCount,
    GetOutputType,
    GetOutputTypeCount,
    AdaInSnakeKernelCompute,
    KernelDestroy,
    GetInputCharacteristic,
    GetOutputCharacteristic,
    GetInputMemoryType,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
};

static OrtCustomOp c_Atan2CudaOp = {
    FASTKOKORO_ORT_API_VERSION,
    CreateAtan2Kernel,
    GetAtan2Name,
    GetExecutionProviderType,
    GetInputType,
    GetAtan2InputTypeCount,
    GetOutputType,
    GetOutputTypeCount,
    Atan2KernelCompute,
    KernelDestroy,
    GetInputCharacteristic,
    GetOutputCharacteristic,
    GetInputMemoryType,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
};

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options,
    const OrtApiBase* api_base) {
  g_ort = api_base->GetApi(FASTKOKORO_ORT_API_VERSION);
  OrtCustomOpDomain* domain = NULL;
  OrtStatus* status = g_ort->CreateCustomOpDomain("fastkokoro", &domain);
  if (status) return status;
  status = g_ort->CustomOpDomain_Add(domain, &c_AdaInCudaOp);
  if (status) return status;
  status = g_ort->CustomOpDomain_Add(domain, &c_AdaInSnakeCudaOp);
  if (status) return status;
  status = g_ort->CustomOpDomain_Add(domain, &c_Atan2CudaOp);
  if (status) return status;
  return g_ort->AddCustomOpDomain(options, domain);
}
