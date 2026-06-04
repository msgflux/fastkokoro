#include <math.h>
#include <stdlib.h>

#include "onnxruntime_c_api.h"

#if defined(__AVX__)
#include <immintrin.h>
#endif

static const OrtApi* g_ort = NULL;

typedef struct {
  int unused;
} AdaInKernel;

static void* ORT_API_CALL CreateKernel(const OrtCustomOp* op, const OrtApi* api,
                                       const OrtKernelInfo* info) {
  (void)op;
  (void)info;
  g_ort = api;
  AdaInKernel* kernel = (AdaInKernel*)calloc(1, sizeof(AdaInKernel));
  return kernel;
}

static const char* ORT_API_CALL GetName(const OrtCustomOp* op) {
  (void)op;
  return "AdaIn";
}

static const char* ORT_API_CALL GetExecutionProviderType(const OrtCustomOp* op) {
  (void)op;
  return NULL;
}

static ONNXTensorElementDataType ORT_API_CALL GetInputType(const OrtCustomOp* op,
                                                          size_t index) {
  (void)op;
  (void)index;
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

static size_t ORT_API_CALL GetInputTypeCount(const OrtCustomOp* op) {
  (void)op;
  return 3;
}

static ONNXTensorElementDataType ORT_API_CALL GetOutputType(const OrtCustomOp* op,
                                                           size_t index) {
  (void)op;
  (void)index;
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
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
  const OrtValue *input = NULL, *scale = NULL, *shift = NULL;
  OrtTensorTypeAndShapeInfo* input_info = NULL;
  int64_t dims[3] = {0, 0, 0};
  const float *x = NULL, *s = NULL, *t = NULL;
  float* y = NULL;
  OrtValue* output = NULL;

  if (g_ort->KernelContext_GetInput(context, 0, &input) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 1, &scale) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 2, &shift) != NULL) return;
  if (g_ort->GetTensorTypeAndShape(input, &input_info) != NULL) return;
  if (g_ort->GetDimensions(input_info, dims, 3) != NULL) goto cleanup;
  if (g_ort->KernelContext_GetOutput(context, 0, dims, 3, &output) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData((OrtValue*)input, (void**)&x) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData((OrtValue*)scale, (void**)&s) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData((OrtValue*)shift, (void**)&t) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData(output, (void**)&y) != NULL) goto cleanup;

  const int64_t n = dims[0];
  const int64_t channels = dims[1];
  const int64_t length = dims[2];
  if (n != 1 || channels <= 0 || length <= 0) goto cleanup;

  #pragma omp parallel for schedule(static)
  for (int64_t c = 0; c < channels; ++c) {
    const float* x_c = x + c * length;
    float* y_c = y + c * length;
    double sum = 0.0;
    double sumsq = 0.0;
#if defined(__AVX__)
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 sumsq_vec = _mm256_setzero_ps();
    int64_t pos = 0;
    for (; pos + 8 <= length; pos += 8) {
      const __m256 values = _mm256_loadu_ps(x_c + pos);
      sum_vec = _mm256_add_ps(sum_vec, values);
#if defined(__FMA__)
      sumsq_vec = _mm256_fmadd_ps(values, values, sumsq_vec);
#else
      sumsq_vec = _mm256_add_ps(sumsq_vec, _mm256_mul_ps(values, values));
#endif
    }
    float tmp_sum[8];
    float tmp_sumsq[8];
    _mm256_storeu_ps(tmp_sum, sum_vec);
    _mm256_storeu_ps(tmp_sumsq, sumsq_vec);
    for (int i = 0; i < 8; ++i) {
      sum += tmp_sum[i];
      sumsq += tmp_sumsq[i];
    }
    for (; pos < length; ++pos) {
      const float value = x_c[pos];
      sum += value;
      sumsq += (double)value * (double)value;
    }
#else
    for (int64_t pos = 0; pos < length; ++pos) {
      const float value = x_c[pos];
      sum += value;
      sumsq += (double)value * (double)value;
    }
#endif
    const double mean = sum / (double)length;
    double variance = sumsq / (double)length - mean * mean;
    if (variance < 0.0) variance = 0.0;
    const float inv_std = 1.0f / sqrtf((float)variance + 1e-5f);
    const float scale_v = s[c];
    const float shift_v = t[c];
#if defined(__AVX__)
    const __m256 mean_vec = _mm256_set1_ps((float)mean);
    const __m256 inv_std_vec = _mm256_set1_ps(inv_std);
    const __m256 scale_vec = _mm256_set1_ps(scale_v);
    const __m256 shift_vec = _mm256_set1_ps(shift_v);
    int64_t out_pos = 0;
    for (; out_pos + 8 <= length; out_pos += 8) {
      __m256 values = _mm256_loadu_ps(x_c + out_pos);
      values = _mm256_sub_ps(values, mean_vec);
      values = _mm256_mul_ps(values, inv_std_vec);
#if defined(__FMA__)
      values = _mm256_fmadd_ps(values, scale_vec, shift_vec);
#else
      values = _mm256_add_ps(_mm256_mul_ps(values, scale_vec), shift_vec);
#endif
      _mm256_storeu_ps(y_c + out_pos, values);
    }
    for (; out_pos < length; ++out_pos) {
      y_c[out_pos] = (x_c[out_pos] - (float)mean) * inv_std * scale_v + shift_v;
    }
#else
    for (int64_t pos = 0; pos < length; ++pos) {
      y_c[pos] = (x_c[pos] - (float)mean) * inv_std * scale_v + shift_v;
    }
#endif
  }

cleanup:
  if (input_info) g_ort->ReleaseTensorTypeAndShapeInfo(input_info);
}

static OrtCustomOp c_AdaInOp = {
    ORT_API_VERSION,
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

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  g_ort = api_base->GetApi(ORT_API_VERSION);
  OrtCustomOpDomain* domain = NULL;
  OrtStatus* status = g_ort->CreateCustomOpDomain("fastkokoro", &domain);
  if (status) return status;
  status = g_ort->CustomOpDomain_Add(domain, &c_AdaInOp);
  if (status) return status;
  return g_ort->AddCustomOpDomain(options, domain);
}
