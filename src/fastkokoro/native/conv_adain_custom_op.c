#include <math.h>
#include <stdlib.h>

#include "onnxruntime_c_api.h"

#if defined(__AVX__)
#include <immintrin.h>
#endif

static const OrtApi* g_ort = NULL;

typedef struct {
  int64_t pad_left;
  int64_t dilation;
} Conv1dAdaInKernel;

static OrtStatus* GetIntAttr(const OrtKernelInfo* info, const char* name, int64_t* value) {
  return g_ort->KernelInfoGetAttribute_int64(info, name, value);
}

static void* ORT_API_CALL CreateKernel(const OrtCustomOp* op, const OrtApi* api,
                                       const OrtKernelInfo* info) {
  (void)op;
  g_ort = api;
  Conv1dAdaInKernel* kernel = (Conv1dAdaInKernel*)calloc(1, sizeof(Conv1dAdaInKernel));
  if (!kernel) return NULL;
  kernel->pad_left = 0;
  kernel->dilation = 1;
  OrtStatus* status = GetIntAttr(info, "pad_left", &kernel->pad_left);
  if (status) g_ort->ReleaseStatus(status);
  status = GetIntAttr(info, "dilation", &kernel->dilation);
  if (status) g_ort->ReleaseStatus(status);
  if (kernel->dilation < 1) kernel->dilation = 1;
  return kernel;
}

static const char* ORT_API_CALL GetName(const OrtCustomOp* op) {
  (void)op;
  return "Conv1dAdaIn";
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
  return 5;
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

static int64_t ShapeDim(const int64_t* dims, size_t count, size_t index) {
  if (index >= count) return 0;
  return dims[index];
}

static void ORT_API_CALL KernelCompute(void* op_kernel, OrtKernelContext* context) {
  const Conv1dAdaInKernel* kernel = (const Conv1dAdaInKernel*)op_kernel;
  const OrtValue *input = NULL, *weight = NULL, *conv_bias = NULL, *scale = NULL, *shift = NULL;
  OrtTensorTypeAndShapeInfo *input_info = NULL, *weight_info = NULL;
  int64_t input_dims[3] = {0, 0, 0};
  int64_t weight_dims[3] = {0, 0, 0};
  const float *x = NULL, *w = NULL, *b = NULL, *s = NULL, *t = NULL;
  float* y = NULL;
  OrtValue* output = NULL;

  if (g_ort->KernelContext_GetInput(context, 0, &input) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 1, &weight) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 2, &conv_bias) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 3, &scale) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 4, &shift) != NULL) return;
  if (g_ort->GetTensorTypeAndShape(input, &input_info) != NULL) return;
  if (g_ort->GetTensorTypeAndShape(weight, &weight_info) != NULL) goto cleanup;
  if (g_ort->GetDimensions(input_info, input_dims, 3) != NULL) goto cleanup;
  if (g_ort->GetDimensions(weight_info, weight_dims, 3) != NULL) goto cleanup;

  const int64_t n = ShapeDim(input_dims, 3, 0);
  const int64_t cin = ShapeDim(input_dims, 3, 1);
  const int64_t lin = ShapeDim(input_dims, 3, 2);
  const int64_t cout = ShapeDim(weight_dims, 3, 0);
  const int64_t kw = ShapeDim(weight_dims, 3, 2);
  int64_t out_dims[3] = {n, cout, lin};
  if (n != 1 || cin <= 0 || cout <= 0 || lin <= 0 || kw <= 0) goto cleanup;

  if (g_ort->KernelContext_GetOutput(context, 0, out_dims, 3, &output) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData((OrtValue*)input, (void**)&x) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData((OrtValue*)weight, (void**)&w) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData((OrtValue*)conv_bias, (void**)&b) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData((OrtValue*)scale, (void**)&s) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData((OrtValue*)shift, (void**)&t) != NULL) goto cleanup;
  if (g_ort->GetTensorMutableData(output, (void**)&y) != NULL) goto cleanup;

  #pragma omp parallel for schedule(static)
  for (int64_t oc = 0; oc < cout; ++oc) {
    double sum = 0.0;
    double sumsq = 0.0;
    float* y_oc = y + oc * lin;
    const float* w_oc = w + oc * cin * kw;
    const float bias = b ? b[oc] : 0.0f;

    for (int64_t pos = 0; pos < lin; ++pos) {
      y_oc[pos] = bias;
    }

    for (int64_t ic = 0; ic < cin; ++ic) {
      const float* x_ic = x + ic * lin;
      const float* w_ic = w_oc + ic * kw;
      for (int64_t k = 0; k < kw; ++k) {
        const int64_t input_base = k * kernel->dilation - kernel->pad_left;
        int64_t pos = 0;
        if (input_base < 0) {
          pos = -input_base;
        }
        const int64_t end = lin - input_base < lin ? lin - input_base : lin;
        const float weight_v = w_ic[k];
#if defined(__AVX__)
        const __m256 weight_vec = _mm256_set1_ps(weight_v);
        for (; pos + 8 <= end; pos += 8) {
          __m256 acc = _mm256_loadu_ps(y_oc + pos);
          const __m256 xv = _mm256_loadu_ps(x_ic + input_base + pos);
#if defined(__FMA__)
          acc = _mm256_fmadd_ps(xv, weight_vec, acc);
#else
          acc = _mm256_add_ps(acc, _mm256_mul_ps(xv, weight_vec));
#endif
          _mm256_storeu_ps(y_oc + pos, acc);
        }
#endif
        for (; pos < end; ++pos) {
          y_oc[pos] += x_ic[input_base + pos] * weight_v;
        }
      }
    }

#if defined(__AVX__)
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 sumsq_vec = _mm256_setzero_ps();
    int64_t pos = 0;
    for (; pos + 8 <= lin; pos += 8) {
      const __m256 values = _mm256_loadu_ps(y_oc + pos);
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
    for (; pos < lin; ++pos) {
      const float value = y_oc[pos];
      sum += value;
      sumsq += (double)value * (double)value;
    }
#else
    for (int64_t pos = 0; pos < lin; ++pos) {
      const float value = y_oc[pos];
      sum += value;
      sumsq += (double)value * (double)value;
    }
#endif
    const double mean = sum / (double)lin;
    double var = sumsq / (double)lin - mean * mean;
    if (var < 0.0) var = 0.0;
    const float inv_std = 1.0f / sqrtf((float)var + 1e-5f);
    const float scale_v = s[oc];
    const float shift_v = t[oc];
#if defined(__AVX__)
    const __m256 mean_vec = _mm256_set1_ps((float)mean);
    const __m256 inv_std_vec = _mm256_set1_ps(inv_std);
    const __m256 scale_vec = _mm256_set1_ps(scale_v);
    const __m256 shift_vec = _mm256_set1_ps(shift_v);
    int64_t norm_pos = 0;
    for (; norm_pos + 8 <= lin; norm_pos += 8) {
      __m256 values = _mm256_loadu_ps(y_oc + norm_pos);
      values = _mm256_sub_ps(values, mean_vec);
      values = _mm256_mul_ps(values, inv_std_vec);
#if defined(__FMA__)
      values = _mm256_fmadd_ps(values, scale_vec, shift_vec);
#else
      values = _mm256_add_ps(_mm256_mul_ps(values, scale_vec), shift_vec);
#endif
      _mm256_storeu_ps(y_oc + norm_pos, values);
    }
    for (; norm_pos < lin; ++norm_pos) {
      y_oc[norm_pos] = (y_oc[norm_pos] - (float)mean) * inv_std * scale_v + shift_v;
    }
#else
    for (int64_t pos = 0; pos < lin; ++pos) {
      y_oc[pos] = (y_oc[pos] - (float)mean) * inv_std * scale_v + shift_v;
    }
#endif
  }

cleanup:
  if (input_info) g_ort->ReleaseTensorTypeAndShapeInfo(input_info);
  if (weight_info) g_ort->ReleaseTensorTypeAndShapeInfo(weight_info);
}

static OrtCustomOp c_Conv1dAdaInOp = {
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
  status = g_ort->CustomOpDomain_Add(domain, &c_Conv1dAdaInOp);
  if (status) return status;
  return g_ort->AddCustomOpDomain(options, domain);
}
