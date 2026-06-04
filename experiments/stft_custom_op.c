#include <math.h>
#include <stdlib.h>

#include "onnxruntime_c_api.h"

#define KOKORO_STFT_FRAME_LENGTH 20
#define KOKORO_STFT_BINS 11

static const OrtApi* g_ort = NULL;

typedef struct {
  int64_t frame_step;
  double cos_table[KOKORO_STFT_BINS][KOKORO_STFT_FRAME_LENGTH];
  double sin_table[KOKORO_STFT_BINS][KOKORO_STFT_FRAME_LENGTH];
} KokoroStftKernel;

static void* ORT_API_CALL CreateKernel(const OrtCustomOp* op, const OrtApi* api,
                                       const OrtKernelInfo* info) {
  (void)op;
  g_ort = api;
  KokoroStftKernel* kernel = (KokoroStftKernel*)calloc(1, sizeof(KokoroStftKernel));
  if (!kernel) return NULL;
  kernel->frame_step = 5;
  OrtStatus* status = g_ort->KernelInfoGetAttribute_int64(info, "frame_step",
                                                          &kernel->frame_step);
  if (status) g_ort->ReleaseStatus(status);
  if (kernel->frame_step < 1) kernel->frame_step = 5;

  for (int b = 0; b < KOKORO_STFT_BINS; ++b) {
    for (int k = 0; k < KOKORO_STFT_FRAME_LENGTH; ++k) {
      const double angle = -2.0 * M_PI * (double)b * (double)k /
                           (double)KOKORO_STFT_FRAME_LENGTH;
      kernel->cos_table[b][k] = cos(angle);
      kernel->sin_table[b][k] = sin(angle);
    }
  }
  return kernel;
}

static const char* ORT_API_CALL GetName(const OrtCustomOp* op) {
  (void)op;
  return "KokoroSTFT";
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
  return 2;
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
  const KokoroStftKernel* kernel = (const KokoroStftKernel*)op_kernel;
  const OrtValue *signal = NULL, *window = NULL;
  OrtTensorTypeAndShapeInfo* signal_info = NULL;
  int64_t signal_dims[2] = {0, 0};
  const float *signal_data = NULL, *window_data = NULL;
  float* output_data = NULL;
  OrtValue* output = NULL;

  if (g_ort->KernelContext_GetInput(context, 0, &signal) != NULL) return;
  if (g_ort->KernelContext_GetInput(context, 1, &window) != NULL) return;
  if (g_ort->GetTensorTypeAndShape(signal, &signal_info) != NULL) return;
  if (g_ort->GetDimensions(signal_info, signal_dims, 2) != NULL) goto cleanup;

  const int64_t batch = signal_dims[0];
  const int64_t length = signal_dims[1];
  if (batch != 1 || length < KOKORO_STFT_FRAME_LENGTH) goto cleanup;
  const int64_t frames =
      ((length - KOKORO_STFT_FRAME_LENGTH) / kernel->frame_step) + 1;
  const int64_t output_dims[4] = {1, frames, KOKORO_STFT_BINS, 2};

  if (g_ort->KernelContext_GetOutput(context, 0, output_dims, 4, &output) != NULL)
    goto cleanup;
  if (g_ort->GetTensorMutableData((OrtValue*)signal, (void**)&signal_data) != NULL)
    goto cleanup;
  if (g_ort->GetTensorMutableData((OrtValue*)window, (void**)&window_data) != NULL)
    goto cleanup;
  if (g_ort->GetTensorMutableData(output, (void**)&output_data) != NULL) goto cleanup;

  for (int64_t frame = 0; frame < frames; ++frame) {
    const int64_t offset = frame * kernel->frame_step;
    for (int bin = 0; bin < KOKORO_STFT_BINS; ++bin) {
      double real = 0.0;
      double imag = 0.0;
      for (int k = 0; k < KOKORO_STFT_FRAME_LENGTH; ++k) {
        const double value = (double)signal_data[offset + k] * (double)window_data[k];
        real += value * kernel->cos_table[bin][k];
        imag += value * kernel->sin_table[bin][k];
      }
      const int64_t base = ((frame * KOKORO_STFT_BINS) + bin) * 2;
      output_data[base] = (float)real;
      output_data[base + 1] = (float)imag;
    }
  }

cleanup:
  if (signal_info) g_ort->ReleaseTensorTypeAndShapeInfo(signal_info);
}

static OrtCustomOp c_KokoroStftOp = {
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

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                          const OrtApiBase* api_base) {
  g_ort = api_base->GetApi(ORT_API_VERSION);
  OrtCustomOpDomain* domain = NULL;
  OrtStatus* status = g_ort->CreateCustomOpDomain("fastkokoro", &domain);
  if (status) return status;
  status = g_ort->CustomOpDomain_Add(domain, &c_KokoroStftOp);
  if (status) return status;
  return g_ort->AddCustomOpDomain(options, domain);
}
