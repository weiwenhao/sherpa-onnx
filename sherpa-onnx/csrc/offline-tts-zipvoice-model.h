// sherpa-onnx/csrc/offline-tts-zipvoice-model.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_MODEL_H_

#include <memory>
#include <string>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-model-meta-data.h"

namespace sherpa_onnx {

class OfflineTtsZipvoiceModel {
 public:
  ~OfflineTtsZipvoiceModel();

  explicit OfflineTtsZipvoiceModel(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsZipvoiceModel(Manager *mgr, const OfflineTtsModelConfig &config);

  // Return a float32 tensor containing the mel
  // of shape (batch_size, mel_dim, num_frames)
  // seed < 0 keeps the original behavior (thread-local random device), so the
  // output differs on every call. seed >= 0 makes the initial flow-matching
  // noise deterministic, which upstream ZipVoice exposes as `--seed` but this
  // port used to drop; without it the same sentence cannot be reproduced.
  Ort::Value Run(Ort::Value tokens, Ort::Value prompt_tokens,
                 Ort::Value prompt_features, float speed, int32_t num_steps,
                 float t_shift = 0.5f, float guidance_scale = 1.0f,
                 int32_t seed = -1) const;

  const OfflineTtsZipvoiceModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_MODEL_H_
