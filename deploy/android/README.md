# SAMA Android research demonstrator — source only, **NOT COMPILED, NOT RUN**

No Android SDK was available in the build environment (ENV.md), so this source has never been
compiled or run on a device. No latency or battery number exists for it; do not quote one.

What it does (steps 1–6 on the phone, no network):
1. `ExposureController` — Camera2 manual exposure: raises exposure time (≤ 1/30 s) then ISO until
   the face's mean green reaches the target share of full scale with < 1 % skin pixels clipped (step 1).
2. `PatchExtractor` — 30 patch means in the shared layout (`sama/model_io.py` LAYOUT).
3. `SamaPipeline` — ring buffer (30 s), POS, patch weighting, network (TFLite), σ̂ (ridge
   coefficients from `deploy/model/sama_model.json`), Mondrian gate, abstention codes.
4. UI shows a number only when the gate passes, otherwise **"No reliable reading"**, with a
   permanent "Research use only — not a medical device" label.

Privacy: the manifest requests CAMERA only — no INTERNET permission; frames are never written.

## Build steps (on a machine with Android Studio / SDK 34)
1. Convert the model: `python -m pip install onnx onnx2tf tensorflow` then
   `onnx2tf -i ../onnx/sama_net_fp32.onnx -o tflite_out` and copy `tflite_out/sama_net_fp32_float32.tflite`
   to `app/src/main/assets/sama_net.tflite`; copy `../model/sama_model.json` to `app/src/main/assets/`.
2. `gradle wrapper && ./gradlew assembleDebug` (AGP 8.x, Kotlin 1.9+, minSdk 26).
3. Verify parity: run `SamaPipelineTest` (instrumented) against `../../results/tables/export_parity.csv` cases.
4. Only then measure latency/battery on the study phones, labelled as device measurements.
