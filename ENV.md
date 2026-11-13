# ENV — environment probe (C0, 2026-11-02)

| Item | Value |
|---|---|
| OS | Linux 6.18.44-fc-v185 |
| CPU | Intel(R) Xeon(R) Processor @ 2.10GHz, 2 cores |
| RAM | 7 GB |
| Disk free | 30G |
| GPU | none (nvidia-smi absent) |
| Python | 3.13.16 |
| numpy / scipy / pandas / sklearn | 2.5.3 1.18.1 3.0.5 1.9.1 |
| matplotlib | 3.11.2 |
| OpenCV | 5.0.0 (no Haar/YuNet model files present) |
| onnxruntime | 1.29.0 (the onnx package is absent) |
| protobuf | 7.36.1 |
| web stack | starlette 1.6.0, uvicorn 0.53.0, Flask 3.1.3 |
| ffmpeg | 6.1.1-3ubuntu5; encoders: libx264, libvpx (VP8), ffv1 present |
| Java / Gradle | openjdk version "21.0.12.1" 2026-08-18; Gradle in /opt/gradle; **no Android SDK** (ANDROID_HOME unset, no sdkmanager) |
| pandoc / LibreOffice | present (report DOCX/PDF) |
| Write access | repository folder, scratchpad |

## Missing (cannot be installed: PyPI and download.pytorch.org refused by egress policy, 403)
PyTorch, statsmodels, pytest, coverage, onnx (python package), TensorFlow / TFLite converter, Android SDK.

## Network probe (curl, 10 s timeout; 000 = refused or unreachable)
- https://github.com → 400
- https://drive.google.com → 000
- https://sites.google.com → 000
- https://www.tu-ilmenau.de → 000
- https://ieee-dataport.org → 000
- https://mahnob-db.eu → 000
- https://huggingface.co → 000
- https://zenodo.org → 000
- https://figshare.com → 000
- https://physionet.org → 000
- https://www.kaggle.com → 000
- https://arxiv.org → 000
- https://pypi.org → 000
- git clone over HTTPS from github.com works for public repos (rPPG-Toolbox cloned to compare implementations).
