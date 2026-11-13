# LIMITATIONS — read before using anything in `deploy/`

**Not a medical device. Research use only. No diagnostic output.**

1. **Trained and tested on simulated data only.** Every weight in `deploy/model` and `deploy/onnx` was
   learned from SAMA-Sim (SYNTHETIC). Nothing here is evidence that the method works on real
   faces, on Indian skin, or on any phone. No REAL-PUBLIC dataset could be downloaded (ISSUES I-22).
2. **The simulator's skin-tone gradient was set to the proposal's own assumption** (D-12), and its
   noise premise follows Chari et al. (2020). Any benefit of steps 1–3 that follows from that
   premise is ASSUMPTION-DRIVEN (I-02).
3. **No neural baselines.** DeepPhys, TS-CAN, PhysNet and EfficientPhys could not be run (I-24);
   "best baseline" means the best of six unsupervised methods.
4. **The conformal gate's guarantee is marginal and per stratum**, holds only if deployment data
   are exchangeable with the calibration data (same light, device, codec mix), and is not
   guaranteed for the windows that are shown after abstention (I-27). Coverage fell under shift
   in the rehearsal (results/tables/c7_coverage_shift.csv).
5. **Skin-tone stratum at run time.** The phone cannot measure ITA° reliably (I-03, I-10). With no
   ITA° entered, the demonstrator uses the darker stratum's (widest) quantile (D-21): it abstains
   more for everyone rather than risk under-covering anyone.
6. **Equal coverage is not equal availability.** A single 5 bpm rule makes the harder stratum get
   fewer readings (I-08; results/tables/abstention_tradeoff.csv).
7. **No field monitoring of coverage.** A deployed phone has no reference heart rate (I-21). Drift
   proxies to monitor instead: abstention rate by reason, median interval half-width, spectral
   SNR distribution, brightness distribution. Recalibration protocol: re-run step 6 calibration on
   ≥ 30 participants per stratum with ECG whenever the device model, camera app, codec or
   operating-system camera pipeline changes, or when the monthly abstention rate moves by > 10
   percentage points.
8. **Face region.** The web demonstrator uses a fixed guide box, not a face mesh (D-20); beards,
   spectacles, bindis and hair inside the box are not masked except for clipped pixels.
9. **Android app** is source only, NOT COMPILED, NOT RUN (D-07); TFLite not produced.
10. **HRV**: the 30 fps frame interval limits beat timing (I-16); camera HRV readings are not shown
    by the demonstrator.
