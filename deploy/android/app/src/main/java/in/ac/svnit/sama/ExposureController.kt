// NOT COMPILED. SAMA step 1: skin-tone-adaptive exposure (mirrors sim/samasim.py choose_exposure,
// exposure mode "adaptive"). Used on every repeating Camera2 capture request after a 10 s
// metering phase; AE is switched off and exposure time, ISO and white balance are locked.
package `in`.ac.svnit.sama

import android.hardware.camera2.CaptureRequest
import kotlin.math.max
import kotlin.math.min

class ExposureController(
    private val targetFrac: Double = 0.50,       // face mean as a share of full scale (fixed from the pilot)
    private val maxExposureNs: Long = 33_333_333, // 1/30 s cap keeps 30 fps
    private val minIso: Int = 100, private val maxIso: Int = 1600,
    private val maxClippedFrac: Double = 0.01,
) {
    var exposureNs: Long = 10_000_000; private set
    var iso: Int = minIso; private set

    /** One metering update from the current face statistics (linear, 0..1 of full scale). */
    fun update(faceMeanLinear: Double, clippedSkinFrac: Double) {
        if (faceMeanLinear <= 0) return
        var want = targetFrac / faceMeanLinear                    // required brightness multiplier
        if (clippedSkinFrac > maxClippedFrac) want = min(want, 0.9)  // back off when >1 % of skin clips
        val total = exposureNs.toDouble() * iso / minIso * want     // exposure-ISO product in base-ISO ns
        exposureNs = min(total.toLong(), maxExposureNs)              // exposure first (more photons) ...
        iso = (minIso * total / exposureNs).toInt().coerceIn(minIso, maxIso)  // ... then gain
        exposureNs = max(exposureNs, 100_000)
    }

    fun apply(b: CaptureRequest.Builder) {
        b.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
        b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureNs)
        b.set(CaptureRequest.SENSOR_SENSITIVITY, iso)
        b.set(CaptureRequest.SENSOR_FRAME_DURATION, maxExposureNs)
        b.set(CaptureRequest.CONTROL_AWB_LOCK, true)
        b.set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_OFF)  // keep the pulse
        b.set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF)
    }
}
