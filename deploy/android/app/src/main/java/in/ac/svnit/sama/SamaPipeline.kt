// NOT COMPILED — see deploy/android/README.md. Mirrors sama/streaming.py, sama/methods.py (POS),
// sama/sama_method.py (weighting, sigma) and sama/conformal.py (Mondrian gate).
package `in`.ac.svnit.sama

import kotlin.math.*

enum class Reason { OK, WARMING_UP, FACE_LOST, FRAME_RATE, LOW_LIGHT, MOTION, IMPLAUSIBLE, INTERVAL_TOO_WIDE }

data class Reading(val shown: Boolean, val reason: Reason, val hr: Double? = null, val lo: Double? = null, val hi: Double? = null)

/** Gate parameters loaded from assets/sama_model.json (quantiles per stratum, ridge sigma model). */
data class GateModel(val q: Map<String, Double>, val halfwidthMax: Double, val sigmaMu: DoubleArray,
                     val sigmaSd: DoubleArray, val sigmaCoef: DoubleArray)

class SamaPipeline(private val gate: GateModel, private val net: SamaNet?, private val ita: Double?) {
    private val fps = 30.0
    private val windowS = 30.0
    private val times = ArrayDeque<Double>()
    private val frames = ArrayDeque<Array<DoubleArray>?>()   // 30 x 3 patch means, null = no face

    fun push(t: Double, patches: Array<DoubleArray>?): Reading? {
        times.addLast(t); frames.addLast(patches)
        while (times.isNotEmpty() && times.first() < t - windowS - 1.0) { times.removeFirst(); frames.removeFirst() }
        return estimateIfDue(t)
    }

    private var lastEmit = -1e9
    private fun estimateIfDue(t: Double): Reading? {
        if (t - lastEmit < 1.0) return null
        lastEmit = t
        if (times.isEmpty() || t - times.first() < windowS) return Reading(false, Reason.WARMING_UP)
        val lost = frames.count { it == null }.toDouble() / frames.size
        if (lost > 0.10) return Reading(false, Reason.FACE_LOST)
        // resample face-mean RGB and patches to a uniform 30 fps grid by linear interpolation
        val n = (windowS * fps).toInt()
        val grid = DoubleArray(n) { t - windowS + it / fps }
        val ts = times.filterIndexed { i, _ -> frames.elementAt(i) != null }.toDoubleArray()
        val fs = frames.filterNotNull()
        val face = Array(n) { DoubleArray(3) }
        for (c in 0 until 3) {
            val y = DoubleArray(fs.size) { i -> fs[i].sumOf { it[c] } / fs[i].size }
            val r = interp(grid, ts, y)
            for (k in 0 until n) face[k][c] = r[k]
        }
        val bright = face.sumOf { it[1] } / n
        if (bright < 25.0) return Reading(false, Reason.LOW_LIGHT)
        val pulse = pos(face)
        val (hrPos, snr) = spectralHr(pulse)
        // network refinement (step 4) when a model is present; otherwise the POS candidate
        val hr = net?.refineAndEstimate(face, ita ?: 15.0, bright) ?: hrPos
        if (hr.isNaN() || hr < 40 || hr > 180) return Reading(false, Reason.IMPLAUSIBLE)
        val sigma = exp(ridge(doubleArrayOf(snr, 0.0, snr, 0.0, bright, ita ?: 15.0, 30.0, 0.0, abs(hr - hrPos), snr, 0.0, abs(hr - hrPos))))
        val stratum = when { ita == null -> "darker"; ita > 28 -> "lighter"; ita > 10 -> "middle"; else -> "darker" }
        val hw = (gate.q[stratum] ?: Double.POSITIVE_INFINITY) * sigma
        if (!(hw <= gate.halfwidthMax)) return Reading(false, Reason.INTERVAL_TOO_WIDE)
        return Reading(true, Reason.OK, hr, hr - hw, hr + hw)
    }

    private fun ridge(q: DoubleArray): Double {
        val z = q.toMutableList(); z.add(q[0] * q[0] / 20.0); z.add(q[9] * q[9] / 20.0)
        var s = gate.sigmaCoef[0]
        for (i in z.indices) s += gate.sigmaCoef[i + 1] * (z[i] - gate.sigmaMu[i]) / gate.sigmaSd[i]
        return s
    }

    companion object {
        fun interp(x: DoubleArray, xp: DoubleArray, fp: DoubleArray): DoubleArray = DoubleArray(x.size) { k ->
            val i = xp.indexOfFirst { it >= x[k] }.let { if (it <= 0) 1 else it }.coerceAtMost(xp.size - 1)
            val w = ((x[k] - xp[i - 1]) / (xp[i] - xp[i - 1])).coerceIn(0.0, 1.0)
            fp[i - 1] + w * (fp[i] - fp[i - 1])
        }

        /** POS (Wang et al. 2017), 1.6 s overlap-add, as sama/methods.py pos(post=False). */
        fun pos(rgb: Array<DoubleArray>, fs: Double = 30.0): DoubleArray {
            val nn = rgb.size; val l = ceil(1.6 * fs).toInt(); val h = DoubleArray(nn)
            for (n in l until nn) {
                val m = n - l
                val mean = DoubleArray(3) { c -> (m until n).sumOf { rgb[it][c] } / l }
                val s0 = DoubleArray(l) { j -> rgb[m + j][1] / mean[1] - rgb[m + j][2] / mean[2] }
                val s1 = DoubleArray(l) { j -> -2 * rgb[m + j][0] / mean[0] + rgb[m + j][1] / mean[1] + rgb[m + j][2] / mean[2] }
                val a = sd(s0) / max(sd(s1), 1e-12)
                val hh = DoubleArray(l) { s0[it] + a * s1[it] }; val mu = hh.average()
                for (j in 0 until l) h[m + j] += hh[j] - mu
            }
            return h
        }

        fun sd(x: DoubleArray): Double { val m = x.average(); return sqrt(x.sumOf { (it - m) * (it - m) } / x.size) }

        /** Spectral peak in 0.7-3.0 Hz by direct DFT at 0.5 bpm steps; SNR as de Haan & Jeanne (2013). */
        fun spectralHr(x: DoubleArray, fs: Double = 30.0): Pair<Double, Double> {
            val m = x.average(); val y = DoubleArray(x.size) { (x[it] - m) * (0.5 - 0.5 * cos(2 * PI * it / (x.size - 1))) }
            val bpms = generateSequence(42.0) { it + 0.5 }.takeWhile { it <= 180.0 }.toList()
            val p = bpms.map { b -> val f = b / 60.0; var re = 0.0; var im = 0.0
                for (k in y.indices) { re += y[k] * cos(2 * PI * f * k / fs); im -= y[k] * sin(2 * PI * f * k / fs) }; re * re + im * im }
            val k = p.indices.maxByOrNull { p[it] } ?: return Pair(Double.NaN, Double.NEGATIVE_INFINITY)
            val f0 = bpms[k]
            val sig = p.indices.filter { abs(bpms[it] - f0) <= 6 || abs(bpms[it] - 2 * f0) <= 12 }.sumOf { p[it] }
            val noise = p.sum() - sig
            return Pair(f0, 10 * log10(sig / max(noise, 1e-30)))
        }
    }
}
