// NOT COMPILED. Minimal activity: front camera preview -> PatchExtractor -> SamaPipeline -> text.
// The camera session setup (Camera2 boilerplate) is abbreviated; ExposureController.apply() is
// called on every repeating request. No frame is written to storage; there is no network permission.
package `in`.ac.svnit.sama

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    private lateinit var pipeline: SamaPipeline
    private lateinit var status: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        status = TextView(this).apply { textSize = 28f; text = getString(R.string.warming_up) }
        setContentView(status)
        title = getString(R.string.research_label)
        val gate = ModelAssets.loadGate(this)          // parses assets/sama_model.json
        pipeline = SamaPipeline(gate, SamaNet(this), ita = null)   // unknown tone -> darker quantile (D-21)
        // CameraSession.start(this, ExposureController(), onFrame = ::onFrame)  // Camera2 setup omitted
    }

    /** Called per camera frame with the YUV->RGB face box already cropped to the guide rectangle. */
    fun onFrame(tSeconds: Double, faceBoxRgb: Array<IntArray>?, width: Int, height: Int) {
        val patches = faceBoxRgb?.let { PatchExtractor.patchMeans(it, width, height) }
        val r = pipeline.push(tSeconds, patches) ?: return
        runOnUiThread {
            status.text = if (r.shown) "%.0f bpm\n(90%% interval %.0f–%.0f)".format(r.hr, r.lo, r.hi)
                          else getString(R.string.no_reading) + "\n" + r.reason.name.lowercase().replace('_', ' ')
        }
    }
}

object PatchExtractor {
    // Same layout as sama/model_io.py LAYOUT: forehead 2x6, each cheek 3x3, fractions of the face box.
    private val layout: List<DoubleArray> = buildList {
        for (r in 0 until 2) for (c in 0 until 6) add(doubleArrayOf(0.10 + 0.8 * c / 6, 0.06 + 0.12 * r, 0.8 / 6, 0.12))
        for (r in 0 until 3) for (c in 0 until 3) add(doubleArrayOf(0.06 + 0.11 * c, 0.46 + 0.10 * r, 0.11, 0.10))
        for (r in 0 until 3) for (c in 0 until 3) add(doubleArrayOf(0.61 + 0.11 * c, 0.46 + 0.10 * r, 0.11, 0.10))
    }

    /** rgb: rows of packed ARGB ints. Ignores clipped (>= 250) and black pixels. */
    fun patchMeans(rgb: Array<IntArray>, w: Int, h: Int): Array<DoubleArray> = Array(30) { i ->
        val (x, y, pw, ph) = layout[i].let { listOf(it[0], it[1], it[2], it[3]) }
        var r = 0.0; var g = 0.0; var b = 0.0; var n = 0
        for (yy in (y * h).toInt() until ((y + ph) * h).toInt()) for (xx in (x * w).toInt() until ((x + pw) * w).toInt()) {
            val p = rgb[yy][xx]; val R = (p shr 16) and 255; val G = (p shr 8) and 255; val B = p and 255
            if (maxOf(R, G, B) >= 250 || minOf(R, G, B) <= 2) continue
            r += R; g += G; b += B; n++
        }
        if (n == 0) doubleArrayOf(0.0, 0.0, 0.0) else doubleArrayOf(r / n, g / n, b / n)
    }
}

object ModelAssets {
    fun loadGate(ctx: android.content.Context): GateModel {
        val j = org.json.JSONObject(ctx.assets.open("sama_model.json").bufferedReader().readText())
        val g = j.getJSONObject("gate"); val s = j.getJSONObject("sigma"); val q = g.getJSONObject("q")
        fun arr(a: org.json.JSONArray) = DoubleArray(a.length()) { a.getDouble(it) }
        return GateModel(q.keys().asSequence().associateWith { q.getDouble(it) }, g.getDouble("halfwidth_max"),
                         arr(s.getJSONArray("mu")), arr(s.getJSONArray("sd")), arr(s.getJSONArray("coef")))
    }
}
