// NOT COMPILED. TFLite wrapper for the step-4 network (input 1x4x450 at 15 Hz, cond 1x2).
package `in`.ac.svnit.sama

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel

class SamaNet(context: Context) {
    private val interpreter: Interpreter = context.assets.openFd("sama_net.tflite").use { fd ->
        FileInputStream(fd.fileDescriptor).channel.use { ch ->
            Interpreter(ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength))
        }
    }

    /** Builds the 4-channel input (weighted POS + relative R, G, B), runs the network and returns
     *  the heart rate of its output's spectral peak. Inputs are band-passed, decimated to 15 Hz and
     *  z-scored exactly as sama/sama_method.py window_features. */
    fun refineAndEstimate(face: Array<DoubleArray>, ita: Double, brightness: Double): Double {
        val pos = SamaPipeline.pos(face)
        val chans = arrayOf(pos, DoubleArray(face.size) { face[it][0] }, DoubleArray(face.size) { face[it][1] }, DoubleArray(face.size) { face[it][2] })
        val x = Array(1) { Array(4) { c -> zscore(decimate2(chans[c])).map { it.toFloat() }.toFloatArray() } }
        val cond = arrayOf(floatArrayOf((ita / 50.0).toFloat(), (brightness / 128.0).toFloat()))
        val y = Array(1) { FloatArray(450) }
        interpreter.runForMultipleInputsOutputs(arrayOf(x, cond), mapOf(0 to y))
        return SamaPipeline.spectralHr(y[0].map { it.toDouble() }.toDoubleArray(), 15.0).first
    }

    private fun decimate2(x: DoubleArray) = DoubleArray(450) { i -> if (2 * i + 1 < x.size) 0.5 * (x[2 * i] + x[2 * i + 1]) else 0.0 }
    private fun zscore(x: DoubleArray): DoubleArray { val m = x.average(); val s = SamaPipeline.sd(x) + 1e-9; return DoubleArray(x.size) { (x[it] - m) / s } }
}
