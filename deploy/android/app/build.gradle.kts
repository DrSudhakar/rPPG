plugins { id("com.android.application") version "8.5.0"; id("org.jetbrains.kotlin.android") version "1.9.24" }
android {
    namespace = "in.ac.svnit.sama"
    compileSdk = 34
    defaultConfig { applicationId = "in.ac.svnit.sama"; minSdk = 26; targetSdk = 34; versionCode = 1; versionName = "0.1.0-research" }
    androidResources { noCompress += "tflite" }
}
dependencies {
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
}
