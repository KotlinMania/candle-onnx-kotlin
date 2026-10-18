pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins { id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0" }

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "candle-onnx-kotlin"

listOf(
    "../candle-kotlin",
    "../candle-nn-kotlin",
    "../prost-kotlin",
    "../bytes-kotlin",
    "../thiserror-kotlin",
    "../km-io",
).forEach { path ->
    if (file(path).exists()) {
        includeBuild(path)
    }
}
