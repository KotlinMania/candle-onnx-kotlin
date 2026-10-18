package io.github.kotlinmania.candleonnx

import io.github.kotlinmania.candle.CandleException
import io.github.kotlinmania.candleonnx.onnx.ModelProto
import io.github.kotlinmania.io.buffered
import io.github.kotlinmania.io.files.Path
import io.github.kotlinmania.io.files.SystemFileSystem
import io.github.kotlinmania.io.readByteArray
import io.github.kotlinmania.io.use

public fun readFile(p: Path): ModelProto {
    val buf = try {
        SystemFileSystem.source(p).buffered().use { it.readByteArray() }
    } catch (e: Exception) {
        throw CandleException.wrap(e)
    }
    return try {
        ModelProto.decode(buf)
    } catch (e: Exception) {
        throw CandleException.wrap(e)
    }
}

public fun readFile(p: String): ModelProto = readFile(Path(p))
