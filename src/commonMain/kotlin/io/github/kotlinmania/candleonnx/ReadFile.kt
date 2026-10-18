package io.github.kotlinmania.candleonnx

import io.github.kotlinmania.candle.CandleException
import io.github.kotlinmania.candleonnx.onnx.ModelProto
import io.github.kotlinmania.io.buffered
import io.github.kotlinmania.io.files.Path
import io.github.kotlinmania.io.files.SystemFileSystem
import io.github.kotlinmania.io.readByteArray

public fun readFile(p: Path): ModelProto {
    val src = try {
        SystemFileSystem.source(p).buffered()
    } catch (e: Exception) {
        throw CandleException.wrap(e)
    }
    val bytes = try {
        src.readByteArray()
    } finally {
        src.close()
    }
    return try {
        ModelProto.decode(bytes).getOrThrow()
    } catch (e: Exception) {
        throw CandleException.wrap(e)
    }
}

public fun readFile(p: String): ModelProto = readFile(Path(p))
