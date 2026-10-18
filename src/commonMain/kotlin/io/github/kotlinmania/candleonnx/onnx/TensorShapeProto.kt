package io.github.kotlinmania.candleonnx.onnx

import io.github.kotlinmania.bytes.buf.Buf
import io.github.kotlinmania.bytes.buf.BufMut
import io.github.kotlinmania.prost.Message
import io.github.kotlinmania.prost.encoding.DecodeContext
import io.github.kotlinmania.prost.encoding.Int64Encoding
import io.github.kotlinmania.prost.encoding.MessageEncoding
import io.github.kotlinmania.prost.encoding.StringEncoding
import io.github.kotlinmania.prost.encoding.WireType
import io.github.kotlinmania.prost.encoding.skipField

public data class TensorShapeProto(
    var dim: MutableList<Dimension> = mutableListOf(),
) : Message {
    public data class Dimension(
        var value: Value? = null,
        var denotation: String = "",
    ) : Message {
        public sealed interface Value {
            public data class DimValue(
                val value: Long,
            ) : Value

            public data class DimParam(
                val value: String,
            ) : Value
        }

        override fun encodeRaw(buf: BufMut) {
            when (val v = value) {
                is Value.DimValue -> Int64Encoding.encode(1u, v.value, buf)
                is Value.DimParam -> StringEncoding.encode(2u, v.value, buf)
                null -> {}
            }
            if (denotation.isNotEmpty()) StringEncoding.encode(3u, denotation, buf)
        }

        override fun mergeField(
            tag: UInt,
            wireType: WireType,
            buf: Buf,
            ctx: DecodeContext,
        ): Result<Unit> =
            when (tag) {
                1u -> {
                    val res = Int64Encoding.merge(wireType, buf, ctx)
                    if (res.isSuccess) {
                        value = Value.DimValue(res.getOrThrow())
                        Result.success(Unit)
                    } else {
                        Result.failure(res.exceptionOrNull()!!)
                    }
                }
                2u -> {
                    val res = StringEncoding.merge(wireType, buf, ctx)
                    if (res.isSuccess) {
                        value = Value.DimParam(res.getOrThrow())
                        Result.success(Unit)
                    } else {
                        Result.failure(res.exceptionOrNull()!!)
                    }
                }
                3u -> {
                    val res = StringEncoding.merge(wireType, buf, ctx)
                    if (res.isSuccess) {
                        denotation = res.getOrThrow()
                        Result.success(Unit)
                    } else {
                        Result.failure(res.exceptionOrNull()!!)
                    }
                }
                else -> skipField(wireType, tag, buf, ctx)
            }

        override fun encodedLen(): Int {
            var len = 0
            when (val v = value) {
                is Value.DimValue -> len += Int64Encoding.encodedLen(1u, v.value)
                is Value.DimParam -> len += StringEncoding.encodedLen(2u, v.value)
                null -> {}
            }
            if (denotation.isNotEmpty()) len += StringEncoding.encodedLen(3u, denotation)
            return len
        }

        override fun clear() {
            value = null
            denotation = ""
        }
    }

    override fun encodeRaw(buf: BufMut) {
        if (dim.isNotEmpty()) MessageEncoding.encodeRepeated(1u, dim, buf)
    }

    override fun mergeField(
        tag: UInt,
        wireType: WireType,
        buf: Buf,
        ctx: DecodeContext,
    ): Result<Unit> =
        when (tag) {
            1u -> MessageEncoding.mergeRepeated(wireType, dim, { Dimension() }, buf, ctx)
            else -> skipField(wireType, tag, buf, ctx)
        }

    override fun encodedLen(): Int =
        if (dim.isNotEmpty()) MessageEncoding.encodedLenRepeated(1u, dim) else 0

    override fun clear() {
        dim.clear()
    }
}
