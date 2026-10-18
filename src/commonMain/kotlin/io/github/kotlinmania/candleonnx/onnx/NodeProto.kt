package io.github.kotlinmania.candleonnx.onnx

import io.github.kotlinmania.bytes.buf.Buf
import io.github.kotlinmania.bytes.buf.BufMut
import io.github.kotlinmania.prost.Message
import io.github.kotlinmania.prost.encoding.DecodeContext
import io.github.kotlinmania.prost.encoding.MessageEncoding
import io.github.kotlinmania.prost.encoding.StringEncoding
import io.github.kotlinmania.prost.encoding.WireType
import io.github.kotlinmania.prost.encoding.skipField

public data class NodeProto(
    var input: MutableList<String> = mutableListOf(),
    var output: MutableList<String> = mutableListOf(),
    var name: String = "",
    var opType: String = "",
    var domain: String = "",
    var attribute: MutableList<AttributeProto> = mutableListOf(),
    var docString: String = "",
) : Message {
    override fun encodeRaw(buf: BufMut) {
        if (input.isNotEmpty()) StringEncoding.encodeRepeated(1u, input, buf)
        if (output.isNotEmpty()) StringEncoding.encodeRepeated(2u, output, buf)
        if (name.isNotEmpty()) StringEncoding.encode(3u, name, buf)
        if (opType.isNotEmpty()) StringEncoding.encode(4u, opType, buf)
        if (attribute.isNotEmpty()) MessageEncoding.encodeRepeated(5u, attribute, buf)
        if (docString.isNotEmpty()) StringEncoding.encode(6u, docString, buf)
        if (domain.isNotEmpty()) StringEncoding.encode(7u, domain, buf)
    }

    override fun mergeField(
        tag: UInt,
        wireType: WireType,
        buf: Buf,
        ctx: DecodeContext,
    ): Result<Unit> =
        when (tag) {
            1u -> StringEncoding.mergeRepeated(wireType, input, buf, ctx)
            2u -> StringEncoding.mergeRepeated(wireType, output, buf, ctx)
            3u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    name = res.getOrThrow()
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            4u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    opType = res.getOrThrow()
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            5u -> {
                val attr = AttributeProto()
                val res = MessageEncoding.merge(wireType, attr, buf, ctx)
                if (res.isSuccess) {
                    attribute.add(attr)
                    Result.success(Unit)
                } else {
                    res
                }
            }
            6u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    docString = res.getOrThrow()
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            7u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    domain = res.getOrThrow()
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            else -> skipField(wireType, tag, buf, ctx)
        }

    override fun encodedLen(): Int {
        var len = 0
        if (input.isNotEmpty()) len += StringEncoding.encodedLenRepeated(1u, input)
        if (output.isNotEmpty()) len += StringEncoding.encodedLenRepeated(2u, output)
        if (name.isNotEmpty()) len += StringEncoding.encodedLen(3u, name)
        if (opType.isNotEmpty()) len += StringEncoding.encodedLen(4u, opType)
        if (attribute.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(5u, attribute)
        if (docString.isNotEmpty()) len += StringEncoding.encodedLen(6u, docString)
        if (domain.isNotEmpty()) len += StringEncoding.encodedLen(7u, domain)
        return len
    }

    override fun clear() {
        input.clear()
        output.clear()
        name = ""
        opType = ""
        domain = ""
        attribute.clear()
        docString = ""
    }

    public companion object {
        public fun decode(buf: Buf): Result<NodeProto> =
            io.github.kotlinmania.prost.decode(buf) { NodeProto() }

        public fun decode(bytes: ByteArray): Result<NodeProto> =
            io.github.kotlinmania.prost.decode(bytes) { NodeProto() }
    }
}
