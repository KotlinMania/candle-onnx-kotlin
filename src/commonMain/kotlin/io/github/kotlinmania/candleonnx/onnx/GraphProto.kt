package io.github.kotlinmania.candleonnx.onnx

import io.github.kotlinmania.bytes.buf.Buf
import io.github.kotlinmania.bytes.buf.BufMut
import io.github.kotlinmania.prost.Message
import io.github.kotlinmania.prost.encoding.DecodeContext
import io.github.kotlinmania.prost.encoding.MessageEncoding
import io.github.kotlinmania.prost.encoding.StringEncoding
import io.github.kotlinmania.prost.encoding.WireType
import io.github.kotlinmania.prost.encoding.skipField

public data class GraphProto(
    var node: MutableList<NodeProto> = mutableListOf(),
    var name: String = "",
    var initializer: MutableList<TensorProto> = mutableListOf(),
    var docString: String = "",
    var input: MutableList<ValueInfoProto> = mutableListOf(),
    var output: MutableList<ValueInfoProto> = mutableListOf(),
    var valueInfo: MutableList<ValueInfoProto> = mutableListOf(),
    var quantizationAnnotation: MutableList<TensorAnnotation> = mutableListOf(),
    var sparseInitializer: MutableList<SparseTensorProto> = mutableListOf(),
) : Message {
    override fun encodeRaw(buf: BufMut) {
        if (node.isNotEmpty()) MessageEncoding.encodeRepeated(1u, node, buf)
        if (name.isNotEmpty()) StringEncoding.encode(2u, name, buf)
        if (initializer.isNotEmpty()) MessageEncoding.encodeRepeated(5u, initializer, buf)
        if (docString.isNotEmpty()) StringEncoding.encode(10u, docString, buf)
        if (input.isNotEmpty()) MessageEncoding.encodeRepeated(11u, input, buf)
        if (output.isNotEmpty()) MessageEncoding.encodeRepeated(12u, output, buf)
        if (valueInfo.isNotEmpty()) MessageEncoding.encodeRepeated(13u, valueInfo, buf)
        if (quantizationAnnotation.isNotEmpty()) {
            MessageEncoding.encodeRepeated(14u, quantizationAnnotation, buf)
        }
        if (sparseInitializer.isNotEmpty()) {
            MessageEncoding.encodeRepeated(15u, sparseInitializer, buf)
        }
    }

    override fun mergeField(
        tag: UInt,
        wireType: WireType,
        buf: Buf,
        ctx: DecodeContext,
    ): Result<Unit> =
        when (tag) {
            1u -> {
                val n = NodeProto()
                val res = MessageEncoding.merge(wireType, n, buf, ctx)
                if (res.isSuccess) {
                    node.add(n)
                    Result.success(Unit)
                } else res
            }
            2u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    name = res.getOrThrow()
                    Result.success(Unit)
                } else Result.failure(res.exceptionOrNull()!!)
            }
            5u -> {
                val t = TensorProto()
                val res = MessageEncoding.merge(wireType, t, buf, ctx)
                if (res.isSuccess) {
                    initializer.add(t)
                    Result.success(Unit)
                } else res
            }
            10u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    docString = res.getOrThrow()
                    Result.success(Unit)
                } else Result.failure(res.exceptionOrNull()!!)
            }
            11u -> {
                val v = ValueInfoProto()
                val res = MessageEncoding.merge(wireType, v, buf, ctx)
                if (res.isSuccess) {
                    input.add(v)
                    Result.success(Unit)
                } else res
            }
            12u -> {
                val v = ValueInfoProto()
                val res = MessageEncoding.merge(wireType, v, buf, ctx)
                if (res.isSuccess) {
                    output.add(v)
                    Result.success(Unit)
                } else res
            }
            13u -> {
                val v = ValueInfoProto()
                val res = MessageEncoding.merge(wireType, v, buf, ctx)
                if (res.isSuccess) {
                    valueInfo.add(v)
                    Result.success(Unit)
                } else res
            }
            14u -> {
                val qa = TensorAnnotation()
                val res = MessageEncoding.merge(wireType, qa, buf, ctx)
                if (res.isSuccess) {
                    quantizationAnnotation.add(qa)
                    Result.success(Unit)
                } else res
            }
            15u -> {
                val st = SparseTensorProto()
                val res = MessageEncoding.merge(wireType, st, buf, ctx)
                if (res.isSuccess) {
                    sparseInitializer.add(st)
                    Result.success(Unit)
                } else res
            }
            else -> skipField(wireType, tag, buf, ctx)
        }

    override fun encodedLen(): Int {
        var len = 0
        if (node.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(1u, node)
        if (name.isNotEmpty()) len += StringEncoding.encodedLen(2u, name)
        if (initializer.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(5u, initializer)
        if (docString.isNotEmpty()) len += StringEncoding.encodedLen(10u, docString)
        if (input.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(11u, input)
        if (output.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(12u, output)
        if (valueInfo.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(13u, valueInfo)
        if (quantizationAnnotation.isNotEmpty()) {
            len += MessageEncoding.encodedLenRepeated(14u, quantizationAnnotation)
        }
        if (sparseInitializer.isNotEmpty()) {
            len += MessageEncoding.encodedLenRepeated(15u, sparseInitializer)
        }
        return len
    }

    override fun clear() {
        node.clear()
        name = ""
        initializer.clear()
        sparseInitializer.clear()
        docString = ""
        input.clear()
        output.clear()
        valueInfo.clear()
        quantizationAnnotation.clear()
    }

    public companion object {
        public fun decode(buf: Buf): Result<GraphProto> =
            io.github.kotlinmania.prost.decode(buf) { GraphProto() }

        public fun decode(bytes: ByteArray): Result<GraphProto> =
            io.github.kotlinmania.prost.decode(bytes) { GraphProto() }
    }
}
