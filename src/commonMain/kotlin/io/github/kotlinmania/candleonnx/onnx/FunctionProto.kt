package io.github.kotlinmania.candleonnx.onnx

import io.github.kotlinmania.bytes.buf.Buf
import io.github.kotlinmania.bytes.buf.BufMut
import io.github.kotlinmania.prost.Message
import io.github.kotlinmania.prost.encoding.DecodeContext
import io.github.kotlinmania.prost.encoding.MessageEncoding
import io.github.kotlinmania.prost.encoding.StringEncoding
import io.github.kotlinmania.prost.encoding.WireType
import io.github.kotlinmania.prost.encoding.skipField

public data class FunctionProto(
    var name: String = "",
    var input: MutableList<String> = mutableListOf(),
    var output: MutableList<String> = mutableListOf(),
    var attribute: MutableList<String> = mutableListOf(),
    var node: MutableList<NodeProto> = mutableListOf(),
    var docString: String = "",
    var opsetImport: MutableList<OperatorSetIdProto> = mutableListOf(),
    var domain: String = "",
    var attributeProto: MutableList<AttributeProto> = mutableListOf(),
) : Message {
    override fun encodeRaw(buf: BufMut) {
        if (name.isNotEmpty()) StringEncoding.encode(1u, name, buf)
        if (input.isNotEmpty()) StringEncoding.encodeRepeated(4u, input, buf)
        if (output.isNotEmpty()) StringEncoding.encodeRepeated(5u, output, buf)
        if (attribute.isNotEmpty()) StringEncoding.encodeRepeated(6u, attribute, buf)
        if (node.isNotEmpty()) MessageEncoding.encodeRepeated(7u, node, buf)
        if (docString.isNotEmpty()) StringEncoding.encode(8u, docString, buf)
        if (opsetImport.isNotEmpty()) MessageEncoding.encodeRepeated(9u, opsetImport, buf)
        if (domain.isNotEmpty()) StringEncoding.encode(10u, domain, buf)
        if (attributeProto.isNotEmpty()) MessageEncoding.encodeRepeated(11u, attributeProto, buf)
    }

    override fun mergeField(
        tag: UInt,
        wireType: WireType,
        buf: Buf,
        ctx: DecodeContext,
    ): Result<Unit> =
        when (tag) {
            1u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    name = res.getOrThrow()
                    Result.success(Unit)
                } else Result.failure(res.exceptionOrNull()!!)
            }
            4u -> StringEncoding.mergeRepeated(wireType, input, buf, ctx)
            5u -> StringEncoding.mergeRepeated(wireType, output, buf, ctx)
            6u -> StringEncoding.mergeRepeated(wireType, attribute, buf, ctx)
            7u -> {
                val n = NodeProto()
                val res = MessageEncoding.merge(wireType, n, buf, ctx)
                if (res.isSuccess) {
                    node.add(n)
                    Result.success(Unit)
                } else res
            }
            8u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    docString = res.getOrThrow()
                    Result.success(Unit)
                } else Result.failure(res.exceptionOrNull()!!)
            }
            9u -> {
                val op = OperatorSetIdProto()
                val res = MessageEncoding.merge(wireType, op, buf, ctx)
                if (res.isSuccess) {
                    opsetImport.add(op)
                    Result.success(Unit)
                } else res
            }
            10u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    domain = res.getOrThrow()
                    Result.success(Unit)
                } else Result.failure(res.exceptionOrNull()!!)
            }
            11u -> {
                val attr = AttributeProto()
                val res = MessageEncoding.merge(wireType, attr, buf, ctx)
                if (res.isSuccess) {
                    attributeProto.add(attr)
                    Result.success(Unit)
                } else res
            }
            else -> skipField(wireType, tag, buf, ctx)
        }

    override fun encodedLen(): Int {
        var len = 0
        if (name.isNotEmpty()) len += StringEncoding.encodedLen(1u, name)
        if (input.isNotEmpty()) len += StringEncoding.encodedLenRepeated(4u, input)
        if (output.isNotEmpty()) len += StringEncoding.encodedLenRepeated(5u, output)
        if (attribute.isNotEmpty()) len += StringEncoding.encodedLenRepeated(6u, attribute)
        if (node.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(7u, node)
        if (docString.isNotEmpty()) len += StringEncoding.encodedLen(8u, docString)
        if (opsetImport.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(9u, opsetImport)
        if (domain.isNotEmpty()) len += StringEncoding.encodedLen(10u, domain)
        if (attributeProto.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(11u, attributeProto)
        return len
    }

    override fun clear() {
        name = ""
        input.clear()
        output.clear()
        attribute.clear()
        node.clear()
        docString = ""
        opsetImport.clear()
        domain = ""
        attributeProto.clear()
    }

    public companion object {
        public fun decode(buf: Buf): Result<FunctionProto> =
            io.github.kotlinmania.prost.decode(buf) { FunctionProto() }

        public fun decode(bytes: ByteArray): Result<FunctionProto> =
            io.github.kotlinmania.prost.decode(bytes) { FunctionProto() }
    }
}
