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
    ) {
        when (tag) {
            1u -> name = StringEncoding.decode(wireType, buf, ctx)
            4u -> StringEncoding.decodeRepeated(wireType, input, buf, ctx)
            5u -> StringEncoding.decodeRepeated(wireType, output, buf, ctx)
            6u -> StringEncoding.decodeRepeated(wireType, attribute, buf, ctx)
            7u -> node.add(MessageEncoding.decode(wireType, NodeProto(), buf, ctx))
            8u -> docString = StringEncoding.decode(wireType, buf, ctx)
            9u -> opsetImport.add(MessageEncoding.decode(wireType, OperatorSetIdProto(), buf, ctx))
            10u -> domain = StringEncoding.decode(wireType, buf, ctx)
            11u -> attributeProto.add(MessageEncoding.decode(wireType, AttributeProto(), buf, ctx))
            else -> skipField(wireType, tag, buf, ctx)
        }
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
        public fun decode(bytes: ByteArray): FunctionProto {
            val msg = FunctionProto()
            msg.merge(Buf.wrap(bytes))
            return msg
        }
    }
}
