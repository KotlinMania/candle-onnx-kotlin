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
    ) {
        when (tag) {
            1u -> StringEncoding.decodeRepeated(wireType, input, buf, ctx)
            2u -> StringEncoding.decodeRepeated(wireType, output, buf, ctx)
            3u -> name = StringEncoding.decode(wireType, buf, ctx)
            4u -> opType = StringEncoding.decode(wireType, buf, ctx)
            5u -> attribute.add(MessageEncoding.decode(wireType, AttributeProto(), buf, ctx))
            6u -> docString = StringEncoding.decode(wireType, buf, ctx)
            7u -> domain = StringEncoding.decode(wireType, buf, ctx)
            else -> skipField(wireType, tag, buf, ctx)
        }
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
        public fun decode(bytes: ByteArray): NodeProto {
            val msg = NodeProto()
            msg.merge(Buf.wrap(bytes))
            return msg
        }
    }
}
