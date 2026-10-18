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
    var sparseInitializer: MutableList<SparseTensorProto> = mutableListOf(),
    var docString: String = "",
    var input: MutableList<ValueInfoProto> = mutableListOf(),
    var output: MutableList<ValueInfoProto> = mutableListOf(),
    var valueInfo: MutableList<ValueInfoProto> = mutableListOf(),
    var quantizationAnnotation: MutableList<TensorAnnotation> = mutableListOf(),
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
    ) {
        when (tag) {
            1u -> node.add(MessageEncoding.decode(wireType, NodeProto(), buf, ctx))
            2u -> name = StringEncoding.decode(wireType, buf, ctx)
            5u -> initializer.add(MessageEncoding.decode(wireType, TensorProto(), buf, ctx))
            10u -> docString = StringEncoding.decode(wireType, buf, ctx)
            11u -> input.add(MessageEncoding.decode(wireType, ValueInfoProto(), buf, ctx))
            12u -> output.add(MessageEncoding.decode(wireType, ValueInfoProto(), buf, ctx))
            13u -> valueInfo.add(MessageEncoding.decode(wireType, ValueInfoProto(), buf, ctx))
            14u -> quantizationAnnotation.add(
                MessageEncoding.decode(wireType, TensorAnnotation(), buf, ctx),
            )
            15u -> sparseInitializer.add(
                MessageEncoding.decode(wireType, SparseTensorProto(), buf, ctx),
            )
            else -> skipField(wireType, tag, buf, ctx)
        }
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
        public fun decode(bytes: ByteArray): GraphProto {
            val msg = GraphProto()
            msg.merge(Buf.wrap(bytes))
            return msg
        }
    }
}
