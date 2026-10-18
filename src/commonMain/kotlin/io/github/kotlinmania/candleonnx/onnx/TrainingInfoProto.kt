package io.github.kotlinmania.candleonnx.onnx

import io.github.kotlinmania.bytes.buf.Buf
import io.github.kotlinmania.bytes.buf.BufMut
import io.github.kotlinmania.prost.Message
import io.github.kotlinmania.prost.encoding.DecodeContext
import io.github.kotlinmania.prost.encoding.MessageEncoding
import io.github.kotlinmania.prost.encoding.WireType
import io.github.kotlinmania.prost.encoding.skipField

public data class TrainingInfoProto(
    var initialization: GraphProto? = null,
    var algorithm: GraphProto? = null,
    var initializationBinding: MutableList<StringStringEntryProto> = mutableListOf(),
    var updateBinding: MutableList<StringStringEntryProto> = mutableListOf(),
) : Message {
    override fun encodeRaw(buf: BufMut) {
        initialization?.let { MessageEncoding.encode(1u, it, buf) }
        algorithm?.let { MessageEncoding.encode(2u, it, buf) }
        if (initializationBinding.isNotEmpty()) {
            MessageEncoding.encodeRepeated(3u, initializationBinding, buf)
        }
        if (updateBinding.isNotEmpty()) {
            MessageEncoding.encodeRepeated(4u, updateBinding, buf)
        }
    }

    override fun mergeField(
        tag: UInt,
        wireType: WireType,
        buf: Buf,
        ctx: DecodeContext,
    ) {
        when (tag) {
            1u -> {
                val init = initialization ?: GraphProto().also { initialization = it }
                MessageEncoding.merge(wireType, init, buf, ctx)
            }
            2u -> {
                val alg = algorithm ?: GraphProto().also { algorithm = it }
                MessageEncoding.merge(wireType, alg, buf, ctx)
            }
            3u -> initializationBinding.add(
                MessageEncoding.decode(wireType, StringStringEntryProto(), buf, ctx),
            )
            4u -> updateBinding.add(
                MessageEncoding.decode(wireType, StringStringEntryProto(), buf, ctx),
            )
            else -> skipField(wireType, tag, buf, ctx)
        }
    }

    override fun encodedLen(): Int {
        var len = 0
        initialization?.let { len += MessageEncoding.encodedLen(1u, it) }
        algorithm?.let { len += MessageEncoding.encodedLen(2u, it) }
        if (initializationBinding.isNotEmpty()) {
            len += MessageEncoding.encodedLenRepeated(3u, initializationBinding)
        }
        if (updateBinding.isNotEmpty()) {
            len += MessageEncoding.encodedLenRepeated(4u, updateBinding)
        }
        return len
    }

    override fun clear() {
        initialization = null
        algorithm = null
        initializationBinding.clear()
        updateBinding.clear()
    }

    public companion object {
        public fun decode(bytes: ByteArray): TrainingInfoProto {
            val msg = TrainingInfoProto()
            msg.merge(Buf.wrap(bytes))
            return msg
        }
    }
}
