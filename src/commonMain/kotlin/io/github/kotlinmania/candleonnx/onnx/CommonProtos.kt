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

public enum class Version(
    public val value: Int,
) {
    START_VERSION(0),
    IR_VERSION_2017_10_10(1),
    IR_VERSION_2017_10_30(2),
    IR_VERSION_2017_11_3(3),
    IR_VERSION_2019_1_22(4),
    IR_VERSION_2019_3_18(5),
    IR_VERSION_2019_9_19(6),
    IR_VERSION_2020_5_8(7),
    IR_VERSION_2021_7_30(8),
    IR_VERSION(9),
    ;

    public companion object {
        public fun fromI32(value: Int): Version? =
            entries.firstOrNull { it.value == value }
    }
}

public enum class OperatorStatus(
    public val value: Int,
) {
    EXPERIMENTAL(0),
    STABLE(1),
    ;

    public companion object {
        public fun fromI32(value: Int): OperatorStatus? =
            entries.firstOrNull { it.value == value }
    }
}

public data class StringStringEntryProto(
    var key: String = "",
    var value: String = "",
) : Message {
    override fun encodeRaw(buf: BufMut) {
        if (key.isNotEmpty()) StringEncoding.encode(1u, key, buf)
        if (value.isNotEmpty()) StringEncoding.encode(2u, value, buf)
    }

    override fun mergeField(
        tag: UInt,
        wireType: WireType,
        buf: Buf,
        ctx: DecodeContext,
    ) {
        when (tag) {
            1u -> key = StringEncoding.decode(wireType, buf, ctx)
            2u -> value = StringEncoding.decode(wireType, buf, ctx)
            else -> skipField(wireType, tag, buf, ctx)
        }
    }

    override fun encodedLen(): Int {
        var len = 0
        if (key.isNotEmpty()) len += StringEncoding.encodedLen(1u, key)
        if (value.isNotEmpty()) len += StringEncoding.encodedLen(2u, value)
        return len
    }

    override fun clear() {
        key = ""
        value = ""
    }

    public companion object {
        public fun decode(bytes: ByteArray): StringStringEntryProto {
            val msg = StringStringEntryProto()
            msg.merge(Buf.wrap(bytes))
            return msg
        }
    }
}

public data class TensorAnnotation(
    var tensorName: String = "",
    var quantParameterTensorNames: MutableList<StringStringEntryProto> = mutableListOf(),
) : Message {
    override fun encodeRaw(buf: BufMut) {
        if (tensorName.isNotEmpty()) StringEncoding.encode(1u, tensorName, buf)
        if (quantParameterTensorNames.isNotEmpty()) {
            MessageEncoding.encodeRepeated(2u, quantParameterTensorNames, buf)
        }
    }

    override fun mergeField(
        tag: UInt,
        wireType: WireType,
        buf: Buf,
        ctx: DecodeContext,
    ) {
        when (tag) {
            1u -> tensorName = StringEncoding.decode(wireType, buf, ctx)
            2u -> quantParameterTensorNames.add(
                MessageEncoding.decode(wireType, StringStringEntryProto(), buf, ctx),
            )
            else -> skipField(wireType, tag, buf, ctx)
        }
    }

    override fun encodedLen(): Int {
        var len = 0
        if (tensorName.isNotEmpty()) len += StringEncoding.encodedLen(1u, tensorName)
        if (quantParameterTensorNames.isNotEmpty()) {
            len += MessageEncoding.encodedLenRepeated(2u, quantParameterTensorNames)
        }
        return len
    }

    override fun clear() {
        tensorName = ""
        quantParameterTensorNames.clear()
    }

    public companion object {
        public fun decode(bytes: ByteArray): TensorAnnotation {
            val msg = TensorAnnotation()
            msg.merge(Buf.wrap(bytes))
            return msg
        }
    }
}

public data class ValueInfoProto(
    var name: String = "",
    var type: TypeProto? = null,
    var docString: String = "",
) : Message {
    public fun type(): TypeProto? = type

    public fun r#type(): TypeProto? = type

    override fun encodeRaw(buf: BufMut) {
        if (name.isNotEmpty()) StringEncoding.encode(1u, name, buf)
        type?.let { MessageEncoding.encode(2u, it, buf) }
        if (docString.isNotEmpty()) StringEncoding.encode(3u, docString, buf)
    }

    override fun mergeField(
        tag: UInt,
        wireType: WireType,
        buf: Buf,
        ctx: DecodeContext,
    ) {
        when (tag) {
            1u -> name = StringEncoding.decode(wireType, buf, ctx)
            2u -> {
                val t = type ?: TypeProto().also { type = it }
                MessageEncoding.merge(wireType, t, buf, ctx)
            }
            3u -> docString = StringEncoding.decode(wireType, buf, ctx)
            else -> skipField(wireType, tag, buf, ctx)
        }
    }

    override fun encodedLen(): Int {
        var len = 0
        if (name.isNotEmpty()) len += StringEncoding.encodedLen(1u, name)
        type?.let { len += MessageEncoding.encodedLen(2u, it) }
        if (docString.isNotEmpty()) len += StringEncoding.encodedLen(3u, docString)
        return len
    }

    override fun clear() {
        name = ""
        type = null
        docString = ""
    }

    public companion object {
        public fun decode(bytes: ByteArray): ValueInfoProto {
            val msg = ValueInfoProto()
            msg.merge(Buf.wrap(bytes))
            return msg
        }
    }
}

public data class SparseTensorProto(
    var values: TensorProto? = null,
    var indices: TensorProto? = null,
    var dims: MutableList<Long> = mutableListOf(),
) : Message {
    override fun encodeRaw(buf: BufMut) {
        values?.let { MessageEncoding.encode(1u, it, buf) }
        indices?.let { MessageEncoding.encode(2u, it, buf) }
        if (dims.isNotEmpty()) Int64Encoding.encodePacked(3u, dims, buf)
    }

    override fun mergeField(
        tag: UInt,
        wireType: WireType,
        buf: Buf,
        ctx: DecodeContext,
    ) {
        when (tag) {
            1u -> {
                val v = values ?: TensorProto().also { values = it }
                MessageEncoding.merge(wireType, v, buf, ctx)
            }
            2u -> {
                val i = indices ?: TensorProto().also { indices = it }
                MessageEncoding.merge(wireType, i, buf, ctx)
            }
            3u -> Int64Encoding.decodeRepeated(wireType, dims, buf, ctx)
            else -> skipField(wireType, tag, buf, ctx)
        }
    }

    override fun encodedLen(): Int {
        var len = 0
        values?.let { len += MessageEncoding.encodedLen(1u, it) }
        indices?.let { len += MessageEncoding.encodedLen(2u, it) }
        if (dims.isNotEmpty()) len += Int64Encoding.encodedLenPacked(3u, dims)
        return len
    }

    override fun clear() {
        values = null
        indices = null
        dims.clear()
    }

    public companion object {
        public fun decode(bytes: ByteArray): SparseTensorProto {
            val msg = SparseTensorProto()
            msg.merge(Buf.wrap(bytes))
            return msg
        }
    }
}

public data class OperatorSetIdProto(
    var domain: String = "",
    var version: Long = 0L,
) : Message {
    override fun encodeRaw(buf: BufMut) {
        if (domain.isNotEmpty()) StringEncoding.encode(1u, domain, buf)
        if (version != 0L) Int64Encoding.encode(2u, version, buf)
    }

    override fun mergeField(
        tag: UInt,
        wireType: WireType,
        buf: Buf,
        ctx: DecodeContext,
    ) {
        when (tag) {
            1u -> domain = StringEncoding.decode(wireType, buf, ctx)
            2u -> version = Int64Encoding.decode(wireType, buf, ctx)
            else -> skipField(wireType, tag, buf, ctx)
        }
    }

    override fun encodedLen(): Int {
        var len = 0
        if (domain.isNotEmpty()) len += StringEncoding.encodedLen(1u, domain)
        if (version != 0L) len += Int64Encoding.encodedLen(2u, version)
        return len
    }

    override fun clear() {
        domain = ""
        version = 0L
    }

    public companion object {
        public fun decode(bytes: ByteArray): OperatorSetIdProto {
            val msg = OperatorSetIdProto()
            msg.merge(Buf.wrap(bytes))
            return msg
        }
    }
}
