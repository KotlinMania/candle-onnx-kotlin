package io.github.kotlinmania.candleonnx.onnx

import io.github.kotlinmania.bytes.buf.Buf
import io.github.kotlinmania.bytes.buf.BufMut
import io.github.kotlinmania.prost.Message
import io.github.kotlinmania.prost.encoding.BytesEncoding
import io.github.kotlinmania.prost.encoding.DecodeContext
import io.github.kotlinmania.prost.encoding.FloatEncoding
import io.github.kotlinmania.prost.encoding.Int32Encoding
import io.github.kotlinmania.prost.encoding.Int64Encoding
import io.github.kotlinmania.prost.encoding.MessageEncoding
import io.github.kotlinmania.prost.encoding.StringEncoding
import io.github.kotlinmania.prost.encoding.WireType
import io.github.kotlinmania.prost.encoding.skipField

public data class AttributeProto(
    var name: String = "",
    var refAttrName: String = "",
    var docString: String = "",
    var type: AttributeType = AttributeType.UNDEFINED,
    var f: Float = 0f,
    var i: Long = 0L,
    var s: ByteArray = ByteArray(0),
    var t: TensorProto? = null,
    var g: GraphProto? = null,
    var sparseTensor: SparseTensorProto? = null,
    var tp: TypeProto? = null,
    var floats: MutableList<Float> = mutableListOf(),
    var ints: MutableList<Long> = mutableListOf(),
    var strings: MutableList<ByteArray> = mutableListOf(),
    var tensors: MutableList<TensorProto> = mutableListOf(),
    var graphs: MutableList<GraphProto> = mutableListOf(),
    var sparseTensors: MutableList<SparseTensorProto> = mutableListOf(),
    var typeProtos: MutableList<TypeProto> = mutableListOf(),
) : Message {
    public fun type(): AttributeType = type

    public fun r#type(): AttributeType = type

    public enum class AttributeType(
        public val value: Int,
    ) {
        UNDEFINED(0),
        FLOAT(1),
        INT(2),
        STRING(3),
        TENSOR(4),
        GRAPH(5),
        FLOATS(6),
        INTS(7),
        STRINGS(8),
        TENSORS(9),
        GRAPHS(10),
        SPARSE_TENSOR(11),
        SPARSE_TENSORS(12),
        TYPE_PROTO(13),
        TYPE_PROTOS(14),
        ;

        public companion object {
            public fun fromValue(value: Int): AttributeType? =
                entries.firstOrNull { it.value == value }
        }
    }

    override fun encodeRaw(buf: BufMut) {
        if (name.isNotEmpty()) StringEncoding.encode(1u, name, buf)
        if (f != 0f) FloatEncoding.encode(2u, f, buf)
        if (i != 0L) Int64Encoding.encode(3u, i, buf)
        if (s.isNotEmpty()) BytesEncoding.encode(4u, s, buf)
        t?.let { MessageEncoding.encode(5u, it, buf) }
        g?.let { MessageEncoding.encode(6u, it, buf) }
        if (floats.isNotEmpty()) FloatEncoding.encodePacked(7u, floats, buf)
        if (ints.isNotEmpty()) Int64Encoding.encodePacked(8u, ints, buf)
        if (strings.isNotEmpty()) BytesEncoding.encodeRepeated(9u, strings, buf)
        if (tensors.isNotEmpty()) MessageEncoding.encodeRepeated(10u, tensors, buf)
        if (graphs.isNotEmpty()) MessageEncoding.encodeRepeated(11u, graphs, buf)
        if (docString.isNotEmpty()) StringEncoding.encode(13u, docString, buf)
        tp?.let { MessageEncoding.encode(14u, it, buf) }
        if (typeProtos.isNotEmpty()) MessageEncoding.encodeRepeated(15u, typeProtos, buf)
        if (type != AttributeType.UNDEFINED) Int32Encoding.encode(20u, type.value, buf)
        if (refAttrName.isNotEmpty()) StringEncoding.encode(21u, refAttrName, buf)
        sparseTensor?.let { MessageEncoding.encode(22u, it, buf) }
        if (sparseTensors.isNotEmpty()) MessageEncoding.encodeRepeated(23u, sparseTensors, buf)
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
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            2u -> {
                val res = FloatEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    f = res.getOrThrow()
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            3u -> {
                val res = Int64Encoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    i = res.getOrThrow()
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            4u -> {
                val res = BytesEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    s = res.getOrThrow()
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            5u -> {
                val target = t ?: TensorProto().also { t = it }
                MessageEncoding.merge(wireType, target, buf, ctx)
            }
            6u -> {
                val target = g ?: GraphProto().also { g = it }
                MessageEncoding.merge(wireType, target, buf, ctx)
            }
            7u -> FloatEncoding.mergeRepeated(wireType, floats, buf, ctx)
            8u -> Int64Encoding.mergeRepeated(wireType, ints, buf, ctx)
            9u -> BytesEncoding.mergeRepeated(wireType, strings, buf, ctx)
            10u -> MessageEncoding.mergeRepeated(wireType, tensors, { TensorProto() }, buf, ctx)
            11u -> MessageEncoding.mergeRepeated(wireType, graphs, { GraphProto() }, buf, ctx)
            13u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    docString = res.getOrThrow()
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            14u -> {
                val target = tp ?: TypeProto().also { tp = it }
                MessageEncoding.merge(wireType, target, buf, ctx)
            }
            15u -> MessageEncoding.mergeRepeated(wireType, typeProtos, { TypeProto() }, buf, ctx)
            20u -> {
                val res = Int32Encoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    type = AttributeType.fromValue(res.getOrThrow()) ?: AttributeType.UNDEFINED
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            21u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    refAttrName = res.getOrThrow()
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            22u -> {
                val target = sparseTensor ?: SparseTensorProto().also { sparseTensor = it }
                MessageEncoding.merge(wireType, target, buf, ctx)
            }
            23u -> MessageEncoding.mergeRepeated(wireType, sparseTensors, { SparseTensorProto() }, buf, ctx)
            else -> skipField(wireType, tag, buf, ctx)
        }

    override fun encodedLen(): Int {
        var len = 0
        if (name.isNotEmpty()) len += StringEncoding.encodedLen(1u, name)
        if (f != 0f) len += FloatEncoding.encodedLen(2u, f)
        if (i != 0L) len += Int64Encoding.encodedLen(3u, i)
        if (s.isNotEmpty()) len += BytesEncoding.encodedLen(4u, s)
        t?.let { len += MessageEncoding.encodedLen(5u, it) }
        g?.let { len += MessageEncoding.encodedLen(6u, it) }
        if (floats.isNotEmpty()) len += FloatEncoding.encodedLenPacked(7u, floats)
        if (ints.isNotEmpty()) len += Int64Encoding.encodedLenPacked(8u, ints)
        if (strings.isNotEmpty()) len += BytesEncoding.encodedLenRepeated(9u, strings)
        if (tensors.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(10u, tensors)
        if (graphs.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(11u, graphs)
        if (docString.isNotEmpty()) len += StringEncoding.encodedLen(13u, docString)
        tp?.let { len += MessageEncoding.encodedLen(14u, it) }
        if (typeProtos.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(15u, typeProtos)
        if (type != AttributeType.UNDEFINED) len += Int32Encoding.encodedLen(20u, type.value)
        if (refAttrName.isNotEmpty()) len += StringEncoding.encodedLen(21u, refAttrName)
        sparseTensor?.let { len += MessageEncoding.encodedLen(22u, it) }
        if (sparseTensors.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(23u, sparseTensors)
        return len
    }

    override fun clear() {
        name = ""
        refAttrName = ""
        docString = ""
        type = AttributeType.UNDEFINED
        f = 0f
        i = 0L
        s = ByteArray(0)
        t = null
        g = null
        sparseTensor = null
        tp = null
        floats.clear()
        ints.clear()
        strings.clear()
        tensors.clear()
        graphs.clear()
        sparseTensors.clear()
        typeProtos.clear()
    }
}
