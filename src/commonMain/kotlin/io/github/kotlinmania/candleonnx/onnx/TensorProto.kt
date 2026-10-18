package io.github.kotlinmania.candleonnx.onnx

import io.github.kotlinmania.bytes.buf.Buf
import io.github.kotlinmania.bytes.buf.BufMut
import io.github.kotlinmania.prost.Message
import io.github.kotlinmania.prost.encoding.BytesEncoding
import io.github.kotlinmania.prost.encoding.DecodeContext
import io.github.kotlinmania.prost.encoding.DoubleEncoding
import io.github.kotlinmania.prost.encoding.FloatEncoding
import io.github.kotlinmania.prost.encoding.Int32Encoding
import io.github.kotlinmania.prost.encoding.Int64Encoding
import io.github.kotlinmania.prost.encoding.MessageEncoding
import io.github.kotlinmania.prost.encoding.StringEncoding
import io.github.kotlinmania.prost.encoding.UInt64Encoding
import io.github.kotlinmania.prost.encoding.WireType
import io.github.kotlinmania.prost.encoding.skipField

public data class TensorProto(
    var dims: MutableList<Long> = mutableListOf(),
    var dataType: Int = 0,
    var segment: Segment? = null,
    var floatData: MutableList<Float> = mutableListOf(),
    var int32Data: MutableList<Int> = mutableListOf(),
    var stringData: MutableList<ByteArray> = mutableListOf(),
    var int64Data: MutableList<Long> = mutableListOf(),
    var name: String = "",
    var docString: String = "",
    var rawData: ByteArray = ByteArray(0),
    var externalData: MutableList<StringStringEntryProto> = mutableListOf(),
    var dataLocation: DataLocation = DataLocation.DEFAULT,
    var doubleData: MutableList<Double> = mutableListOf(),
    var uint64Data: MutableList<ULong> = mutableListOf(),
) : Message {
    public enum class DataType(
        public val value: Int,
    ) {
        UNDEFINED(0),
        FLOAT(1),
        UINT8(2),
        INT8(3),
        UINT16(4),
        INT16(5),
        INT32(6),
        INT64(7),
        STRING(8),
        BOOL(9),
        FLOAT16(10),
        DOUBLE(11),
        UINT32(12),
        UINT64(13),
        COMPLEX64(14),
        COMPLEX128(15),
        BFLOAT16(16),
        FLOAT8E4M3FN(17),
        FLOAT8E4M3FNUZ(18),
        FLOAT8E5M2(19),
        FLOAT8E5M2FNUZ(20),
        ;

        public fun asStrName(): String = name

        public companion object {
            public fun fromI32(value: Int): DataType? = fromValue(value)

            public fun fromValue(value: Int): DataType? =
                entries.firstOrNull { it.value == value }

            public fun tryFrom(value: Int): Result<DataType> =
                fromValue(value)?.let { Result.success(it) }
                    ?: Result.failure(IllegalArgumentException("Unknown DataType value $value"))
        }
    }

    public enum class DataLocation(
        public val value: Int,
    ) {
        DEFAULT(0),
        EXTERNAL(1),
        ;

        public companion object {
            public fun fromI32(value: Int): DataLocation? = fromValue(value)

            public fun fromValue(value: Int): DataLocation? =
                entries.firstOrNull { it.value == value }
        }
    }

    public data class Segment(
        var begin: Long = 0L,
        var end: Long = 0L,
    ) : Message {
        override fun encodeRaw(buf: BufMut) {
            if (begin != 0L) Int64Encoding.encode(1u, begin, buf)
            if (end != 0L) Int64Encoding.encode(2u, end, buf)
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
                        begin = res.getOrThrow()
                        Result.success(Unit)
                    } else {
                        Result.failure(res.exceptionOrNull()!!)
                    }
                }
                2u -> {
                    val res = Int64Encoding.merge(wireType, buf, ctx)
                    if (res.isSuccess) {
                        end = res.getOrThrow()
                        Result.success(Unit)
                    } else {
                        Result.failure(res.exceptionOrNull()!!)
                    }
                }
                else -> skipField(wireType, tag, buf, ctx)
            }

        override fun encodedLen(): Int {
            var len = 0
            if (begin != 0L) len += Int64Encoding.encodedLen(1u, begin)
            if (end != 0L) len += Int64Encoding.encodedLen(2u, end)
            return len
        }

        override fun clear() {
            begin = 0L
            end = 0L
        }
    }

    override fun encodeRaw(buf: BufMut) {
        if (dims.isNotEmpty()) Int64Encoding.encodePacked(1u, dims, buf)
        if (dataType != 0) Int32Encoding.encode(2u, dataType, buf)
        segment?.let { MessageEncoding.encode(3u, it, buf) }
        if (floatData.isNotEmpty()) FloatEncoding.encodePacked(4u, floatData, buf)
        if (int32Data.isNotEmpty()) Int32Encoding.encodePacked(5u, int32Data, buf)
        if (stringData.isNotEmpty()) BytesEncoding.encodeRepeated(6u, stringData, buf)
        if (int64Data.isNotEmpty()) Int64Encoding.encodePacked(7u, int64Data, buf)
        if (name.isNotEmpty()) StringEncoding.encode(8u, name, buf)
        if (rawData.isNotEmpty()) BytesEncoding.encode(9u, rawData, buf)
        if (doubleData.isNotEmpty()) DoubleEncoding.encodePacked(10u, doubleData, buf)
        if (uint64Data.isNotEmpty()) UInt64Encoding.encodePacked(11u, uint64Data, buf)
        if (docString.isNotEmpty()) StringEncoding.encode(12u, docString, buf)
        if (externalData.isNotEmpty()) MessageEncoding.encodeRepeated(13u, externalData, buf)
        if (dataLocation != DataLocation.DEFAULT) Int32Encoding.encode(14u, dataLocation.value, buf)
    }

    override fun mergeField(
        tag: UInt,
        wireType: WireType,
        buf: Buf,
        ctx: DecodeContext,
    ): Result<Unit> =
        when (tag) {
            1u -> Int64Encoding.mergeRepeated(wireType, dims, buf, ctx)
            2u -> {
                val res = Int32Encoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    dataType = res.getOrThrow()
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            3u -> {
                val target = segment ?: Segment().also { segment = it }
                MessageEncoding.merge(wireType, target, buf, ctx)
            }
            4u -> FloatEncoding.mergeRepeated(wireType, floatData, buf, ctx)
            5u -> Int32Encoding.mergeRepeated(wireType, int32Data, buf, ctx)
            6u -> BytesEncoding.mergeRepeated(wireType, stringData, buf, ctx)
            7u -> Int64Encoding.mergeRepeated(wireType, int64Data, buf, ctx)
            8u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    name = res.getOrThrow()
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            9u -> {
                val res = BytesEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    rawData = res.getOrThrow()
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            10u -> DoubleEncoding.mergeRepeated(wireType, doubleData, buf, ctx)
            11u -> UInt64Encoding.mergeRepeated(wireType, uint64Data, buf, ctx)
            12u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    docString = res.getOrThrow()
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            13u -> MessageEncoding.mergeRepeated(wireType, externalData, { StringStringEntryProto() }, buf, ctx)
            14u -> {
                val res = Int32Encoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    dataLocation = DataLocation.fromValue(res.getOrThrow()) ?: DataLocation.DEFAULT
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            else -> skipField(wireType, tag, buf, ctx)
        }

    override fun encodedLen(): Int {
        var len = 0
        if (dims.isNotEmpty()) len += Int64Encoding.encodedLenPacked(1u, dims)
        if (dataType != 0) len += Int32Encoding.encodedLen(2u, dataType)
        segment?.let { len += MessageEncoding.encodedLen(3u, it) }
        if (floatData.isNotEmpty()) len += FloatEncoding.encodedLenPacked(4u, floatData)
        if (int32Data.isNotEmpty()) len += Int32Encoding.encodedLenPacked(5u, int32Data)
        if (stringData.isNotEmpty()) len += BytesEncoding.encodedLenRepeated(6u, stringData)
        if (int64Data.isNotEmpty()) len += Int64Encoding.encodedLenPacked(7u, int64Data)
        if (name.isNotEmpty()) len += StringEncoding.encodedLen(8u, name)
        if (rawData.isNotEmpty()) len += BytesEncoding.encodedLen(9u, rawData)
        if (doubleData.isNotEmpty()) len += DoubleEncoding.encodedLenPacked(10u, doubleData)
        if (uint64Data.isNotEmpty()) len += UInt64Encoding.encodedLenPacked(11u, uint64Data)
        if (docString.isNotEmpty()) len += StringEncoding.encodedLen(12u, docString)
        if (externalData.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(13u, externalData)
        if (dataLocation != DataLocation.DEFAULT) len += Int32Encoding.encodedLen(14u, dataLocation.value)
        return len
    }

    override fun clear() {
        dims.clear()
        dataType = 0
        segment = null
        floatData.clear()
        int32Data.clear()
        stringData.clear()
        int64Data.clear()
        name = ""
        docString = ""
        rawData = ByteArray(0)
        externalData.clear()
        dataLocation = DataLocation.DEFAULT
        doubleData.clear()
        uint64Data.clear()
    }
}
