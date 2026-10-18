package io.github.kotlinmania.candleonnx.onnx

import io.github.kotlinmania.bytes.buf.Buf
import io.github.kotlinmania.bytes.buf.BufMut
import io.github.kotlinmania.prost.Message
import io.github.kotlinmania.prost.encoding.DecodeContext
import io.github.kotlinmania.prost.encoding.Int32Encoding
import io.github.kotlinmania.prost.encoding.MessageEncoding
import io.github.kotlinmania.prost.encoding.StringEncoding
import io.github.kotlinmania.prost.encoding.WireType
import io.github.kotlinmania.prost.encoding.skipField

public data class TypeProto(
    var value: Value? = null,
    var denotation: String = "",
) : Message {
    public sealed interface Value {
        public data class TensorType(
            val value: Tensor,
        ) : Value

        public data class SequenceType(
            val value: Sequence,
        ) : Value

        public data class MapType(
            val value: Map,
        ) : Value

        public data class OptionalType(
            val value: Optional,
        ) : Value

        public data class SparseTensorType(
            val value: SparseTensor,
        ) : Value
    }

    public data class Tensor(
        var elemType: Int = 0,
        var shape: TensorShapeProto? = null,
    ) : Message {
        override fun encodeRaw(buf: BufMut) {
            if (elemType != 0) Int32Encoding.encode(1u, elemType, buf)
            shape?.let { MessageEncoding.encode(2u, it, buf) }
        }

        override fun mergeField(
            tag: UInt,
            wireType: WireType,
            buf: Buf,
            ctx: DecodeContext,
        ): Result<Unit> =
            when (tag) {
                1u -> {
                    val res = Int32Encoding.merge(wireType, buf, ctx)
                    if (res.isSuccess) {
                        elemType = res.getOrThrow()
                        Result.success(Unit)
                    } else {
                        Result.failure(res.exceptionOrNull()!!)
                    }
                }
                2u -> {
                    val target = shape ?: TensorShapeProto().also { shape = it }
                    MessageEncoding.merge(wireType, target, buf, ctx)
                }
                else -> skipField(wireType, tag, buf, ctx)
            }

        override fun encodedLen(): Int {
            var len = 0
            if (elemType != 0) len += Int32Encoding.encodedLen(1u, elemType)
            shape?.let { len += MessageEncoding.encodedLen(2u, it) }
            return len
        }

        override fun clear() {
            elemType = 0
            shape = null
        }
    }

    public data class Sequence(
        var elemType: TypeProto? = null,
    ) : Message {
        override fun encodeRaw(buf: BufMut) {
            elemType?.let { MessageEncoding.encode(1u, it, buf) }
        }

        override fun mergeField(
            tag: UInt,
            wireType: WireType,
            buf: Buf,
            ctx: DecodeContext,
        ): Result<Unit> =
            when (tag) {
                1u -> {
                    val target = elemType ?: TypeProto().also { elemType = it }
                    MessageEncoding.merge(wireType, target, buf, ctx)
                }
                else -> skipField(wireType, tag, buf, ctx)
            }

        override fun encodedLen(): Int =
            elemType?.let { MessageEncoding.encodedLen(1u, it) } ?: 0

        override fun clear() {
            elemType = null
        }
    }

    public data class Map(
        var keyType: Int = 0,
        var valueType: TypeProto? = null,
    ) : Message {
        override fun encodeRaw(buf: BufMut) {
            if (keyType != 0) Int32Encoding.encode(1u, keyType, buf)
            valueType?.let { MessageEncoding.encode(2u, it, buf) }
        }

        override fun mergeField(
            tag: UInt,
            wireType: WireType,
            buf: Buf,
            ctx: DecodeContext,
        ): Result<Unit> =
            when (tag) {
                1u -> {
                    val res = Int32Encoding.merge(wireType, buf, ctx)
                    if (res.isSuccess) {
                        keyType = res.getOrThrow()
                        Result.success(Unit)
                    } else {
                        Result.failure(res.exceptionOrNull()!!)
                    }
                }
                2u -> {
                    val target = valueType ?: TypeProto().also { valueType = it }
                    MessageEncoding.merge(wireType, target, buf, ctx)
                }
                else -> skipField(wireType, tag, buf, ctx)
            }

        override fun encodedLen(): Int {
            var len = 0
            if (keyType != 0) len += Int32Encoding.encodedLen(1u, keyType)
            valueType?.let { len += MessageEncoding.encodedLen(2u, it) }
            return len
        }

        override fun clear() {
            keyType = 0
            valueType = null
        }
    }

    public data class Optional(
        var elemType: TypeProto? = null,
    ) : Message {
        override fun encodeRaw(buf: BufMut) {
            elemType?.let { MessageEncoding.encode(1u, it, buf) }
        }

        override fun mergeField(
            tag: UInt,
            wireType: WireType,
            buf: Buf,
            ctx: DecodeContext,
        ): Result<Unit> =
            when (tag) {
                1u -> {
                    val target = elemType ?: TypeProto().also { elemType = it }
                    MessageEncoding.merge(wireType, target, buf, ctx)
                }
                else -> skipField(wireType, tag, buf, ctx)
            }

        override fun encodedLen(): Int =
            elemType?.let { MessageEncoding.encodedLen(1u, it) } ?: 0

        override fun clear() {
            elemType = null
        }
    }

    public data class SparseTensor(
        var elemType: Int = 0,
        var shape: TensorShapeProto? = null,
    ) : Message {
        override fun encodeRaw(buf: BufMut) {
            if (elemType != 0) Int32Encoding.encode(1u, elemType, buf)
            shape?.let { MessageEncoding.encode(2u, it, buf) }
        }

        override fun mergeField(
            tag: UInt,
            wireType: WireType,
            buf: Buf,
            ctx: DecodeContext,
        ): Result<Unit> =
            when (tag) {
                1u -> {
                    val res = Int32Encoding.merge(wireType, buf, ctx)
                    if (res.isSuccess) {
                        elemType = res.getOrThrow()
                        Result.success(Unit)
                    } else {
                        Result.failure(res.exceptionOrNull()!!)
                    }
                }
                2u -> {
                    val target = shape ?: TensorShapeProto().also { shape = it }
                    MessageEncoding.merge(wireType, target, buf, ctx)
                }
                else -> skipField(wireType, tag, buf, ctx)
            }

        override fun encodedLen(): Int {
            var len = 0
            if (elemType != 0) len += Int32Encoding.encodedLen(1u, elemType)
            shape?.let { len += MessageEncoding.encodedLen(2u, it) }
            return len
        }

        override fun clear() {
            elemType = 0
            shape = null
        }
    }

    override fun encodeRaw(buf: BufMut) {
        when (val v = value) {
            is Value.TensorType -> MessageEncoding.encode(1u, v.value, buf)
            is Value.SequenceType -> MessageEncoding.encode(4u, v.value, buf)
            is Value.MapType -> MessageEncoding.encode(5u, v.value, buf)
            is Value.OptionalType -> MessageEncoding.encode(9u, v.value, buf)
            is Value.SparseTensorType -> MessageEncoding.encode(8u, v.value, buf)
            null -> {}
        }
        if (denotation.isNotEmpty()) StringEncoding.encode(6u, denotation, buf)
    }

    override fun mergeField(
        tag: UInt,
        wireType: WireType,
        buf: Buf,
        ctx: DecodeContext,
    ): Result<Unit> =
        when (tag) {
            1u -> {
                val current = (value as? Value.TensorType)?.value ?: Tensor()
                value = Value.TensorType(current)
                MessageEncoding.merge(wireType, current, buf, ctx)
            }
            4u -> {
                val current = (value as? Value.SequenceType)?.value ?: Sequence()
                value = Value.SequenceType(current)
                MessageEncoding.merge(wireType, current, buf, ctx)
            }
            5u -> {
                val current = (value as? Value.MapType)?.value ?: Map()
                value = Value.MapType(current)
                MessageEncoding.merge(wireType, current, buf, ctx)
            }
            6u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    denotation = res.getOrThrow()
                    Result.success(Unit)
                } else {
                    Result.failure(res.exceptionOrNull()!!)
                }
            }
            8u -> {
                val current = (value as? Value.SparseTensorType)?.value ?: SparseTensor()
                value = Value.SparseTensorType(current)
                MessageEncoding.merge(wireType, current, buf, ctx)
            }
            9u -> {
                val current = (value as? Value.OptionalType)?.value ?: Optional()
                value = Value.OptionalType(current)
                MessageEncoding.merge(wireType, current, buf, ctx)
            }
            else -> skipField(wireType, tag, buf, ctx)
        }

    override fun encodedLen(): Int {
        var len = 0
        when (val v = value) {
            is Value.TensorType -> len += MessageEncoding.encodedLen(1u, v.value)
            is Value.SequenceType -> len += MessageEncoding.encodedLen(4u, v.value)
            is Value.MapType -> len += MessageEncoding.encodedLen(5u, v.value)
            is Value.OptionalType -> len += MessageEncoding.encodedLen(9u, v.value)
            is Value.SparseTensorType -> len += MessageEncoding.encodedLen(8u, v.value)
            null -> {}
        }
        if (denotation.isNotEmpty()) len += StringEncoding.encodedLen(6u, denotation)
        return len
    }

    override fun clear() {
        value = null
        denotation = ""
    }
}
