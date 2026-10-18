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

public data class ModelProto(
    var irVersion: Long = 0L,
    var opsetImport: MutableList<OperatorSetIdProto> = mutableListOf(),
    var producerName: String = "",
    var producerVersion: String = "",
    var domain: String = "",
    var modelVersion: Long = 0L,
    var docString: String = "",
    var graph: GraphProto? = null,
    var metadataProps: MutableList<StringStringEntryProto> = mutableListOf(),
    var trainingInfo: MutableList<TrainingInfoProto> = mutableListOf(),
    var functions: MutableList<FunctionProto> = mutableListOf(),
) : Message {
    override fun encodeRaw(buf: BufMut) {
        if (irVersion != 0L) Int64Encoding.encode(1u, irVersion, buf)
        if (producerName.isNotEmpty()) StringEncoding.encode(2u, producerName, buf)
        if (producerVersion.isNotEmpty()) StringEncoding.encode(3u, producerVersion, buf)
        if (domain.isNotEmpty()) StringEncoding.encode(4u, domain, buf)
        if (modelVersion != 0L) Int64Encoding.encode(5u, modelVersion, buf)
        if (docString.isNotEmpty()) StringEncoding.encode(6u, docString, buf)
        graph?.let { MessageEncoding.encode(7u, it, buf) }
        if (opsetImport.isNotEmpty()) MessageEncoding.encodeRepeated(8u, opsetImport, buf)
        if (metadataProps.isNotEmpty()) MessageEncoding.encodeRepeated(14u, metadataProps, buf)
        if (trainingInfo.isNotEmpty()) MessageEncoding.encodeRepeated(20u, trainingInfo, buf)
        if (functions.isNotEmpty()) MessageEncoding.encodeRepeated(25u, functions, buf)
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
                    irVersion = res.getOrThrow()
                    Result.success(Unit)
                } else Result.failure(res.exceptionOrNull()!!)
            }
            2u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    producerName = res.getOrThrow()
                    Result.success(Unit)
                } else Result.failure(res.exceptionOrNull()!!)
            }
            3u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    producerVersion = res.getOrThrow()
                    Result.success(Unit)
                } else Result.failure(res.exceptionOrNull()!!)
            }
            4u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    domain = res.getOrThrow()
                    Result.success(Unit)
                } else Result.failure(res.exceptionOrNull()!!)
            }
            5u -> {
                val res = Int64Encoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    modelVersion = res.getOrThrow()
                    Result.success(Unit)
                } else Result.failure(res.exceptionOrNull()!!)
            }
            6u -> {
                val res = StringEncoding.merge(wireType, buf, ctx)
                if (res.isSuccess) {
                    docString = res.getOrThrow()
                    Result.success(Unit)
                } else Result.failure(res.exceptionOrNull()!!)
            }
            7u -> {
                val g = graph ?: GraphProto().also { graph = it }
                MessageEncoding.merge(wireType, g, buf, ctx)
            }
            8u -> {
                val op = OperatorSetIdProto()
                val res = MessageEncoding.merge(wireType, op, buf, ctx)
                if (res.isSuccess) {
                    opsetImport.add(op)
                    Result.success(Unit)
                } else res
            }
            14u -> {
                val prop = StringStringEntryProto()
                val res = MessageEncoding.merge(wireType, prop, buf, ctx)
                if (res.isSuccess) {
                    metadataProps.add(prop)
                    Result.success(Unit)
                } else res
            }
            20u -> {
                val ti = TrainingInfoProto()
                val res = MessageEncoding.merge(wireType, ti, buf, ctx)
                if (res.isSuccess) {
                    trainingInfo.add(ti)
                    Result.success(Unit)
                } else res
            }
            25u -> {
                val fn = FunctionProto()
                val res = MessageEncoding.merge(wireType, fn, buf, ctx)
                if (res.isSuccess) {
                    functions.add(fn)
                    Result.success(Unit)
                } else res
            }
            else -> skipField(wireType, tag, buf, ctx)
        }

    override fun encodedLen(): Int {
        var len = 0
        if (irVersion != 0L) len += Int64Encoding.encodedLen(1u, irVersion)
        if (producerName.isNotEmpty()) len += StringEncoding.encodedLen(2u, producerName)
        if (producerVersion.isNotEmpty()) len += StringEncoding.encodedLen(3u, producerVersion)
        if (domain.isNotEmpty()) len += StringEncoding.encodedLen(4u, domain)
        if (modelVersion != 0L) len += Int64Encoding.encodedLen(5u, modelVersion)
        if (docString.isNotEmpty()) len += StringEncoding.encodedLen(6u, docString)
        graph?.let { len += MessageEncoding.encodedLen(7u, it) }
        if (opsetImport.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(8u, opsetImport)
        if (metadataProps.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(14u, metadataProps)
        if (trainingInfo.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(20u, trainingInfo)
        if (functions.isNotEmpty()) len += MessageEncoding.encodedLenRepeated(25u, functions)
        return len
    }

    override fun clear() {
        irVersion = 0L
        opsetImport.clear()
        producerName = ""
        producerVersion = ""
        domain = ""
        modelVersion = 0L
        docString = ""
        graph = null
        metadataProps.clear()
        trainingInfo.clear()
        functions.clear()
    }

    public companion object {
        public fun decode(buf: Buf): Result<ModelProto> =
            io.github.kotlinmania.prost.decode(buf) { ModelProto() }

        public fun decode(bytes: ByteArray): Result<ModelProto> =
            io.github.kotlinmania.prost.decode(bytes) { ModelProto() }
    }
}
