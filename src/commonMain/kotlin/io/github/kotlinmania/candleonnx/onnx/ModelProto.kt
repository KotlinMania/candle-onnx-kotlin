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
    ) {
        when (tag) {
            1u -> irVersion = Int64Encoding.decode(wireType, buf, ctx)
            2u -> producerName = StringEncoding.decode(wireType, buf, ctx)
            3u -> producerVersion = StringEncoding.decode(wireType, buf, ctx)
            4u -> domain = StringEncoding.decode(wireType, buf, ctx)
            5u -> modelVersion = Int64Encoding.decode(wireType, buf, ctx)
            6u -> docString = StringEncoding.decode(wireType, buf, ctx)
            7u -> {
                val g = graph ?: GraphProto().also { graph = it }
                MessageEncoding.merge(wireType, g, buf, ctx)
            }
            8u -> opsetImport.add(MessageEncoding.decode(wireType, OperatorSetIdProto(), buf, ctx))
            14u -> metadataProps.add(MessageEncoding.decode(wireType, StringStringEntryProto(), buf, ctx))
            20u -> trainingInfo.add(MessageEncoding.decode(wireType, TrainingInfoProto(), buf, ctx))
            25u -> functions.add(MessageEncoding.decode(wireType, FunctionProto(), buf, ctx))
            else -> skipField(wireType, tag, buf, ctx)
        }
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
        public fun decode(bytes: ByteArray): ModelProto {
            val msg = ModelProto()
            msg.merge(Buf.wrap(bytes))
            return msg
        }
    }
}
