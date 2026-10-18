// port-lint: source eval.rs
package io.github.kotlinmania.candleonnx

import io.github.kotlinmania.candle.CandleException
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.Var
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cat
import io.github.kotlinmania.candle.conv1d
import io.github.kotlinmania.candle.conv2d
import io.github.kotlinmania.candle.fromRawBuffer
import io.github.kotlinmania.candlenn.LSTMConfig
import io.github.kotlinmania.candlenn.LSTMState
import io.github.kotlinmania.candlenn.PReLU
import io.github.kotlinmania.candlenn.VarBuilder
import io.github.kotlinmania.candlenn.VarMap
import io.github.kotlinmania.candlenn.lstm
import io.github.kotlinmania.candlenn.hardSigmoid
import io.github.kotlinmania.candlenn.leakyRelu
import io.github.kotlinmania.candlenn.logSoftmax
import io.github.kotlinmania.candlenn.selu
import io.github.kotlinmania.candlenn.sigmoid
import io.github.kotlinmania.candlenn.softmax
import io.github.kotlinmania.candlenn.softmaxLastDim
import io.github.kotlinmania.candleonnx.onnx.AttributeProto
import io.github.kotlinmania.candleonnx.onnx.GraphProto
import io.github.kotlinmania.candleonnx.onnx.ModelProto
import io.github.kotlinmania.candleonnx.onnx.NodeProto
import io.github.kotlinmania.candleonnx.onnx.TensorProto
import io.github.kotlinmania.candleonnx.onnx.TensorShapeProto
import io.github.kotlinmania.candleonnx.onnx.TypeProto

public typealias Value = Tensor

public fun dtype(dt: TensorProto.DataType): DType? =
    when (dt) {
        TensorProto.DataType.UINT8 -> DType.U8
        TensorProto.DataType.UINT32 -> DType.U32
        TensorProto.DataType.INT64 -> DType.I64
        TensorProto.DataType.FLOAT16 -> DType.F16
        TensorProto.DataType.FLOAT -> DType.F32
        TensorProto.DataType.DOUBLE -> DType.F64
        TensorProto.DataType.BOOL -> DType.U8
        else -> null
    }

public interface Attr<T> {
    public val type: AttributeProto.AttributeType
    public fun get(attr: AttributeProto): T
}

public interface AttrOwned<T> {
    public val type: AttributeProto.AttributeType
    public fun get(attr: AttributeProto): T
}

public object LongAttr : Attr<Long> {
    override val type: AttributeProto.AttributeType = AttributeProto.AttributeType.INT
    override fun get(attr: AttributeProto): Long = attr.i
}

public object FloatAttr : Attr<Float> {
    override val type: AttributeProto.AttributeType = AttributeProto.AttributeType.FLOAT
    override fun get(attr: AttributeProto): Float = attr.f
}

public object LongListAttr : Attr<List<Long>> {
    override val type: AttributeProto.AttributeType = AttributeProto.AttributeType.INTS
    override fun get(attr: AttributeProto): List<Long> = attr.ints
}

public object StringAttr : Attr<String> {
    override val type: AttributeProto.AttributeType = AttributeProto.AttributeType.STRING
    override fun get(attr: AttributeProto): String = attr.s.decodeToString()
}

public object GraphProtoAttr : Attr<GraphProto> {
    override val type: AttributeProto.AttributeType = AttributeProto.AttributeType.GRAPH
    override fun get(attr: AttributeProto): GraphProto =
        attr.g ?: bail("attribute does not contain graph")
}

public object StringListAttrOwned : AttrOwned<List<String>> {
    override val type: AttributeProto.AttributeType = AttributeProto.AttributeType.STRINGS
    override fun get(attr: AttributeProto): List<String> =
        attr.strings.map { it.decodeToString() }
}

public object TensorAttrOwned : AttrOwned<Tensor> {
    override val type: AttributeProto.AttributeType = AttributeProto.AttributeType.TENSOR
    override fun get(attr: AttributeProto): Tensor {
        val tensorProto = attr.t
            ?: bail("attribute ${attr.name} was of type TENSOR, but no tensor was found")
        val dataType = TensorProto.DataType.fromI32(tensorProto.dataType)
            ?: bail("attribute ${attr.name} of type TENSOR was an invalid data_type number ${tensorProto.dataType}")
        val dt = dtype(dataType)
            ?: bail("attribute ${attr.name} of type TENSOR has an unsupported data_type ${dataType.name}")
        val dims = ArrayList<Int>(tensorProto.dims.size)
        for (dim in tensorProto.dims) {
            if (dim < 0) {
                bail("attribute ${attr.name} of type TENSOR has a negative dimension, which is unsupported")
            }
            dims.add(dim.toInt())
        }
        return fromRawBuffer(tensorProto.rawData, dt, dims.toIntArray(), Device.Cpu).unwrap()
    }
}

public fun getAttr_(node: NodeProto, name: String): AttributeProto {
    return node.attribute.firstOrNull { it.name == name }
        ?: bail("cannot find the '$name' attribute in '${node.opType}' for ${node.name}")
}

public inline fun <reified T> getAttr(node: NodeProto, name: String): T {
    val attr = getAttr_(node, name)
    return when (T::class) {
        Long::class -> {
            if (attr.type() != AttributeProto.AttributeType.INT) {
                bail("unsupported type ${attr.type()} for '$name' attribute in '${node.opType}' for ${node.name}")
            }
            attr.i as T
        }
        Float::class -> {
            if (attr.type() != AttributeProto.AttributeType.FLOAT) {
                bail("unsupported type ${attr.type()} for '$name' attribute in '${node.opType}' for ${node.name}")
            }
            attr.f as T
        }
        String::class -> {
            if (attr.type() != AttributeProto.AttributeType.STRING) {
                bail("unsupported type ${attr.type()} for '$name' attribute in '${node.opType}' for ${node.name}")
            }
            attr.s.decodeToString() as T
        }
        GraphProto::class -> {
            if (attr.type() != AttributeProto.AttributeType.GRAPH) {
                bail("unsupported type ${attr.type()} for '$name' attribute in '${node.opType}' for ${node.name}")
            }
            (attr.g ?: bail("attribute does not contain graph")) as T
        }
        List::class -> {
            if (attr.type() == AttributeProto.AttributeType.INTS) {
                attr.ints as T
            } else if (attr.type() == AttributeProto.AttributeType.STRINGS) {
                attr.strings.map { it.decodeToString() } as T
            } else {
                bail("unsupported type ${attr.type()} for '$name' attribute in '${node.opType}' for ${node.name}")
            }
        }
        Tensor::class -> {
            if (attr.type() != AttributeProto.AttributeType.TENSOR) {
                bail("unsupported type ${attr.type()} for '$name' attribute in '${node.opType}' for ${node.name}")
            }
            TensorAttrOwned.get(attr) as T
        }
        else -> bail("unsupported type parameter for getAttr")
    }
}

public inline fun <reified T> getAttrOpt(node: NodeProto, name: String): T? {
    val attr = node.attribute.firstOrNull { it.name == name } ?: return null
    return when (T::class) {
        Long::class -> {
            if (attr.type() != AttributeProto.AttributeType.INT) {
                bail("unsupported type ${attr.type()} for '$name' attribute in '${node.opType}' for ${node.name}")
            }
            attr.i as T
        }
        Float::class -> {
            if (attr.type() != AttributeProto.AttributeType.FLOAT) {
                bail("unsupported type ${attr.type()} for '$name' attribute in '${node.opType}' for ${node.name}")
            }
            attr.f as T
        }
        String::class -> {
            if (attr.type() != AttributeProto.AttributeType.STRING) {
                bail("unsupported type ${attr.type()} for '$name' attribute in '${node.opType}' for ${node.name}")
            }
            attr.s.decodeToString() as T
        }
        GraphProto::class -> {
            if (attr.type() != AttributeProto.AttributeType.GRAPH) {
                bail("unsupported type ${attr.type()} for '$name' attribute in '${node.opType}' for ${node.name}")
            }
            (attr.g ?: bail("attribute does not contain graph")) as T
        }
        List::class -> {
            if (attr.type() == AttributeProto.AttributeType.INTS) {
                attr.ints as T
            } else if (attr.type() == AttributeProto.AttributeType.STRINGS) {
                attr.strings.map { it.decodeToString() } as T
            } else {
                bail("unsupported type ${attr.type()} for '$name' attribute in '${node.opType}' for ${node.name}")
            }
        }
        Tensor::class -> {
            if (attr.type() != AttributeProto.AttributeType.TENSOR) {
                bail("unsupported type ${attr.type()} for '$name' attribute in '${node.opType}' for ${node.name}")
            }
            TensorAttrOwned.get(attr) as T
        }
        else -> bail("unsupported type parameter for getAttrOpt")
    }
}

public inline fun <reified T> getAttrOptOwned(node: NodeProto, name: String): T? = getAttrOpt<T>(node, name)

public fun getTensor(t: TensorProto, name: String): Tensor {
    val dims = t.dims.map { it.toInt() }
    val dataType = TensorProto.DataType.fromI32(t.dataType)
        ?: bail("unsupported 'value' data-type ${t.dataType} for $name")
    return when (dataType) {
        TensorProto.DataType.INT32 -> {
            if (t.int32Data.isEmpty()) {
                val len = t.rawData.size / 4
                val data = LongArray(len)
                for (idx in 0 until len) {
                    val b0 = t.rawData[idx * 4].toLong() and 0xFF
                    val b1 = t.rawData[idx * 4 + 1].toLong() and 0xFF
                    val b2 = t.rawData[idx * 4 + 2].toLong() and 0xFF
                    val b3 = t.rawData[idx * 4 + 3].toLong()
                    val v = (b3 shl 24) or (b2 shl 16) or (b1 shl 8) or b0
                    data[idx] = v.toInt().toLong()
                }
                Tensor.fromSlice(data, Shape(len), Device.Cpu)
            } else {
                val data = t.int32Data.map { it.toLong() }.toLongArray()
                Tensor.fromSlice(data, Shape(t.int32Data.size), Device.Cpu)
            }
        }
        else -> {
            val dt = dtype(dataType)
                ?: bail("unsupported 'value' data-type $dataType for $name")
            if (dt == DType.F32 && t.floatData.isNotEmpty()) {
                Tensor.fromSlice(t.floatData.toFloatArray(), Shape(dims), Device.Cpu)
            } else if (dt == DType.F64 && t.doubleData.isNotEmpty()) {
                Tensor.fromSlice(t.doubleData.toDoubleArray(), Shape(dims), Device.Cpu)
            } else if (dt == DType.I64 && t.int64Data.isNotEmpty()) {
                Tensor.fromSlice(t.int64Data.toLongArray(), Shape(dims), Device.Cpu)
            } else {
                fromRawBuffer(t.rawData, dt, dims.toIntArray(), Device.Cpu).unwrap()
            }
        }
    }
}

// This function provides a direct evaluation of the proto.
// Longer-term, we should first convert the proto to an intermediate representation of the compute
// graph so as to make multiple evaluations more efficient.
// An example upside of this would be to remove intermediary values when they are not needed
// anymore.
public fun simpleEval(
    model: ModelProto,
    inputs: MutableMap<String, Value>,
): MutableMap<String, Value> {
    val graph = model.graph ?: bail("no graph defined in proto")
    return simpleEval_(graph, inputs)
}

public fun simpleEval_(
    graph: GraphProto,
    values: MutableMap<String, Value>,
): MutableMap<String, Value> {
    for (t in graph.initializer) {
        val tensor = getTensor(t, t.name)
        values[t.name] = tensor
    }
    for (input in graph.input) {
        val inputType = input.type ?: continue
        val tensorType = (inputType.value as? TypeProto.Value.TensorType)?.value ?: continue
        val tensor = values[input.name] ?: bail("missing input ${input.name}")
        val dtEnum = TensorProto.DataType.fromI32(tensorType.elemType)
            ?: bail("unsupported input type ${tensorType.elemType}")
        val dt = dtype(dtEnum)
            ?: bail("unsupported 'value' data-type $dtEnum for ${input.name}")
        val shape = tensorType.shape
        if (shape != null) {
            if (shape.dim.size != tensor.rank()) {
                bail("unexpected rank for ${input.name}, got ${shape.dim}, expected ${tensor.shape()}")
            }
            for ((idx, pair) in shape.dim.zip(tensor.dims()).withIndex()) {
                val (d, dim) = pair
                val dimVal = d.value
                if (dimVal is TensorShapeProto.Dimension.Value.DimValue) {
                    if (dimVal.value.toInt() != dim) {
                        bail("unexpected dim $idx for ${input.name}, got ${shape.dim}, expected ${tensor.shape()}")
                    }
                }
            }
        }
        if (dt != tensor.dtype()) {
            bail("unexpected dtype for ${input.name}, got ${tensor.dtype()}, expected $dt")
        }
    }

    // The nodes are topologically sorted so we can just process them in order.
    for (node in graph.node) {
        val get = { inputName: String ->
            values[inputName] ?: bail("cannot find $inputName for op '${node.name}'")
        }
        val getOpt = { i: Int ->
            node.input.getOrNull(i)?.takeIf { it.isNotEmpty() }?.let { get(it) }
        }

        when (node.opType) {
            "Add" -> {
                val input0 = get(node.input[0])
                val input1 = get(node.input[1])
                val output = input0.broadcastAdd(input1)
                values[node.output[0]] = output
            }
            "Sub" -> {
                val input0 = get(node.input[0])
                val input1 = get(node.input[1])
                val output = input0.broadcastSub(input1)
                values[node.output[0]] = output
            }
            "Mul" -> {
                val input0 = get(node.input[0])
                val input1 = get(node.input[1])
                val output = input0.broadcastMul(input1)
                values[node.output[0]] = output
            }
            "Div" -> {
                val input0 = get(node.input[0])
                val input1 = get(node.input[1])
                val output = input0.broadcastDiv(input1)
                values[node.output[0]] = output
            }
            "Pow" -> {
                val input0 = get(node.input[0])
                val input1 = get(node.input[1])
                val exp = try {
                    toScalarFlexible<Double>(input1.toDtype(DType.F64))
                } catch (_: Exception) {
                    null
                }
                val output = if (exp != null) {
                    input0.powf(exp)
                } else {
                    input0.broadcastPow(input1)
                }
                values[node.output[0]] = output
            }
            "Exp" -> {
                val xs = get(node.input[0])
                val output = xs.exp()
                values[node.output[0]] = output
            }
            "Equal" -> {
                val input0 = get(node.input[0])
                val input1 = get(node.input[1])
                val output = input0.broadcastEq(input1)
                values[node.output[0]] = output
            }
            "Not" -> {
                val xs = get(node.input[0])
                val output = xs.eq(xs.zerosLike())
                values[node.output[0]] = output
            }
            "MatMul" -> {
                val input0 = get(node.input[0])
                val input1 = get(node.input[1])
                val output = input0.broadcastMatmul(input1)
                values[node.output[0]] = output
            }
            "Reshape" -> {
                val input0 = get(node.input[0])
                val input1 = get(node.input[1]).toVec1<Long>()
                val allowzero = getAttrOpt<Long>(node, "allowzero") == 1L
                if (input1.count { it == -1L } > 1) {
                    bail("Reshape: at most one dimension of the target shape can be -1")
                }
                val resolved = ArrayList<Int?>(input1.size)
                for ((idx, v) in input1.withIndex()) {
                    resolved.add(
                        when {
                            v == -1L -> null
                            v == 0L && allowzero -> 0
                            v == 0L -> input0.dim(idx)
                            v > 0L -> v.toInt()
                            else -> bail("Reshape: invalid dimension $v in target shape")
                        },
                    )
                }
                var known = 1
                for (d in resolved) {
                    if (d != null) known *= d
                }
                val targetDims = ArrayList<Int>(resolved.size)
                for (d in resolved) {
                    if (d != null) {
                        targetDims.add(d)
                    } else {
                        if (known == 0) {
                            bail("Reshape: -1 cannot be inferred when another dimension is zero")
                        }
                        targetDims.add(input0.elemCount() / known)
                    }
                }
                val output = input0.reshape(Shape(targetDims))
                values[node.output[0]] = output
            }
            "LogSoftmax" -> {
                val input = get(node.input[0])
                val axisAttr = getAttrOpt<Long>(node, "axis")
                val output = if (axisAttr == null) {
                    logSoftmax(input, input.rank() - 1)
                } else {
                    val axis = input.normalizeAxis(axisAttr)
                    logSoftmax(input, axis)
                }
                values[node.output[0]] = output
            }
            "Softmax" -> {
                val input = get(node.input[0])
                val axisAttr = getAttrOpt<Long>(node, "axis")
                val output = if (axisAttr == null) {
                    softmaxLastDim(input)
                } else {
                    val axis = input.normalizeAxis(axisAttr)
                    softmax(input, axis)
                }
                values[node.output[0]] = output
            }
            "Transpose" -> {
                val input = get(node.input[0])
                val permAttr = getAttrOpt<List<Long>>(node, "perm")
                val output = if (permAttr == null) {
                    input.t()
                } else {
                    val perm = permAttr.map { it.toInt() }
                    input.permute(perm).contiguous()
                }
                values[node.output[0]] = output
            }
            "Dropout" -> {
                val input = get(node.input[0])
                values[node.output[0]] = input
            }
            "MaxPool" -> {
                val dilations = getAttrOpt<List<Long>>(node, "dilations")
                val kernelShape = getAttr<List<Long>>(node, "kernel_shape")
                val pads = getAttrOpt<List<Long>>(node, "pads")
                val strides = getAttrOpt<List<Long>>(node, "strides")
                val autoPad = getAttrOpt<String>(node, "auto_pad")
                when (autoPad) {
                    null, "NOTSET" -> Unit
                    else -> bail("unsupported auto_pad $autoPad")
                }
                if (dilations != null && dilations.any { it != 1L }) {
                    bail("MaxPool with dilation != 1, $dilations")
                }
                if (pads != null && pads.any { it != 0L }) {
                    bail("MaxPool with pads != 0, $pads")
                }
                val xs = get(node.input[0])
                if (kernelShape.size != 2) {
                    bail("only 2d MaxPool is supported, kernel shape $kernelShape")
                }
                val k1 = kernelShape[0].toInt()
                val k2 = kernelShape[1].toInt()
                val ys = if (strides == null) {
                    xs.maxPool2d(Pair(k1, k2))
                } else if (strides.size == 2) {
                    xs.maxPool2dWithStride(Pair(k1, k2), Pair(strides[0].toInt(), strides[1].toInt()))
                } else {
                    bail("only 2d MaxPool is supported, strides $strides")
                }
                values[node.output[0]] = ys
            }
            "AveragePool" -> {
                val dilations = getAttrOpt<List<Long>>(node, "dilations")
                val kernelShape = getAttr<List<Long>>(node, "kernel_shape")
                val pads = getAttrOpt<List<Long>>(node, "pads")
                val strides = getAttrOpt<List<Long>>(node, "strides")
                val autoPad = getAttrOpt<String>(node, "auto_pad")
                when (autoPad) {
                    null, "NOTSET" -> Unit
                    else -> bail("unsupported auto_pad $autoPad")
                }
                if (dilations != null && dilations.any { it != 1L }) {
                    bail("AvgPool with dilation != 1, $dilations")
                }
                if (pads != null && pads.any { it != 0L }) {
                    bail("AvgPool with pads != 0, $pads")
                }
                val xs = get(node.input[0])
                if (kernelShape.size != 2) {
                    bail("only 2d AvgPool is supported, kernel shape $kernelShape")
                }
                val k1 = kernelShape[0].toInt()
                val k2 = kernelShape[1].toInt()
                val ys = if (strides == null) {
                    xs.avgPool2d(Pair(k1, k2))
                } else if (strides.size == 2) {
                    xs.avgPool2dWithStride(Pair(k1, k2), Pair(strides[0].toInt(), strides[1].toInt()))
                } else {
                    bail("only 2d AvgPool is supported, strides $strides")
                }
                values[node.output[0]] = ys
            }
            "BatchNormalization" -> {
                val trainingMode = getAttrOpt<Long>(node, "training_mode") ?: 0L
                if (trainingMode != 0L) {
                    bail("training mode is not supported for BatchNorm")
                }
                val eps = getAttrOpt<Float>(node, "epsilon") ?: 1e-5f
                val xs = get(node.input[0])
                val weight = get(node.input[1])
                val bias = get(node.input[2])
                val runningMean = get(node.input[3])
                val runningVar = get(node.input[4])
                val targetShape = xs.dims().mapIndexed { idx, v -> if (idx == 1) v else 1 }
                val targetShapeObj = Shape(targetShape)
                val normalized = xs
                    .broadcastSub(runningMean.reshape(targetShapeObj))
                    .broadcastDiv(runningVar.reshape(targetShapeObj).affine(1.0, eps.toDouble()).sqrt())
                val w = weight.reshape(targetShapeObj)
                val b = bias.reshape(targetShapeObj)
                val out = normalized.broadcastMul(w).broadcastAdd(b)
                values[node.output[0]] = out
            }
            "Squeeze" -> {
                val xs = get(node.input[0])
                val axes = if (node.input.size <= 1) {
                    xs.dims().mapIndexedNotNull { idx, s -> if (s == 1 && idx > 0) idx else null }.toMutableList()
                } else {
                    get(node.input[1]).toVec1<Long>().map { xs.normalizeAxis(it) }.toMutableList()
                }
                axes.sort()
                var currentXs = xs
                for (axis in axes.reversed()) {
                    currentXs = currentXs.squeeze(axis)
                }
                values[node.output[0]] = currentXs
            }
            "ConstantOfShape" -> {
                val input = get(node.input[0])
                val value = getAttrOptOwned<Tensor>(node, "value")
                    ?: Tensor.zeros(Shape.SCALAR, DType.F32, Device.Cpu)
                val shapeVec = input.toVec1<Long>().map { it.toInt() }
                val xs = Tensor.ones(Shape(shapeVec), value.dtype(), input.device()).broadcastMul(value)
                values[node.output[0]] = xs
            }
            "Unsqueeze" -> {
                val xs = get(node.input[0])
                val axes = getAttrOpt<List<Long>>(node, "axes")
                    ?: get(node.input[1]).toVec1<Long>()
                val normalizedAxes = axes.map { i ->
                    if (i == xs.rank().toLong()) {
                        xs.rank()
                    } else if (i < 0) {
                        xs.rank() - (-i.toInt()) + 1
                    } else {
                        xs.normalizeAxis(i)
                    }
                }.toMutableList()
                normalizedAxes.sort()
                var currentXs = xs
                for (axis in normalizedAxes.reversed()) {
                    currentXs = currentXs.unsqueeze(axis)
                }
                values[node.output[0]] = currentXs
            }
            "Clip" -> {
                var xs = get(node.input[0])
                val mins = getOpt(1)
                if (mins != null) {
                    xs = xs.broadcastMaximum(mins)
                }
                val maxs = getOpt(2)
                if (maxs != null) {
                    xs = xs.broadcastMinimum(maxs)
                }
                values[node.output[0]] = xs
            }
            "Gather" -> {
                val xs = get(node.input[0])
                val indices = get(node.input[1])
                val axisI64 = getAttrOpt<Long>(node, "axis") ?: 0L
                val axis = xs.normalizeAxis(axisI64)

                val zeros = Tensor.zeros(indices.shape(), indices.dtype(), indices.device())
                val max = Tensor.new(longArrayOf(xs.dim(axis).toLong()), Shape(1), indices.device())
                    .toDtype(indices.dtype())
                val mask = indices.lt(zeros)
                val normIndices = mask.toDtype(indices.dtype()).broadcastMul(max).add(indices)

                val out = when (normIndices.dims().size) {
                    0 -> {
                        val index = normIndices.toVec0<Long>().toInt()
                        xs.narrow(axis, index, 1).squeeze(axis)
                    }
                    1 -> xs.indexSelect(normIndices, axis)
                    2 -> {
                        val first = normIndices.dims()[0]
                        val v = ArrayList<Tensor>(first)
                        for (i in 0 until first) {
                            v.add(xs.indexSelect(normIndices.get(i), axis))
                        }
                        Tensor.stack(v, axis)
                    }
                    else -> bail("implement gather for $xs $normIndices axis $axis")
                }
                values[node.output[0]] = out
            }
            "GatherElements" -> {
                val data = get(node.input[0])
                val indices = get(node.input[1])
                val rank = data.rank()
                if (rank != indices.rank()) {
                    bail("indices must have same rank as input data. Data rank [$rank] != indices rank [${indices.rank()}]")
                }
                val axisI64 = getAttrOpt<Long>(node, "axis") ?: 0L
                val axis = data.normalizeAxis(axisI64)
                if (axis >= rank) {
                    bail("axis ($axisI64) out of accepted range [-rank, rank-1] which was [-$rank, ${rank - 1}]")
                }
                val zeros = Tensor.zeros(indices.shape(), indices.dtype(), indices.device())
                val max = Tensor.new(longArrayOf(data.dims()[axis].toLong()), Shape(1), indices.device())
                    .toDtype(indices.dtype())
                val mask = indices.lt(zeros)
                val normIndices = mask.toDtype(indices.dtype()).broadcastMul(max).add(indices)
                values[node.output[0]] = data.gather(normIndices, axis)
            }
            "Shape" -> {
                val xs = get(node.input[0])
                val start = xs.normalizeAxis(getAttrOpt<Long>(node, "start") ?: 0L)
                val end = xs.normalizeAxis(getAttrOpt<Long>(node, "end") ?: -1L)
                val dims = ArrayList<Long>()
                for (idx in start..end) {
                    dims.add(xs.dim(idx).toLong())
                }
                val dimsTensor = Tensor.fromSlice(dims.toLongArray(), Shape(xs.rank()), xs.device())
                values[node.output[0]] = dimsTensor
            }
            "Size" -> {
                val data = get(node.input[0])
                val size = data.elemCount().toLong()
                val output = Tensor.fromSlice(longArrayOf(size), Shape.SCALAR, data.device())
                values[node.output[0]] = output
            }
            "Sqrt" -> {
                val xs = get(node.input[0])
                val output = xs.sqrt()
                values[node.output[0]] = output
            }
            "Range" -> {
                val start = get(node.input[0])
                val limit = get(node.input[1])
                val delta = get(node.input[2])

                val output = when (start.dtype()) {
                    DType.U8 -> arangeStep(
                        toVec0Flexible<UByte>(start).toLong(),
                        toVec0Flexible<UByte>(limit).toLong(),
                        toVec0Flexible<UByte>(delta).toLong(),
                        Device.Cpu,
                    ).toDtype(DType.U8)
                    DType.U32 -> arangeStep(
                        toVec0Flexible<UInt>(start).toLong(),
                        toVec0Flexible<UInt>(limit).toLong(),
                        toVec0Flexible<UInt>(delta).toLong(),
                        Device.Cpu,
                    ).toDtype(DType.U32)
                    DType.I64 -> arangeStep(
                        toVec0Flexible<Long>(start),
                        toVec0Flexible<Long>(limit),
                        toVec0Flexible<Long>(delta),
                        Device.Cpu,
                    )
                    DType.BF16, DType.F16, DType.F32, DType.F8E4M3 -> arangeStep(
                        toVec0Flexible<Float>(start).toDouble(),
                        toVec0Flexible<Float>(limit).toDouble(),
                        toVec0Flexible<Float>(delta).toDouble(),
                        Device.Cpu,
                    ).toDtype(start.dtype())
                    DType.F64 -> arangeStep(
                        toVec0Flexible<Double>(start),
                        toVec0Flexible<Double>(limit),
                        toVec0Flexible<Double>(delta),
                        Device.Cpu,
                    )
                    else -> bail("unsupported Range type ${start.dtype()}")
                }
                values[node.output[0]] = output
            }
            "Greater" -> {
                val a = get(node.input[0])
                val b = get(node.input[1])
                val output = a.broadcastGt(b)
                values[node.output[0]] = output
            }
            "Less" -> {
                val a = get(node.input[0])
                val b = get(node.input[1])
                val output = a.broadcastLt(b)
                values[node.output[0]] = output
            }
            "LessOrEqual" -> {
                val a = get(node.input[0])
                val b = get(node.input[1])
                val output = a.broadcastLe(b)
                values[node.output[0]] = output
            }
            "GreaterOrEqual" -> {
                val a = get(node.input[0])
                val b = get(node.input[1])
                val output = a.broadcastGe(b)
                values[node.output[0]] = output
            }
            "Log" -> {
                val a = get(node.input[0])
                val output = a.log()
                values[node.output[0]] = output
            }
            "Min" -> {
                var output = get(node.input[0])
                for (input in node.input) {
                    output = output.broadcastMinimum(get(input))
                }
                values[node.output[0]] = output
            }
            "Where" -> {
                val cond = get(node.input[0])
                val a = get(node.input[1])
                val b = get(node.input[2])
                val shape = broadcastShapeFromMany(listOf(cond.dims(), a.dims(), b.dims()))
                val condB = cond.broadcastAs(Shape(shape))
                val aB = a.broadcastAs(Shape(shape))
                val bB = b.broadcastAs(Shape(shape))
                val output = condB.whereCond(aB, bB)
                values[node.output[0]] = output
            }
            "Conv" -> {
                val dilations = getAttrOpt<List<Long>>(node, "dilations")
                val groups = (getAttrOpt<Long>(node, "group") ?: 1L).toInt()
                val _kernelShape = getAttrOpt<List<Long>>(node, "kernel_shape")
                val pads = getAttrOpt<List<Long>>(node, "pads")
                val strides = getAttrOpt<List<Long>>(node, "strides")
                val autoPad = getAttrOpt<String>(node, "auto_pad")
                when (autoPad) {
                    null, "NOTSET" -> Unit
                    else -> bail("unsupported auto_pad $autoPad")
                }
                val xs = get(node.input[0])
                val ws = get(node.input[1])
                val ys = when (ws.rank()) {
                    3 -> {
                        val (padsVal, xsPadded) = when {
                            pads == null -> Pair(0, xs)
                            pads.size == 1 -> Pair(pads[0].toInt(), xs)
                            pads.size == 2 -> {
                                val p1 = pads[0].toInt()
                                val p2 = pads[1].toInt()
                                if (p1 != p2) {
                                    Pair(0, xs.padWithZeros(2, p1, p2))
                                } else {
                                    Pair(p1, xs)
                                }
                            }
                            else -> bail("more pads than expected in conv1d $pads ${node.name}")
                        }
                        val stridesVal = when {
                            strides == null -> 1
                            strides.size == 1 -> strides[0].toInt()
                            else -> bail("more strides than expected in conv1d $strides ${node.name}")
                        }
                        val dilationsVal = when {
                            dilations == null -> 1
                            dilations.size == 1 -> dilations[0].toInt()
                            else -> bail("more dilations than expected in conv1d $dilations ${node.name}")
                        }
                        xsPadded.conv1d(ws, padsVal, stridesVal, dilationsVal, groups)
                    }
                    4 -> {
                        val (padsVal, xsPadded) = when {
                            pads == null -> Pair(0, xs)
                            pads.size == 1 -> Pair(pads[0].toInt(), xs)
                            pads.size == 4 -> {
                                val p1 = pads[0].toInt()
                                val p2 = pads[1].toInt()
                                val p3 = pads[2].toInt()
                                val p4 = pads[3].toInt()
                                if (p1 != p2 || p1 != p3 || p1 != p4) {
                                    Pair(0, xs.padWithZeros(2, p1, p3).padWithZeros(3, p2, p4))
                                } else {
                                    Pair(p1, xs)
                                }
                            }
                            else -> bail("more pads than expected in conv2d $pads ${node.name}")
                        }
                        val stridesVal = when {
                            strides == null -> 1
                            strides.size == 1 -> strides[0].toInt()
                            strides.size == 2 -> {
                                if (strides[0] != strides[1]) {
                                    bail("strides have to be the same on both axis $pads ${node.name}")
                                }
                                strides[0].toInt()
                            }
                            else -> bail("more strides than expected in conv2d $strides ${node.name}")
                        }
                        val dilationsVal = when {
                            dilations == null -> 1
                            dilations.size == 1 -> dilations[0].toInt()
                            dilations.size == 2 -> {
                                if (dilations[0] != dilations[1]) {
                                    bail("dilations have to be the same on both axis $pads ${node.name}")
                                }
                                dilations[0].toInt()
                            }
                            else -> bail("more dilations than expected in conv2d $dilations ${node.name}")
                        }
                        xsPadded.conv2d(ws, padsVal, stridesVal, dilationsVal, groups)
                    }
                    else -> bail("unsupported rank for weight matrix ${ws.rank()} in conv ${node.name}")
                }
                val finalYs = if (node.input.size > 2) {
                    val bs = get(node.input[2])
                    val bsShape = MutableList(ys.rank()) { 1 }
                    bsShape[1] = bs.elemCount()
                    ys.broadcastAdd(bs.reshape(Shape(bsShape)))
                } else {
                    ys
                }
                values[node.output[0]] = finalYs
            }
            "Concat" -> {
                val inputs = node.input.map { n -> get(n) }
                val axisI64 = getAttr<Long>(node, "axis")
                if (inputs.isEmpty()) {
                    bail("empty concat")
                }
                val minRank = inputs.minOf { it.rank() }
                val normalizedInputs = inputs.map { t ->
                    var current = t
                    while (current.rank() > minRank) {
                        val lastDim = current.rank() - 1
                        if (current.dims()[lastDim] == 1) {
                            current = try { current.squeeze(lastDim) } catch (_: Exception) { current }
                        } else {
                            break
                        }
                    }
                    current
                }
                val axis = normalizedInputs[0].normalizeAxis(axisI64)
                val output = Tensor.cat(normalizedInputs, axis)
                values[node.output[0]] = output
            }
            "Abs" -> {
                val input = get(node.input[0])
                val output = input.abs()
                values[node.output[0]] = output
            }
            "Cos" -> {
                val input = get(node.input[0])
                val output = input.cos()
                values[node.output[0]] = output
            }
            "Sin" -> {
                val input = get(node.input[0])
                val output = input.sin()
                values[node.output[0]] = output
            }
            "Neg" -> {
                val input = get(node.input[0])
                val output = if (input.dtype() == DType.I64) {
                    val minusOne = Tensor.new(longArrayOf(-1L), Shape(1), input.device()).broadcastAs(input.shape())
                    input.mul(minusOne)
                } else {
                    input.neg()
                }
                values[node.output[0]] = output
            }
            "Erf" -> {
                val input = get(node.input[0])
                val output = input.erf()
                values[node.output[0]] = output
            }
            "Tanh" -> {
                val input = get(node.input[0])
                val output = input.tanh()
                values[node.output[0]] = output
            }
            "Sigmoid" -> {
                val input = get(node.input[0])
                val output = sigmoid(input)
                values[node.output[0]] = output
            }
            "Gelu" -> {
                val input = get(node.input[0])
                val output = input.geluErf()
                values[node.output[0]] = output
            }
            "Relu" -> {
                val input = get(node.input[0])
                val output = input.relu()
                values[node.output[0]] = output
            }
            "PRelu" -> {
                val input = get(node.input[0])
                val slope = get(node.input[1])
                val output = PReLU(slope, false).forward(input)
                values[node.output[0]] = output
            }
            "Ceil" -> {
                val input = get(node.input[0])
                val output = input.ceil()
                values[node.output[0]] = output
            }
            "Floor" -> {
                val input = get(node.input[0])
                val output = input.floor()
                values[node.output[0]] = output
            }
            "Constant" -> {
                val value = node.attribute.firstOrNull { it.name == "value" }
                    ?: bail("cannot find 'value' attr in 'Constant' for ${node.name}")
                val output = when (value.type()) {
                    AttributeProto.AttributeType.TENSOR -> {
                        val t = value.t!!
                        getTensor(t, node.name)
                    }
                    else -> bail("unsupported 'value' type ${value.type()} for ${node.name}")
                }
                values[node.output[0]] = output
            }
            "Cast" -> {
                val input = get(node.input[0])
                val dt = getAttr<Long>(node, "to")
                val dtEnum = TensorProto.DataType.fromI32(dt.toInt())
                val dtypeVal = when (dtEnum) {
                    TensorProto.DataType.INT32 -> DType.I64
                    null -> bail("unsupported 'to' value $dt for cast ${node.name}")
                    else -> dtype(dtEnum) ?: bail("unsupported 'to' value $dt for cast ${node.name}")
                }
                val output = input.toDtype(dtypeVal)
                values[node.output[0]] = output
            }
            "CumSum" -> {
                val exclusive = getAttrOpt<Long>(node, "exclusive") ?: 0L
                val reverse = getAttrOpt<Long>(node, "reverse") ?: 0L
                if (exclusive != 0L) {
                    bail("only exclusive == 0 is supported in CumSum")
                }
                if (reverse != 0L) {
                    bail("only reverse == 0 is supported in CumSum")
                }
                val input = get(node.input[0])
                val axis = toVec0Flexible<UInt>(get(node.input[1]).toDtype(DType.U32)).toInt()
                val output = input.cumsum(axis)
                values[node.output[0]] = output
            }
            "Flatten" -> {
                val axis = (getAttrOpt<Long>(node, "axis") ?: 1L).toInt()
                val input = get(node.input[0])
                val firstPart = input.dims().take(axis).fold(1) { acc, v -> acc * v }
                val endIndex = input.elemCount()
                val newShape = Shape(firstPart, endIndex / firstPart)
                val output = input.reshape(newShape)
                values[node.output[0]] = output
            }
            "Identity" -> {
                val input = get(node.input[0])
                values[node.output[0]] = input
            }
            "If" -> {
                val cond = toScalarFlexible<UByte>(get(node.input[0]).get(0)).toInt()
                val attrName = if (cond != 0) "then_branch" else "else_branch"
                val subGraph = getAttr<GraphProto>(node, attrName)
                if (subGraph.output.size != node.output.size) {
                    bail("If node ${node.name} is malformed: branch outputs (${subGraph.output.size}) don't match node outputs (${node.output.size})")
                }
                val branchOut = simpleEval_(subGraph, values)
                for ((i, out) in node.output.withIndex()) {
                    values[out] = branchOut[subGraph.output[i].name]!!
                }
            }
            "Pad" -> {
                val mode = getAttrOpt<String>(node, "mode") ?: "constant"
                val data = get(node.input[0])
                val pads = get(node.input[1])
                if (node.input.size > 2) {
                    bail("unsupported number of inputs ${node.input.size} for Pad node ${node.name}, expected 2")
                }
                if (pads.rank() != 1) {
                    bail("Pad expects 'pads' input to be 1D vector: $pads")
                }
                if (pads.dim(0) != 2 * data.rank()) {
                    bail("Pad expects 'pads' input len to be 2 * rank of 'data' input: pads: ${pads.dim(0)}, data rank: ${data.rank()}")
                }
                val padsVec = pads.toVec1<Long>()
                val half = padsVec.size / 2
                val padsPre = padsVec.subList(0, half)
                val padsPost = padsVec.subList(half, padsVec.size)

                when (mode) {
                    "reflect" -> {
                        var out = data
                        for (i in data.dims().indices.reversed()) {
                            val dim = data.dims()[i]
                            if (padsPre[i] == 0L && padsPost[i] == 0L) {
                                continue
                            }
                            fun zigzag(min: Long, max: Long): Sequence<Long> = sequence {
                                while (true) {
                                    for (v in min until max) yield(v)
                                    for (v in max downTo (min + 1L)) yield(v)
                                }
                            }
                            val idxTensor = if (dim > 1) {
                                val cycleLen = dim * 2 - 2
                                val skip = cycleLen - (padsPre[i].toInt() % cycleLen)
                                val count = padsPre[i].toInt() + dim + padsPost[i].toInt()
                                val idxList = zigzag(0L, (dim - 1).toLong()).drop(skip).take(count).toList()
                                Tensor.fromSlice(idxList.toLongArray(), Shape(idxList.size), out.device())
                            } else {
                                Tensor.zeros(Shape(dim), DType.I64, out.device())
                            }
                            out = out.indexSelect(idxTensor, i)
                        }
                        values[node.output[0]] = out
                    }
                    else -> bail("unsupported 'mode' value $mode for Pad node ${node.name}")
                }
            }
            "Slice" -> {
                val data = get(node.input[0])
                val starts = get(node.input[1])
                val ends = get(node.input[2])
                val axes: Tensor
                val steps: Tensor
                when (node.input.size) {
                    3 -> {
                        val len = starts.dims()[0]
                        axes = Tensor.fromSlice(LongArray(len) { it.toLong() }, Shape(len), starts.device())
                        steps = Tensor.ones(Shape(len), DType.I64, starts.device())
                    }
                    4 -> {
                        val len = starts.dims()[0]
                        axes = get(node.input[3])
                        steps = Tensor.ones(Shape(len), DType.I64, starts.device())
                    }
                    5 -> {
                        axes = get(node.input[3])
                        steps = get(node.input[4])
                    }
                    else -> bail("Slice node is invalid, expected 3-5 inputs, got ${node.input.size}: $node")
                }

                var out = data
                val axesVec = axes.toVec1<Long>()
                for ((i, rawAxis) in axesVec.withIndex()) {
                    val axis = (if (rawAxis < 0) rawAxis + data.rank().toLong() else rawAxis).toInt()
                    val dataDim = data.dims()[axis].toLong()
                    var s = toScalarFlexible<Long>(starts.get(i))
                    var e = toScalarFlexible<Long>(ends.get(i))
                    if (s < 0) s += dataDim
                    if (e < 0) e += dataDim
                    val p = toScalarFlexible<Long>(steps.get(i))
                    if (p >= 0) {
                        s = s.coerceIn(0L, dataDim)
                        e = e.coerceIn(0L, dataDim)
                    } else {
                        s = s.coerceIn(0L, dataDim - 1L)
                        e = e.coerceIn(-1L, dataDim - 1L)
                    }
                    val indexes = arangeStep(s, e, p, data.device())
                    out = out.contiguous().indexSelect(indexes, axis)
                }
                values[node.output[0]] = out
            }
            "ReduceMax" -> {
                val input = get(node.input[0])
                val axesOpt = getOpt(1)
                val keepdims = (getAttrOpt<Long>(node, "keepdims") ?: 1L) == 1L
                val axesList = if (axesOpt != null) {
                    try { axesOpt.toVec1<Long>() } catch (_: Exception) { null }
                } else {
                    getAttrOpt<List<Long>>(node, "axes")
                }

                val axes = if (axesList != null) {
                    val rank = input.rank()
                    val axesSet = HashSet<Int>()
                    val resolvedAxes = ArrayList<Int>(axesList.size)
                    for (a in axesList) {
                        val axis = (if (a < 0) rank.toLong() + a else a).toInt()
                        axesSet.add(axis)
                        resolvedAxes.add(axis)
                    }
                    if (axesSet.size < resolvedAxes.size) {
                        bail("Duplicate value in 'axes'")
                    }
                    resolvedAxes.sort()
                    resolvedAxes
                } else {
                    null
                }

                if (input.elemCount() == 0) {
                    bail("reduction over zero-size tensor not supported")
                }

                val output = if (axes != null) {
                    var result = input
                    for (axis in axes.reversed()) {
                        result = if (keepdims) result.maxKeepdim(axis) else result.max(axis)
                    }
                    result
                } else {
                    if (getAttrOpt<Long>(node, "noop_with_empty_axes") == 1L) {
                        input
                    } else {
                        var result = input.flattenAll()
                        if (keepdims) {
                            result = result.maxKeepdim(0)
                            val shape = Shape(List(input.rank()) { 1 })
                            result.reshape(shape)
                        } else {
                            result.max(0)
                        }
                    }
                }
                values[node.output[0]] = output
            }
            "ReduceMean" -> {
                val input = get(node.input[0])
                val axesAttr = getAttrOpt<List<Long>>(node, "axes")
                val keepdims = (getAttrOpt<Long>(node, "keepdims") ?: 1L) == 1L
                val nDims = input.dims().size
                val axes = if (axesAttr != null) {
                    axesAttr.map { (if (it < 0) nDims.toLong() + it else it).toInt() }
                } else {
                    (0 until nDims).toList()
                }
                val output = if (keepdims) input.meanKeepdim(axes) else input.mean(axes)
                values[node.output[0]] = output
            }
            "ReduceMin" -> {
                val input = get(node.input[0])
                val axesOpt = getOpt(1)
                val keepdims = (getAttrOpt<Long>(node, "keepdims") ?: 1L) == 1L
                val axesList = if (axesOpt != null) {
                    try { axesOpt.toVec1<Long>() } catch (_: Exception) { null }
                } else {
                    getAttrOpt<List<Long>>(node, "axes")
                }

                val axes = if (axesList != null) {
                    val rank = input.rank()
                    val axesSet = HashSet<Int>()
                    val resolvedAxes = ArrayList<Int>(axesList.size)
                    for (a in axesList) {
                        val axis = (if (a < 0) rank.toLong() + a else a).toInt()
                        axesSet.add(axis)
                        resolvedAxes.add(axis)
                    }
                    if (axesSet.size < resolvedAxes.size) {
                        bail("Duplicate value in 'axes'")
                    }
                    resolvedAxes.sort()
                    resolvedAxes
                } else {
                    null
                }

                if (input.elemCount() == 0) {
                    bail("reduction over zero-size tensor not supported")
                }

                val output = if (axes != null) {
                    var result = input
                    for (axis in axes.reversed()) {
                        result = if (keepdims) result.minKeepdim(axis) else result.min(axis)
                    }
                    result
                } else {
                    if (getAttrOpt<Long>(node, "noop_with_empty_axes") == 1L) {
                        input
                    } else {
                        var result = input.flattenAll()
                        if (keepdims) {
                            result = result.minKeepdim(0)
                            val shape = Shape(List(input.rank()) { 1 })
                            result.reshape(shape)
                        } else {
                            result.min(0)
                        }
                    }
                }
                values[node.output[0]] = output
            }
            "Split" -> {
                val inputTensor = get(node.input[0])
                val axisI64 = getAttrOpt<Long>(node, "axis") ?: 0L
                val axis = inputTensor.normalizeAxis(axisI64)

                val splits = if (node.input.size > 1) {
                    get(node.input[1]).toVec1<Long>().map { it.toInt() }
                } else {
                    val numOutputs = (getAttrOpt<Long>(node, "num_outputs")?.toInt()) ?: node.output.size
                    val inputDim = inputTensor.dim(axis)
                    val splitSizes = MutableList(numOutputs) { inputDim / numOutputs }
                    val remainder = inputDim % numOutputs
                    if (remainder > 0) {
                        splitSizes[numOutputs - 1] += remainder
                    }
                    splitSizes
                }

                val outputs = ArrayList<Tensor>(splits.size)
                var start = 0
                for (size in splits) {
                    val end = start + size
                    val slice = inputTensor.narrow(axis, start, size)
                    outputs.add(slice)
                    start = end
                }

                for ((output, slice) in node.output.zip(outputs)) {
                    values[output] = slice
                }
            }
            "Expand" -> {
                val inputTensor = get(node.input[0])
                val inputShape = get(node.input[1])
                if (inputShape.rank() != 1) {
                    bail("Expand expects 'shape' input to be 1D tensor: $inputShape")
                }
                val inputTensorDims = inputTensor.dims()
                val inputShapeDims = inputShape.toVec1<Long>().map { it.toInt() }
                val targetShape = broadcastShape(inputTensorDims, inputShapeDims)
                val expandedTensor = inputTensor.broadcastAs(Shape(targetShape))
                values[node.output[0]] = expandedTensor
            }
            "Tile" -> {
                val input = get(node.input[0])
                val repeats = get(node.input[1]).toVec1<Long>()
                var result = input
                for ((dim, repeat) in repeats.withIndex()) {
                    if (repeat > 1L) {
                        val repeatInt = repeat.toInt()
                        val tensors = (0 until repeatInt).map { result }
                        result = Tensor.cat(tensors, dim)
                    }
                }
                values[node.output[0]] = result
            }
            "ReduceSum" -> {
                val input = get(node.input[0])
                val axesOpt = getOpt(1)
                val keepdims = (getAttrOpt<Long>(node, "keepdims") ?: 1L) == 1L
                val noopWithEmptyAxes = (getAttrOpt<Long>(node, "noop_with_empty_axes") ?: 0L) == 1L

                val axes = if (axesOpt != null) {
                    try {
                        axesOpt.toVec1<Long>().map { it.toInt() }
                    } catch (_: Exception) {
                        if (noopWithEmptyAxes) emptyList() else (0 until input.rank()).toList()
                    }
                } else {
                    if (noopWithEmptyAxes) emptyList() else (0 until input.rank()).toList()
                }

                val output = if (keepdims) input.sumKeepdim(axes) else input.sum(axes)
                values[node.output[0]] = output
            }
            "ReduceL2" -> {
                val input = get(node.input[0])
                val axesOpt = getOpt(1)
                val keepdims = (getAttrOpt<Long>(node, "keepdims") ?: 1L) == 1L
                val noopWithEmptyAxes = (getAttrOpt<Long>(node, "noop_with_empty_axes") ?: 0L) == 1L
                val inputSq = input.sqr()

                val axes = if (axesOpt != null) {
                    axesOpt.toVec1<Long>().map { it.toInt() }
                } else {
                    if (noopWithEmptyAxes) emptyList() else (0 until inputSq.rank()).toList()
                }

                val output = if (keepdims) inputSq.sumKeepdim(axes).sqrt() else inputSq.sum(axes).sqrt()
                values[node.output[0]] = output
            }
            "RandomUniform", "RandomNormal" -> {
                val randomType = node.opType
                val dt = getAttrOpt<Long>(node, "dtype") ?: 1L
                val dtEnum = TensorProto.DataType.fromValue(dt.toInt())
                val dtypeVal = when (dtEnum) {
                    TensorProto.DataType.UINT8, TensorProto.DataType.UINT32, TensorProto.DataType.INT64 -> {
                        bail("unsupported 'dtype' value $dtEnum, only floats are allowed, for $randomType ${node.name}")
                    }
                    null -> bail("unsupported 'dtype' value $dt for $randomType ${node.name}")
                    else -> dtype(dtEnum) ?: bail("unsupported 'dtype' value $dtEnum for $randomType ${node.name}")
                }
                val seed = getAttrOpt<Float>(node, "seed")
                if (seed != null) {
                    bail("seed for $randomType is currently not supported")
                }
                val shapeDims = getAttr<List<Long>>(node, "shape").map { it.toInt() }
                val shapeObj = Shape(shapeDims)
                val output = if (randomType == "RandomUniform") {
                    val low = getAttrOpt<Float>(node, "low") ?: 0.0f
                    val high = getAttrOpt<Float>(node, "high") ?: 1.0f
                    Tensor.rand(low, high, shapeObj, Device.Cpu).toDtype(dtypeVal)
                } else {
                    val mean = getAttrOpt<Float>(node, "mean") ?: 0.0f
                    val scale = getAttrOpt<Float>(node, "scale") ?: 1.0f
                    Tensor.randn(mean, scale, shapeObj, Device.Cpu).toDtype(dtypeVal)
                }
                values[node.output[0]] = output
            }
            "ArgMin" -> {
                val input = get(node.input[0])
                val axisI64 = getAttrOpt<Long>(node, "axis") ?: 0L
                val rankI64 = input.rank().toLong()
                if (axisI64 < -rankI64 || axisI64 >= rankI64) {
                    bail("axis ($axisI64) out of accepted range [-rank, rank-1] which was [-$rankI64, ${rankI64 - 1}]")
                }
                val axis = input.normalizeAxis(axisI64)
                val keepdims = (getAttrOpt<Long>(node, "keepdims") ?: 1L) == 1L
                val selectLastIndex = getAttrOpt<Long>(node, "select_last_index") ?: 0L
                if (selectLastIndex == 1L) {
                    bail("select_last_index for ArgMin is currently not supported")
                }
                val output = if (keepdims) input.argminKeepdim(axis) else input.argmin(axis)
                values[node.output[0]] = output.toDtype(DType.I64)
            }
            "ArgMax" -> {
                val input = get(node.input[0])
                val axisI64 = getAttrOpt<Long>(node, "axis") ?: 0L
                val rankI64 = input.rank().toLong()
                if (axisI64 < -rankI64 || axisI64 >= rankI64) {
                    bail("axis ($axisI64) out of accepted range [-rank, rank-1] which was [-$rankI64, ${rankI64 - 1}]")
                }
                val axis = input.normalizeAxis(axisI64)
                val keepdims = (getAttrOpt<Long>(node, "keepdims") ?: 1L) == 1L
                val selectLastIndex = getAttrOpt<Long>(node, "select_last_index") ?: 0L
                if (selectLastIndex == 1L) {
                    bail("select_last_index for ArgMin is currently not supported")
                }
                val output = if (keepdims) input.argmaxKeepdim(axis) else input.argmax(axis)
                values[node.output[0]] = output.toDtype(DType.I64)
            }
            "LeakyRelu" -> {
                val input = get(node.input[0])
                when (val dt = input.dtype()) {
                    DType.U8, DType.U32, DType.I64 -> {
                        bail("unsupported dtype $dt, only float types are allowed for LeakyRelu")
                    }
                    DType.BF16, DType.F16, DType.F32, DType.F64 -> Unit
                    else -> bail("unsupported dtype $dt for LeakyRelu")
                }
                val alpha = getAttrOpt<Float>(node, "alpha") ?: 0.01f
                val output = leakyRelu(input, alpha.toDouble())
                values[node.output[0]] = output
            }
            "Gemm" -> {
                val a = get(node.input[0])
                val b = get(node.input[1])
                val c = get(node.input[2])
                val alpha = getAttrOpt<Float>(node, "alpha") ?: 1.0f
                val beta = getAttrOpt<Float>(node, "beta") ?: 1.0f
                val alphaTensor = Tensor.new(floatArrayOf(alpha), Shape.SCALAR, Device.Cpu).broadcastAs(a.shape())
                val betaTensor = Tensor.new(floatArrayOf(beta), Shape.SCALAR, Device.Cpu).broadcastAs(c.shape())
                val transA = getAttrOpt<Long>(node, "transA") ?: 0L
                val transB = getAttrOpt<Long>(node, "transB") ?: 0L
                val matA = if (transA == 0L) a else a.t()
                val matB = if (transB == 0L) b else b.t()
                val output = matA.broadcastMul(alphaTensor).broadcastMatmul(matB).broadcastAdd(c.broadcastMul(betaTensor))
                values[node.output[0]] = output
            }
            "LSTM" -> {
                val direction = getAttrOpt<String>(node, "direction") ?: "forward"
                if (direction != "forward") {
                    bail("LSTM currently only supports direction == \"forward\"")
                }
                val numDirections = if (direction == "bidirectional") 2 else 1
                val hiddenSize = getAttr<Long>(node, "hidden_size")
                val inputForget = getAttrOpt<Long>(node, "input_forget") ?: 0L
                if (inputForget != 0L) {
                    bail("LSTM currently only supports input_forget == 0")
                }
                val activationsDefault = listOf("Sigmoid", "Tanh", "Tanh")
                val activations = getAttrOptOwned<List<String>>(node, "activations") ?: activationsDefault
                if (activations != activationsDefault) {
                    bail("LSTM currently only supports default activations ($activationsDefault)")
                }
                if (getAttrOpt<Float>(node, "clip") != null) {
                    bail("LSTM does not currently support clip attribute")
                }
                val layout = getAttrOpt<Long>(node, "layout") ?: 0L
                if (layout != 0L) {
                    bail("LSTM currently only supports layout == 0")
                }

                val x = get(node.input[0])
                val (seqLength, batchSize, inputSize) = x.dims3()
                val w = get(node.input[1])
                val r = get(node.input[2])

                val b = if (node.input.size > 3 && node.input[3].isNotEmpty()) {
                    get(node.input[3])
                } else {
                    Tensor.zeros(Shape(numDirections, 8 * hiddenSize.toInt()), DType.F32, x.device())
                }

                val seqLens = if (node.input.size > 4 && node.input[4].isNotEmpty()) {
                    get(node.input[4])
                } else {
                    Tensor.new(LongArray(batchSize) { seqLength.toLong() }, Shape(batchSize), x.device())
                }
                val seqLensIsDefault = seqLens.toVec1<Long>().all { it.toInt() == seqLength }
                if (!seqLensIsDefault) {
                    bail("LSTM currently only supports default value of seq_lens")
                }

                val initialH = if (node.input.size > 5 && node.input[5].isNotEmpty()) {
                    get(node.input[5])
                } else {
                    Tensor.zeros(Shape(numDirections, batchSize, hiddenSize.toInt()), DType.F32, x.device())
                }

                val initialC = if (node.input.size > 6 && node.input[6].isNotEmpty()) {
                    get(node.input[6])
                } else {
                    Tensor.zeros(Shape(numDirections, batchSize, hiddenSize.toInt()), DType.F32, x.device())
                }

                val p = if (node.input.size > 7 && node.input[7].isNotEmpty()) {
                    get(node.input[7])
                } else {
                    Tensor.zeros(Shape(numDirections, 3 * hiddenSize.toInt()), DType.F32, x.device())
                }
                val pIsZeros = p.toVec2<Float>().all { v -> v.all { it == 0.0f } }
                if (!pIsZeros) {
                    bail("LSTM currently only supports default value of p (a Tensor of all zeroes)")
                }

                val w0 = w.get(0)
                val r0 = r.get(0)
                val b0 = b.get(0)
                val idxWb = arange(0L, 4 * hiddenSize, x.device())
                val idxRb = arange(4 * hiddenSize, 8 * hiddenSize, x.device())
                val wb = b0.indexSelect(idxWb, 0)
                val rb = b0.indexSelect(idxRb, 0)
                val c0 = initialC.get(0)
                val h0 = initialH.get(0)

                val idxI = arange(0L, hiddenSize, x.device())
                val idxO = arange(hiddenSize, 2 * hiddenSize, x.device())
                val idxF = arange(2 * hiddenSize, 3 * hiddenSize, x.device())
                val idxC = arange(3 * hiddenSize, 4 * hiddenSize, x.device())
                val idxIfco = Tensor.cat(listOf(idxI, idxF, idxC, idxO), 0)
                val wIfco = w0.indexSelect(idxIfco, 0)
                val rIfco = r0.indexSelect(idxIfco, 0)
                val wbIfco = wb.indexSelect(idxIfco, 0)
                val rbIfco = rb.indexSelect(idxIfco, 0)

                val vmap = VarMap.new()
                val dataMap = vmap.data() as MutableMap<String, Var>
                dataMap["weight_ih_l0"] = Var.fromTensor(wIfco)
                dataMap["weight_hh_l0"] = Var.fromTensor(rIfco)
                dataMap["bias_ih_l0"] = Var.fromTensor(wbIfco)
                dataMap["bias_hh_l0"] = Var.fromTensor(rbIfco)

                val lstmModel = lstm(
                    inputSize,
                    hiddenSize.toInt(),
                    LSTMConfig.default(),
                    VarBuilder.fromVarmap(vmap, wIfco.dtype(), wIfco.device()),
                )

                var lstmState = LSTMState.new(h0, c0)
                val hAcc = if (node.output.firstOrNull()?.isNotEmpty() == true) ArrayList<LSTMState>() else null
                for (t in 0 until seqLength) {
                    val xt = x.get(t)
                    lstmState = lstmModel.step(xt, lstmState)
                    hAcc?.add(lstmState.clone())
                }

                if (node.output.isNotEmpty() && hAcc != null) {
                    val hAccTensor = lstmModel.statesToTensor(hAcc)
                    val hAccReshaped = hAccTensor.reshape(Shape(seqLength, numDirections, batchSize, hiddenSize.toInt()))
                    values[node.output[0]] = hAccReshaped
                }
                if (node.output.size > 1) {
                    values[node.output[1]] = lstmState.h().reshape(Shape(numDirections, batchSize, hiddenSize.toInt()))
                }
                if (node.output.size > 2) {
                    values[node.output[2]] = lstmState.c().reshape(Shape(numDirections, batchSize, hiddenSize.toInt()))
                }
            }
            "RNN" -> {
                val activationsDefault = listOf("Tanh", "Tanh")
                val activations = getAttrOptOwned<List<String>>(node, "activations") ?: activationsDefault
                if (getAttrOpt<Float>(node, "clip") != null) {
                    bail("RNN does not currently support clip attribute")
                }
                val direction = getAttrOpt<String>(node, "direction") ?: "forward"
                if (direction != "forward") {
                    bail("RNN currently only supports direction == \"forward\"")
                }
                val numDirections = if (direction == "bidirectional") 2 else 1
                val hiddenSize = getAttr<Long>(node, "hidden_size")
                val layout = getAttrOpt<Long>(node, "layout") ?: 0L
                if (layout != 0L) {
                    bail("RNN currently only supports layout == 0")
                }

                val x = get(node.input[0])
                val (seqLength, batchSize, _) = x.dims3()
                val w = get(node.input[1])
                val r = get(node.input[2])

                val b = if (node.input.size > 3 && node.input[3].isNotEmpty()) {
                    get(node.input[3])
                } else {
                    Tensor.zeros(Shape(numDirections, 2 * hiddenSize.toInt()), DType.F32, x.device())
                }

                val seqLens = if (node.input.size > 4 && node.input[4].isNotEmpty()) {
                    get(node.input[4])
                } else {
                    Tensor.new(LongArray(batchSize) { seqLength.toLong() }, Shape(batchSize), x.device())
                }
                val seqLensIsDefault = seqLens.toVec1<Long>().all { it.toInt() == seqLength }
                if (!seqLensIsDefault) {
                    bail("RNN currently does not support variable-length sequences. All sequences must use the full sequence length of $seqLength")
                }

                val initialH = if (node.input.size > 5 && node.input[5].isNotEmpty()) {
                    get(node.input[5])
                } else {
                    Tensor.zeros(Shape(numDirections, batchSize, hiddenSize.toInt()), DType.F32, x.device())
                }

                fun chooseActivation(activation: String, tensor: Tensor): Tensor =
                    when (activation) {
                        "Tanh" -> tensor.tanh()
                        else -> bail("unsupported activation $activation")
                    }

                val w0 = w.get(0)
                val r0 = r.get(0)
                val b0 = b.get(0)
                val idxWb = arange(0L, hiddenSize, x.device())
                val idxRb = arange(hiddenSize, 2 * hiddenSize, x.device())
                val wb = b0.indexSelect(idxWb, 0)
                val rb = b0.indexSelect(idxRb, 0)
                var hT = initialH.get(0)
                val hList = ArrayList<Tensor>(seqLength)
                for (i in 0 until seqLength) {
                    val xs = x.get(i)
                    val hStep = xs
                        .matmul(w0.t())
                        .add(hT.matmul(r0.t()))
                        .add(wb.unsqueeze(0))
                        .add(rb.unsqueeze(0))
                    val activatedH = chooseActivation(activations[0], hStep)
                    hList.add(activatedH)
                    hT = activatedH
                }
                val hStack = Tensor.stack(hList, 0)
                val hFinal = hStack.reshape(Shape(seqLength, numDirections, batchSize, hiddenSize.toInt()))
                values[node.output[0]] = hFinal
                values[node.output[1]] = hT.reshape(Shape(numDirections, batchSize, hiddenSize.toInt()))
            }
            "Xor" -> {
                val input0 = get(node.input[0])
                val input1 = get(node.input[1])
                val a = input0.gt(input0.zerosLike())
                val b = input1.gt(input1.zerosLike())
                val sum = a.broadcastAdd(b)
                val out = sum.eq(sum.onesLike())
                values[node.output[0]] = out
            }
            "And" -> {
                val input0 = get(node.input[0])
                val input1 = get(node.input[1])
                val a = input0.gt(input0.zerosLike())
                val b = input1.gt(input1.zerosLike())
                val out = a.broadcastMul(b)
                values[node.output[0]] = out
            }
            "Or" -> {
                val input0 = get(node.input[0])
                val input1 = get(node.input[1])
                val a = input0.gt(input0.zerosLike())
                val b = input1.gt(input1.zerosLike())
                val sum = a.broadcastAdd(b)
                val out = sum.gt(sum.zerosLike())
                values[node.output[0]] = out
            }
            "Sign" -> {
                val input = get(node.input[0])
                val output = input.sign()
                values[node.output[0]] = output
            }
            "Selu" -> {
                val input = get(node.input[0])
                val alpha = getAttrOpt<Float>(node, "alpha") ?: 1.6732632f
                val gamma = getAttrOpt<Float>(node, "gamma") ?: 1.050701f
                val out = selu(input, alpha, gamma)
                values[node.output[0]] = out
            }
            "OneHot" -> {
                val indices = get(node.input[0])
                val origShape = indices.dims()
                val depthTensor = get(node.input[1])
                val valuesTensor = get(node.input[2])
                val depth = toScalarFlexible<Long>(depthTensor).toInt()
                val valuesVec = valuesTensor.toVec1<Float>()
                if (valuesVec.size != 2) {
                    bail("OneHot: expected 2-element values tensor")
                }
                val offValue = valuesVec[0]
                val onValue = valuesVec[1]
                var axis = node.attribute.firstOrNull { it.name == "axis" }?.i ?: -1L
                val rank = indices.rank()
                if (axis < -(rank + 1).toLong() || axis > rank.toLong()) {
                    bail("OneHot: invalid axis $axis for rank $rank")
                }
                if (axis < 0) {
                    axis += (rank + 1).toLong()
                }

                val flatIndices = indices.flattenAll()
                val indicesVec = flatIndices.toVec1<Long>()
                val outArray = FloatArray(depth * flatIndices.elemCount()) { offValue }
                for ((i, index) in indicesVec.withIndex()) {
                    val idx = if (index < 0) (index + depth.toLong()).toInt() else index.toInt()
                    if (idx in 0 until depth) {
                        outArray[i * depth + idx] = onValue
                    }
                }
                val targetShape = ArrayList<Int>(origShape.size + 1)
                targetShape.addAll(origShape)
                targetShape.add(depth)
                val output = Tensor.fromSlice(outArray, Shape(targetShape), indices.device())
                val finalOutput = if (axis.toInt() == output.rank() - 1) {
                    output
                } else {
                    fun moveAxisTo(rank: Int, from: Int, to: Int): List<Int> {
                        val dims = (0 until rank).toMutableList()
                        val axisVal = dims.removeAt(from)
                        dims.add(to, axisVal)
                        return dims
                    }
                    val perm = moveAxisTo(output.rank(), output.rank() - 1, axis.toInt())
                    output.permute(perm)
                }
                values[node.output[0]] = finalOutput
            }
            "HardSwish" -> {
                val input = get(node.input[0])
                val hardSigmoidVal = hardSigmoid(input)
                val output = input * hardSigmoidVal
                values[node.output[0]] = output
            }
            "Resize" -> {
                val input = get(node.input[0])
                if (input.rank() != 4) {
                    bail("Unsupported rank for nearest resize: ${input.rank()}")
                }
                val scales = if (node.input.size > 2 && node.input[2].isNotEmpty()) get(node.input[2]) else null
                val sizes = if (node.input.size > 3 && node.input[3].isNotEmpty()) get(node.input[3]) else null
                val outputDims = when {
                    scales != null && sizes != null -> bail("Scales and sizes cannot both be set for Resize operation")
                    scales != null -> {
                        val scaleValues = scales.toVec1<Float>()
                        input.dims().mapIndexed { i, d -> (d.toFloat() * scaleValues[i]).toInt() }
                    }
                    sizes != null -> sizes.toVec1<Long>().map { it.toInt() }
                    else -> bail("Either scales or sizes should be present")
                }
                val coordMode = getAttrOpt<String>(node, "coordinate_transformation_mode") ?: "half_pixel"
                val mode = getAttrOpt<String>(node, "mode") ?: "nearest"
                val nearestMode = getAttrOpt<String>(node, "nearest_mode") ?: "round_prefer_floor"
                if (mode != "nearest") {
                    bail("Unsupported resize mode: $mode")
                }
                if (nearestMode != "floor") {
                    bail("Unsupported nearest_mode for resize: $nearestMode")
                }
                if (coordMode != "asymmetric") {
                    bail("Unsupported coordinate_transformation_mode for resize: $coordMode")
                }
                val h = outputDims[2]
                val w = outputDims[3]
                val output = input.upsampleNearest2d(h, w)
                values[node.output[0]] = output
            }
            "Trilu" -> {
                val input = get(node.input[0])
                val k = if (node.input.size > 1 && node.input[1].isNotEmpty()) {
                    toVec0Flexible<Long>(get(node.input[1]))
                } else {
                    0L
                }
                val upper = getAttrOpt<Long>(node, "upper") ?: 1L
                val dims = input.dims()
                if (dims.size < 2) {
                    bail("Trilu expects input with at least 2 dimensions: $dims")
                }
                val n = dims[dims.size - 2]
                val m = dims[dims.size - 1]
                val maxDim = maxOf(n, m)
                val mask = if (k != 0L) {
                    val data = UIntArray(n * m)
                    for (i in 0 until n) {
                        for (j in 0 until m) {
                            if ((upper != 0L && j.toLong() >= i.toLong() + k) ||
                                (upper == 0L && j.toLong() <= i.toLong() + k)
                            ) {
                                data[i * m + j] = 1u
                            }
                        }
                    }
                    Tensor.fromSlice(data, Shape(n, m), input.device()).toDtype(input.dtype())
                } else if (upper == 0L) {
                    Tensor.tril2(maxDim, input.dtype(), input.device())
                } else {
                    Tensor.triu2(maxDim, input.dtype(), input.device())
                }
                val finalMask = if (n != m) {
                    mask.narrow(0, 0, n).narrow(1, 0, m)
                } else {
                    mask
                }
                val output = input * finalMask
                values[node.output[0]] = output
            }
            "ScatterND" -> {
                val data = get(node.input[0])
                val indices = get(node.input[1]).toDtype(DType.I64)
                val updates = get(node.input[2])
                val reduction = getAttrOpt<String>(node, "reduction") ?: "none"
                val indicesShape = indices.dims()
                val dataShape = data.dims()
                val k = indicesShape.last()
                if (k > data.rank()) {
                    bail("ScatterND expects k (indices.shape[-1]) to be at most the rank of data")
                }
                val numUpdates = indicesShape.subList(0, indicesShape.size - 1).fold(1) { acc, v -> acc * v }
                val flatIndices = if (indices.rank() == 1 && k == 1) {
                    indices.unsqueeze(0)
                } else {
                    indices.reshape(Shape(numUpdates, k))
                }
                val updateElementShape = if (k < dataShape.size) {
                    dataShape.subList(k, dataShape.size)
                } else {
                    emptyList()
                }
                val expectedUpdatesShape = ArrayList<Int>(indicesShape.size - 1 + updateElementShape.size)
                expectedUpdatesShape.addAll(indicesShape.subList(0, indicesShape.size - 1))
                expectedUpdatesShape.addAll(updateElementShape)

                val updatesFinal = if (updates.dims() != expectedUpdatesShape) {
                    if (updates.rank() == 0) {
                        val targetShape = ArrayList<Int>(1 + updateElementShape.size)
                        targetShape.add(numUpdates)
                        targetShape.addAll(updateElementShape)
                        updates.broadcastAs(Shape(targetShape))
                    } else {
                        val flatShape = Shape(numUpdates, updateElementShape.fold(1) { acc, v -> acc * v })
                        val flattened = updates.reshape(flatShape)
                        flattened.reshape(Shape(expectedUpdatesShape))
                    }
                } else {
                    updates
                }

                var flatOutput = data.flattenAll()
                val flatUpdates = if (updateElementShape.isEmpty()) {
                    updatesFinal.reshape(Shape(numUpdates))
                } else {
                    val product = updateElementShape.fold(1) { acc, v -> acc * v }
                    updatesFinal.reshape(Shape(numUpdates, product))
                }

                val strides = ArrayList<Int>()
                strides.add(1)
                for (i in (0 until dataShape.size - 1).reversed()) {
                    strides.add(strides.last() * dataShape[i + 1])
                }
                strides.reverse()

                for (i in 0 until numUpdates) {
                    val indexSlice = flatIndices.narrow(0, i, 1)
                    val indicesVec = indexSlice.squeeze(0).toVec1<Long>()
                    var flatIdx = 0
                    for ((dim, idx) in indicesVec.withIndex()) {
                        val dimSize = dataShape[dim].toLong()
                        val normIdx = if (idx < 0) dimSize + idx else idx
                        if (normIdx < 0 || normIdx >= dimSize) {
                            bail("Index $idx out of bounds for dimension $dim with size $dimSize")
                        }
                        flatIdx += normIdx.toInt() * strides[dim]
                    }

                    val updateSlice = if (updateElementShape.isEmpty()) {
                        flatUpdates.narrow(0, i, 1).squeeze(0)
                    } else {
                        flatUpdates.narrow(0, i, 1)
                    }

                    when (reduction) {
                        "add" -> {
                            if (updateElementShape.isEmpty()) {
                                val existing = flatOutput.narrow(0, flatIdx, 1)
                                val newValue = existing.add(updateSlice.unsqueeze(0))
                                flatOutput = flatOutput.sliceScatter(newValue, 0, flatIdx)
                            } else {
                                val sliceSize = updateElementShape.fold(1) { acc, v -> acc * v }
                                val existing = flatOutput.narrow(0, flatIdx, sliceSize)
                                val newValue = existing.add(updateSlice)
                                flatOutput = flatOutput.sliceScatter(newValue, 0, flatIdx)
                            }
                        }
                        else -> {
                            if (updateElementShape.isEmpty()) {
                                flatOutput = flatOutput.sliceScatter(updateSlice.unsqueeze(0), 0, flatIdx)
                            } else {
                                flatOutput = flatOutput.sliceScatter(updateSlice, 0, flatIdx)
                            }
                        }
                    }
                }

                val output = flatOutput.reshape(Shape(dataShape))
                values[node.output[0]] = output
            }
            else -> bail("unsupported op_type ${node.opType} for op $node")
        }
    }

    val resultMap = LinkedHashMap<String, Value>()
    for (output in graph.output) {
        val value = values.remove(output.name) ?: bail("cannot find output ${output.name}")
        resultMap[output.name] = value
    }
    return resultMap
}

public fun broadcastShape(shapeA: List<Int>, shapeB: List<Int>): List<Int> {
    val (longest, shortest) = if (shapeA.size > shapeB.size) {
        Pair(shapeA, shapeB)
    } else {
        Pair(shapeB, shapeA)
    }
    val diff = longest.size - shortest.size
    val targetShape = ArrayList<Int>(longest.size)
    for (i in 0 until diff) {
        targetShape.add(longest[i])
    }
    for (i in 0 until shortest.size) {
        val dim1 = longest[diff + i]
        val dim2 = shortest[i]
        if (dim1 == dim2 || dim2 == 1 || dim1 == 1) {
            targetShape.add(maxOf(dim1, dim2))
        } else {
            bail("Expand: incompatible shapes for broadcast, $shapeA and $shapeB")
        }
    }
    return targetShape
}

public fun broadcastShapeFromMany(shapes: List<List<Int>>): List<Int> {
    if (shapes.isEmpty()) {
        return emptyList()
    }
    var shapeOut = shapes[0]
    for (i in 1 until shapes.size) {
        shapeOut = broadcastShape(shapeOut, shapes[i])
    }
    return shapeOut
}

/**
 * Extract scalar from tensors that may be wrapped in extra dimensions.
 * Some ONNX exports use shape [1] or [1,1] where scalars are expected.
 * Only accepts single-element tensors; multi-element tensors still fail.
 */
public inline fun <reified T> toScalarFlexible(t: Tensor): T {
    return if (t.rank() > 0 && t.elemCount() == 1) {
        t.flattenAll().get(0).toScalar<T>()
    } else {
        t.toScalar<T>()
    }
}

/**
 * Same as toScalarFlexible but returns via toVec0 for types that need it.
 */
public inline fun <reified T> toVec0Flexible(t: Tensor): T {
    return if (t.rank() > 0 && t.elemCount() == 1) {
        t.flattenAll().get(0).toVec0<T>()
    } else {
        t.toVec0<T>()
    }
}

private fun arange(start: Long, end: Long, device: Device): Tensor = arangeStep(start, end, 1L, device)

private fun arangeStep(start: Long, end: Long, step: Long, device: Device): Tensor {
    if (step == 0L) bail("step cannot be zero")
    val count = if (step > 0L) {
        if (start < end) ((end - start - 1L) / step + 1L).toInt() else 0
    } else {
        if (start > end) ((start - end - 1L) / (-step) + 1L).toInt() else 0
    }
    val data = LongArray(count)
    var curr = start
    for (i in 0 until count) {
        data[i] = curr
        curr += step
    }
    return Tensor.fromSlice(data, Shape(count), device)
}

private fun arangeStep(start: Double, end: Double, step: Double, device: Device): Tensor {
    if (step == 0.0) bail("step cannot be zero")
    val data = ArrayList<Double>()
    var curr = start
    if (step > 0.0) {
        while (curr < end) {
            data.add(curr)
            curr += step
        }
    } else {
        while (curr > end) {
            data.add(curr)
            curr += step
        }
    }
    return Tensor.fromSlice(data.toDoubleArray(), Shape(data.size), device)
}
