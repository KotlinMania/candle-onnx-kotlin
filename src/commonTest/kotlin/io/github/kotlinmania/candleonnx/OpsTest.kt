// port-lint: tests tests/ops.rs
package io.github.kotlinmania.candleonnx

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.NdArray
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.toVec2Round
import io.github.kotlinmania.candleonnx.onnx.AttributeProto
import io.github.kotlinmania.candleonnx.onnx.GraphProto
import io.github.kotlinmania.candleonnx.onnx.ModelProto
import io.github.kotlinmania.candleonnx.onnx.NodeProto
import io.github.kotlinmania.candleonnx.onnx.TensorProto
import io.github.kotlinmania.candleonnx.onnx.TensorShapeProto
import io.github.kotlinmania.candleonnx.onnx.TypeProto
import io.github.kotlinmania.candleonnx.onnx.ValueInfoProto
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

private const val INPUT_X: String = "x"
private const val INPUT_Y: String = "y"
private const val INPUT_A: String = "a"
private const val OUTPUT_Z: String = "z"

private fun createModelProtoWithGraph(graph: GraphProto?): ModelProto {
    return ModelProto(graph = graph)
}

private fun reshapeGraph(allowzero: Long?): ModelProto {
    val attribute = if (allowzero != null) {
        mutableListOf(
            AttributeProto(
                name = "allowzero",
                refAttrName = "allowzero",
                i = allowzero,
                docString = "allowzero",
                type = AttributeProto.AttributeType.INT,
            )
        )
    } else {
        mutableListOf()
    }
    return createModelProtoWithGraph(
        GraphProto(
            node = mutableListOf(
                NodeProto(
                    opType = "Reshape",
                    domain = "",
                    attribute = attribute,
                    input = mutableListOf(INPUT_X, INPUT_Y),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )
            ),
            name = "",
            initializer = mutableListOf(),
            input = mutableListOf(
                ValueInfoProto(name = INPUT_X, docString = "", type = null),
                ValueInfoProto(name = INPUT_Y, docString = "", type = null),
            ),
            output = mutableListOf(
                ValueInfoProto(name = OUTPUT_Z, docString = "", type = null),
            ),
            valueInfo = mutableListOf(),
            docString = "",
            sparseInitializer = mutableListOf(),
            quantizationAnnotation = mutableListOf(),
        )
    )
}

private fun runReshape(allowzero: Long?, x: Tensor, shape: List<Long>): Tensor {
    val len = shape.size
    val y = Tensor.fromVec(shape, Shape(len), Device.Cpu)
    val inputs = mutableMapOf<String, Tensor>()
    inputs[INPUT_X] = x
    inputs[INPUT_Y] = y
    val eval = simpleEval(reshapeGraph(allowzero), inputs)
    return eval[OUTPUT_Z] ?: error("Output 'z' not found")
}

private fun makeGraphHelper(
    opName: String,
    inputs: List<String>,
    outputs: List<String>,
    attribs: MutableList<AttributeProto>,
): ModelProto {
    return createModelProtoWithGraph(
        GraphProto(
            node = mutableListOf(
                NodeProto(
                    opType = opName,
                    domain = "",
                    attribute = attribs,
                    input = inputs.toMutableList(),
                    output = outputs.toMutableList(),
                    name = "",
                    docString = "",
                )
            ),
            name = "",
            initializer = mutableListOf(),
            input = inputs.map { name ->
                ValueInfoProto(name = name, docString = "", type = null)
            }.toMutableList(),
            output = outputs.map { name ->
                ValueInfoProto(name = name, docString = "", type = null)
            }.toMutableList(),
            valueInfo = mutableListOf(),
            docString = "",
            sparseInitializer = mutableListOf(),
            quantizationAnnotation = mutableListOf(),
        )
    )
}

private fun makeSplitGraphHelper(inputs: List<String>, outputs: List<String>, axis: Long): ModelProto {
    val attribs = mutableListOf(
        AttributeProto(
            name = "axis",
            refAttrName = "axis",
            i = axis,
            docString = "axis",
            type = AttributeProto.AttributeType.INT,
        )
    )
    return makeGraphHelper("Split", inputs, outputs, attribs)
}

private fun makeReduceSumGraphHelper(
    inputs: List<String>,
    outputs: List<String>,
    keepdims: Long?,
    noopWithEmptyAxes: Long?,
): ModelProto {
    val attribs = mutableListOf<AttributeProto>()
    if (keepdims != null) {
        attribs.add(
            AttributeProto(
                name = "keepdims",
                refAttrName = "keepdims",
                i = keepdims,
                docString = "keepdims",
                type = AttributeProto.AttributeType.INT,
            )
        )
    }
    if (noopWithEmptyAxes != null) {
        attribs.add(
            AttributeProto(
                name = "noop_with_empty_axes",
                refAttrName = "noop_with_empty_axes",
                i = noopWithEmptyAxes,
                docString = "noop_with_empty_axes",
                type = AttributeProto.AttributeType.INTS,
            )
        )
    }
    return makeGraphHelper("ReduceSum", inputs, outputs, attribs)
}

class OpsTest {
    @Test
    fun testEvaluationFailsWithoutDefinedGraph() {
        val manualGraph = createModelProtoWithGraph(null);
            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            when (simpleEval(manualGraph, inputs) ) {
                Err(err) => assertEquals("no graph defined in proto", err),
                Ok(_) => error("Expected an error due to undefined graph"),
            }
    }

    @Test
    fun testAddOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Add",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X, INPUT_Y),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs.insert(INPUT_X, Tensor.new(doubleArrayOf(2.0), Device.Cpu));
            inputs.insert(INPUT_Y, Tensor.new(doubleArrayOf(2.0), Device.Cpu));

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
            val first = z.toVec1<Double>()[0];
            assertEquals(4.0f64, first)
    }

    @Test
    fun testSubOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Sub",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X, INPUT_Y),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs.insert(INPUT_X, Tensor.new(doubleArrayOf(2.0), Device.Cpu));
            inputs.insert(INPUT_Y, Tensor.new(doubleArrayOf(2.0), Device.Cpu));

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
            val first = z.toVec1<Double>()[0];
            assertEquals(0.0f64, first)
    }

    @Test
    fun testMulOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Mul",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X, INPUT_Y),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs.insert(INPUT_X, Tensor.new(doubleArrayOf(2.0), Device.Cpu));
            inputs.insert(INPUT_Y, Tensor.new(doubleArrayOf(2.0), Device.Cpu));

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
            val first = z.toVec1<Double>()[0];
            assertEquals(4.0f64, first)
    }

    @Test
    fun testDivOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Div",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X, INPUT_Y),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs.insert(INPUT_X, Tensor.new(doubleArrayOf(2.0), Device.Cpu));
            inputs.insert(INPUT_Y, Tensor.new(doubleArrayOf(2.0), Device.Cpu));

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
            val first = z.toVec1<Double>()[0];
            assertEquals(1.0f64, first)
    }

    @Test
    fun testExpOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Exp",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));

            val x = Tensor.fromVec(mutableListOf(-1.0f32, 0.0f32, 1.0f32, 2.0f32), &[2, 2], Device.Cpu);

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            val results = z.toVec2<Float>();

            assertEquals(0.36787944f32, results[0][0])
            assertEquals(1.0f32, results[0][1])
            assertEquals(mutableListOf(kotlin.math.E.toFloat(), 7.389056f32), results[1])
    }

    @Test
    fun testEqualOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Equal",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X, INPUT_Y),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs.insert(INPUT_X, Tensor.new(doubleArrayOf(2.0), Device.Cpu));
            inputs.insert(INPUT_Y, Tensor.new(doubleArrayOf(2.0), Device.Cpu));

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
            val first = z.toDtype(DType.U8).toVec1<UByte>()[0];
            assertEquals(1, first)
    }

    @Test
    fun testNotOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Not",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs.insert(INPUT_X, Tensor.new(doubleArrayOf(0.0), Device.Cpu));

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
            val first = z.toDtype(DType.U8).toVec1<UByte>()[0];
            assertEquals(1, first)
    }

    @Test
    fun testMatmulOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "MatMul",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X, INPUT_Y),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs.insert(
                INPUT_X,
                Tensor.fromVec(
                    //
                    mutableListOf(1.0f32, 2.0f32, 3.0f32, 4.0f32),
                    &[2, 2],
                    Device.Cpu,
                )?,
            );
            inputs.insert(
                INPUT_Y,
                Tensor.fromVec(
                    //
                    mutableListOf(5.0f32, 6.0f32, 7.0f32, 8.0f32),
                    &[2, 2],
                    Device.Cpu,
                )?,
            );

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
            val results = z.toVec2<Float>();
            assertEquals(mutableListOf(mutableListOf(19.0, 22.0), mutableListOf(43.0, 50.0)), results)
    }

    @Test
    fun testReshapeOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Reshape",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X, INPUT_Y),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(
                    ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = INPUT_Y,
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));

            val x = Tensor.fromVec(
                //
                mutableListOf(1.0f32, 2.0f32, 3.0f32, 4.0f32),
                &[2, 2],
                Device.Cpu,
            );
            val y = Tensor.fromVec(
                //
                mutableListOf(4i64),
                &[1],
                Device.Cpu,
            );

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x
            inputs[INPUT_Y] = y

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            val results = z.toVec1<Float>();

            assertEquals(mutableListOf(1.0, 2.0, 3.0, 4.0), results)
    }

    @Test
    fun testReshapeAllowzeroKeepsALiteralZero() {
        val x = Tensor.fromVec(Vec::<f32>::new(), &[3, 0], Device.Cpu);
            val z = runReshape(1, x, mutableListOf(0));
            assertEquals(&[0], z.dims())
    }

    @Test
    fun testReshapeInfersMinusOnePastACopiedZero() {
        val x = Tensor.fromVec(
                (0..24).map(|i| i as f32).collect::<Vec<_>>(),
                &[2, 3, 4],
                Device.Cpu,
            );
            val z = runReshape(null, x, mutableListOf(2, 0, 1, -1));
            // The 0 copies 3, so the known product is 6 and the -1 is 24 / 6 = 4. Leaving the copied
            // dimension out of the product would infer 12 instead.
            assertEquals(&[2, 3, 1, 4], z.dims())
    }

    @Test
    fun testReshapeRejectsMultipleMinusOne() {
        val x = Tensor.fromVec(mutableListOf(5.0f32), &[1], Device.Cpu);
            assertTrue(runCatching { runReshape(null, x, mutableListOf(-1, -1)) }.isFailure)
    }

    @Test
    fun testReshapeAllowzeroRejectsZeroWithMinusOne() {
        val x = Tensor.fromVec(Vec::<f32>::new(), &[3, 0], Device.Cpu);
            assertTrue(runCatching { runReshape(1, x, mutableListOf(0, -1)) }.isFailure)
    }

    @Test
    fun testLogsoftmaxOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "LogSoftmax",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(
                    ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = INPUT_Y,
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));

            val x = Tensor.fromVec(
                //
                mutableListOf(1.0f32, 2.0f32, 3.0f32, 4.0f32),
                &[2, 2],
                Device.Cpu,
            );

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            // Without an axis attribute the reduction runs over the last dimension, and the
            // result is log(softmax(x)), not softmax(x).
            val results = toVec2Round(z, 4);

            assertEquals(mutableListOf(mutableListOf(-1.3133, -0.3133), mutableListOf(-1.3133, -0.3133)), results)
    }

    @Test
    fun testSoftmaxOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Softmax",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(
                    ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = INPUT_Y,
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));

            val x = Tensor.fromVec(
                //
                mutableListOf(1.0f32, 2.0f32, 3.0f32, 4.0f32),
                &[2, 2],
                Device.Cpu,
            );

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            val results = z.toVec2<Float>();

            assertEquals(mutableListOf(mutableListOf(0.26894143, 0.7310586), mutableListOf(0.26894143, 0.7310586)), results)
    }

    @Test
    fun testTransposeOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Transpose",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(
                    ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = INPUT_Y,
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));

            val x = Tensor.fromVec(
                //
                mutableListOf(1.0f32, 2.0f32, 3.0f32, 4.0f32),
                &[2, 2],
                Device.Cpu,
            );

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            val results = z.toVec2<Float>();

            assertEquals(mutableListOf(mutableListOf(1.0, 3.0), mutableListOf(2.0, 4.0)), results)
    }

    @Test
    fun testDropoutOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Dropout",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(
                    ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = INPUT_Y,
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x = Tensor.fromVec(
                //
                mutableListOf(1.0f32, 2.0f32, 3.0f32, 4.0f32),
                &[2, 2],
                Device.Cpu,
            );

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            val results = z.toVec2<Float>();

            assertEquals(mutableListOf(mutableListOf(1.0, 2.0), mutableListOf(3.0, 4.0)), results)
    }

    @Test
    fun testFlattenOperation() {
        val attAxis = AttributeProto(
                name = "axis",
                refAttrName = "axis",
                i = 0,
                docString = "axis",
                type = 2,
                f = 0.0,
                s = mutableListOf(),
                t = null,
                g = null,
                sparseTensor = null,
                tp = null,
                floats = mutableListOf(),
                ints = mutableListOf(),
                strings = mutableListOf(),
                tensors = mutableListOf(),
                graphs = mutableListOf(),
                sparseTensors = mutableListOf(),
                typeProtos = mutableListOf(),
            );
            val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Flatten",
                    domain = "",
                    attribute = mutableListOf(attAxis),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(
                    ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = INPUT_Y,
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x = Tensor.fromVec(
                mutableListOf(
                    1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32,
                ),
                &[2, 2, 2],
                Device.Cpu,
            );

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            val results = z.toVec2<Float>();

            assertEquals(mutableListOf(mutableListOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)), results)

            attAxis.i = 1;
            val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Flatten",
                    domain = "",
                    attribute = mutableListOf(attAxis),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(
                    ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = INPUT_Y,
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            val results = z.toVec2<Float>();

            assertEquals(mutableListOf(mutableListOf(1.0, 2.0, 3.0, 4.0), mutableListOf(5.0, 6.0, 7.0, 8.0)), results)
    }

    @Test
    fun testConstantOfShape() {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-31
            test(
                &[4i64, 3, 2],
                1.,
                &[
                    [[1., 1.], [1., 1.], [1., 1.]],
                    [[1., 1.], [1., 1.], [1., 1.]],
                    [[1., 1.], [1., 1.], [1., 1.]],
                    [[1., 1.], [1., 1.], [1., 1.]],
                ],
            );

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-31
            test(&[1i64], 0i64, &[0i64]);

            // "value" defaults to 0 f32
            test(&[4i64], null as Option<i64>, &[0., 0., 0., 0.]);

            fun test(
                input = impl NdArray,
                value = Option<impl NdArray>,
                expected: impl NdArray,
            ) {
                val attribute = mutableListOf();

                if val value = value {
                    val tensor = Tensor::new(value, Device.Cpu);

                    val (value, dataType) = when (tensor.dtype()) {
                        DType.U8 -> Pair(
                            tensor.toVec0<UByte>().toLeBytes(),
                            DataType::Uint8,
                        ),
                        DType.U32 -> Pair(
                            tensor.toVec0::<u32>()?.toLeBytes(),
                            DataType::Uint32,
                        ),
                        DType.I64 -> Pair(
                            tensor.toVec0<Long>().toLeBytes(),
                            DataType::Int64,
                        ),
                        DType.F32 -> Pair(
                            tensor.toVec0<Float>().toLeBytes(),
                            DataType::Float,
                        ),
                        DType.F64 -> Pair(
                            tensor.toVec0<Double>().toLeBytes(),
                            DataType::Double,
                        ),
                        else -> error("unsupported DType in test"),
                    };
                    val tensor = TensorProto(
                        dataType = dataType.into(),
                        dims = tensor.dims().iter().map(|v| *v as i64).collect(),
                        rawData = value,
                        segment = null,
                        floatData = mutableListOf(),
                        int32Data = mutableListOf(),
                        stringData = mutableListOf(),
                        int64Data = mutableListOf(),
                        name = "",
                        docString = "",
                        externalData = mutableListOf(),
                        dataLocation = 0,
                        doubleData = mutableListOf(),
                        uint64Data = mutableListOf(),
                    );

                    attribute.push(AttributeProto(
                        name = "value",
                        refAttrName = "value",
                        i = 0,
                        docString = "value",
                        type = AttributeProto.AttributeType.TENSOR,
                        f = 0.0,
                        s = mutableListOf(),
                        t = tensor,
                        g = null,
                        sparseTensor = null,
                        tp = null,
                        floats = mutableListOf(),
                        ints = mutableListOf(),
                        strings = mutableListOf(),
                        tensors = mutableListOf(),
                        graphs = mutableListOf(),
                        sparseTensors = mutableListOf(),
                        typeProtos = mutableListOf(),
                    ))
                }

                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "ConstantOfShape",
                        domain = "",
                        attribute,
                        input = mutableListOf(INPUT_X),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(INPUT_X, Tensor::new(input, Device.Cpu)?);

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval
                    [OUTPUT_Z]
                     ?: error("Output 'z' not found")
                    .toDtype(DType.F64);

                val expected = Tensor::new(expected, Device.Cpu)?.toDtype(DType.F64);
                when (expected.dims().size) {
                    0 -> assertEquals(expected.toVec0<Double>(), z.toVec0<Double>()),
                    1 -> assertEquals(expected.toVec1<Double>(), z.toVec1<Double>()),
                    2 -> assertEquals(expected.toVec2<Double>(), z.toVec2<Double>()),
                    3 -> assertEquals(expected.toVec3<Double>(), z.toVec3<Double>()),
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testUnsqueeze() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Unsqueeze",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X, INPUT_Y),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(ValueInfoProto(
                    name = INPUT_X,
                    docString = "",
                    type = null,
                )),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x = Tensor.fromVec(
                mutableListOf(
                    1.0f32, 2.0f32, //
                    3.0f32, 4.0f32, //
                ),
                &[2, 2],
                Device.Cpu,
            );
            val y = Tensor.fromVec(mutableListOf(-1i64), &[1], Device.Cpu);

            val inputs = mutableMapOf((INPUT_X, x), (INPUT_Y, y)]);

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
            assertEquals(&[2, 2, 1], z.dims())
            assertEquals(x.flattenAll()?.toVec1<Float>(), z.flattenAll()?.toVec1<Float>())
    }

    @Test
    fun testGatherOperation() {
        // test taken from https://onnx.ai/onnx/operators/onnxGather.html#summary.
            test(
                &[[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]],
                &[[0i64, 1], [1, 2]],
                0,
                &[[[1.0, 1.2], [2.3, 3.4]], [[2.3, 3.4], [4.5, 5.7]]],
            );

            // test taken from https://onnx.ai/onnx/operators/onnxGather.html#summary.
            test(
                &[[1.0, 1.2, 1.9], [2.3, 3.4, 3.9], [4.5, 5.7, 5.9]],
                &[[0i64, 2]],
                1,
                &[[[1.0, 1.9]], [[2.3, 3.9]], [[4.5, 5.9]]],
            );

            // all the tests below are generated from numpy.take, which works like
            // onnx's Gather operation.
            test(&[1.0, 2.0, 3.0, 4.0], 3i64, 0, 4.0);

            test(&[[1.0, 2.0, 3.0, 4.0]], 3i64, 1, &[4.0]);

            test(
                &[[1.0], [2.0], [3.0], [4.0]],
                &[3i64, 2],
                0,
                &[[4.0], [3.0]],
            );

            test(
                &[
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[9.0, 10.0], [11.0, 12.0]],
                    [[13.0, 14.0], [15.0, 16.0]],
                ],
                1i64,
                0,
                &[[5.0, 6.0], [7.0, 8.0]],
            );

            test(
                &[
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[9.0, 10.0], [11.0, 12.0]],
                    [[13.0, 14.0], [15.0, 16.0]],
                ],
                &[1i64, 0],
                0,
                &[[[5.0, 6.0], [7.0, 8.0]], [[1.0, 2.0], [3.0, 4.0]]],
            );

            fun test(
                data: impl NdArray,
                indices: impl NdArray,
                axis: i64,
                expected: impl NdArray,
            ) {
                val attAxis = AttributeProto(
                    name = "axis",
                    refAttrName = "axis",
                    i = axis,
                    docString = "axis",
                    type = 2,
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );

                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Gather",
                        domain = "",
                        attribute = mutableListOf(attAxis),
                        input = mutableListOf(INPUT_X, INPUT_Y),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(INPUT_X, Tensor::new(data, Device.Cpu)?);
                inputs.insert(INPUT_Y, Tensor::new(indices, Device.Cpu)?);

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

                val expected = Tensor::new(expected, Device.Cpu);
                when (expected.dims().size) {
                    0 -> assertEquals(expected.toVec0<Double>(), z.toVec0<Double>()),
                    1 -> assertEquals(expected.toVec1<Double>(), z.toVec1<Double>()),
                    2 -> assertEquals(expected.toVec2<Double>(), z.toVec2<Double>()),
                    3 -> assertEquals(expected.toVec3<Double>(), z.toVec3<Double>()),
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testGatherElements() {
        // all the tests below are verified against `torch.gather()`

            // Rank 1 index
            test(&[1.0, 2.0, 3.0, 4.0], &[3i64], 0, &[4.0]);

            // Rank 2 index
            test(&[[1.0, 2.0, 3.0, 4.0]], &[[3i64]], 1, &[[4.0]]);
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-57 gatherelements0
            test(
                &[[1., 2.], [3., 4.]],
                &[[0i64, 0], [1, 0]],
                1,
                &[[1., 1.], [4., 3.]],
            );
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-57 gatherelements1
            test(
                &[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                &[[1i64, 2, 0], [2, 0, 0]],
                0,
                &[[4., 8., 3.], [7., 2., 3.]],
            );
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-57 gatherelementsnegativeindices
            test(
                &[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                &[[-1_i64, -2, 0], [-2, 0, 0]],
                0,
                &[[7., 5., 3.], [4., 2., 3.]],
            );
            test(
                &[[1.0], [2.0], [3.0], [4.0]],
                &[[3i64], [2]],
                0,
                &[[4.], [3.]],
            );

            // Rank 3
            test(
                &[
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[9.0, 10.0], [11.0, 12.0]],
                    [[13.0, 14.0], [15.0, 16.0]],
                ],
                &[[[1i64]]],
                0,
                &[[[5.]]],
            );

            test(
                &[
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[9.0, 10.0], [11.0, 12.0]],
                    [[13.0, 14.0], [15.0, 16.0]],
                ],
                &[[[1i64]]],
                1,
                &[[[3.]]],
            );

            test(
                &[
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[9.0, 10.0], [11.0, 12.0]],
                    [[13.0, 14.0], [15.0, 16.0]],
                ],
                &[[[1i64], [0]]],
                2,
                &[[[2.], [3.]]],
            );

            // Error cases
            // Invalid index
            assertTrue(runCatching { test(&[[1.0, 2.0, 3.0, 4.0]], &[[3i64]], 0, &[[1., 2., 3., 4.]]) }.isFailure)
            // Invalid axis/ dim
            assertTrue(runCatching { test(&[[1.0, 2.0, 3.0, 4.0]], &[[3i64]], 2, &[[1., 2., 3., 4.]]) }.isFailure)
            // Invalid rank
            assertTrue(runCatching { test(&[[1.0, 2.0, 3.0, 4.0]], &[3i64], 0, &[[1.]]) }.isFailure)

            fun test(
                data: impl NdArray,
                indices: impl NdArray,
                axis: i64,
                expected: impl NdArray,
            ) {
                val attAxis = AttributeProto(
                    name = "axis",
                    refAttrName = "axis",
                    i = axis,
                    docString = "axis",
                    type = 2,
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );

                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "GatherElements",
                        domain = "",
                        attribute = mutableListOf(attAxis),
                        input = mutableListOf(INPUT_X, INPUT_Y),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(INPUT_X, Tensor::new(data, Device.Cpu)?);
                inputs.insert(INPUT_Y, Tensor::new(indices, Device.Cpu)?);

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
                val expected = Tensor::new(expected, Device.Cpu);
                when (expected.dims().size) {
                    0 -> assertEquals(expected.toVec0<Double>(), z.toVec0<Double>()),
                    1 -> assertEquals(expected.toVec1<Double>(), z.toVec1<Double>()),
                    2 -> assertEquals(expected.toVec2<Double>(), z.toVec2<Double>()),
                    3 -> assertEquals(expected.toVec3<Double>(), z.toVec3<Double>()),
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testSizeOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Size",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(ValueInfoProto(
                    name = INPUT_X,
                    docString = "",
                    type = null,
                )),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x = Tensor.fromVec(mutableListOf(1.0f32, 2.0f32, 3.0f32, 4.0f32), &[2, 2], Device.Cpu);

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
            val results = z.toScalar::<i64>();
            assertEquals(4, results)
    }

    @Test
    fun testShapeOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Shape",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(ValueInfoProto(
                    name = INPUT_X,
                    docString = "",
                    type = null,
                )),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x = Tensor.fromVec(mutableListOf(1.0f32, 2.0f32, 3.0f32, 4.0f32), &[2, 2], Device.Cpu);

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
            val results = z.toVec1<Long>();
            assertEquals(mutableListOf(2, 2), results)
    }

    @Test
    fun testAbsOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Abs",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(
                    ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = INPUT_Y,
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x = Tensor.fromVec(
                mutableListOf(-1.0f32, 2.0f32, -3.0f32, 4.0f32),
                &[2, 2],
                Device.Cpu,
            );

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            val results = z.toVec2<Float>();

            assertEquals(mutableListOf(mutableListOf(1.0, 2.0), mutableListOf(3.0, 4.0)), results)
    }

    @Test
    fun testCosOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Cos",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(
                    ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = INPUT_Y,
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x = Tensor.fromVec(mutableListOf(0.0f32, 1.0f32, 2.0f32, 3.0f32), &[2, 2], Device.Cpu);

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
            assertEquals([[1.0, 0.5403], [-0.4161, -0.99]], toVec2Round(z, 4))
    }

    @Test
    fun testSinOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Sin",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(
                    ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = INPUT_Y,
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x = Tensor.fromVec(mutableListOf(0.0f32, 1.0f32, 2.0f32, 3.0f32), &[2, 2], Device.Cpu);
            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x
            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)
            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
            assertEquals([[0.0, 0.8415], [0.9093, 0.1411]], toVec2Round(z, 4))
    }

    @Test
    fun testNegOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Neg",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(
                    ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = INPUT_Y,
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x = Tensor.fromVec(mutableListOf(1.0f32, 2.0f32, 3.0f32, 4.0f32), &[2, 2], Device.Cpu);

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            val results = z.toVec2<Float>();

            assertEquals(mutableListOf(mutableListOf(-1.0, -2.0), mutableListOf(-3.0, -4.0)), results)
    }

    @Test
    fun testTanhOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Tanh",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(
                    ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = INPUT_Y,
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x = Tensor.fromVec(mutableListOf(0.0f32, 1.0f32, 2.0f32, 3.0f32), &[2, 2], Device.Cpu);

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            val results = z.toVec2<Float>();

            assertEquals(mutableListOf(mutableListOf(0.0, 0.7615942), mutableListOf(0.9640276, 0.9950548)), results)
    }

    @Test
    fun testSigmoidOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Sigmoid",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(
                    ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = INPUT_Y,
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x = Tensor.fromVec(mutableListOf(0.0f32, 1.0f32, 2.0f32, 3.0f32), &[2, 2], Device.Cpu);

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            val results = z.toVec2<Float>();

            assertEquals(mutableListOf(mutableListOf(0.5, 0.7310586), mutableListOf(0.880797, 0.95257413)), results)
    }

    @Test
    fun testGeluOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Gelu",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(
                    ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = INPUT_Y,
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x = Tensor.fromVec(mutableListOf(0.0f32, 1.0f32, 2.0f32, 3.0f32), &[2, 2], Device.Cpu);

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            val results = z.toVec2<Float>();

            assertEquals(mutableListOf(mutableListOf(0.0, 0.8413448), mutableListOf(1.9544997, 2.9959502)), results)
    }

    @Test
    fun testReluOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Relu",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(ValueInfoProto(
                    name = INPUT_X,
                    docString = "",
                    type = null,
                )),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x = Tensor.fromVec(
                mutableListOf(-1.0f32, 1.0f32, -2.0f32, 3.0f32),
                &[2, 2],
                Device.Cpu,
            );

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            val results = z.toVec2<Float>();

            assertEquals(mutableListOf(mutableListOf(0.0, 1.0), mutableListOf(0.0, 3.0)), results)
    }

    @Test
    fun testPreluOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "PRelu",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X, INPUT_Y),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(
                    ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = INPUT_Y,
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x: Tensor = Tensor.fromVec(
                mutableListOf(-1.0f32, 1.0f32, -2.0f32, 3.0f32),
                &[2, 2],
                Device.Cpu,
            );

            val y: Tensor = Tensor.fromVec(mutableListOf(1.0f32, 1.1f32, 1.2f32, 1.3f32), &[2, 2], Device.Cpu);

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x
            inputs[INPUT_Y] = y

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
            val results = z.toVec2<Float>();
            assertEquals(mutableListOf(mutableListOf(-1.0, 1.0), mutableListOf(-2.4, 3.0)), results)
    }

    @Test
    fun testReduceMax() {
        // Tests with random data generated with `np.random.uniform`
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-119 boolinputs
            // No special treatment required for bool
            // `np.maximum.reduce(data, axis=axes, keepdims=True)`
            test(
                &[[1u.toUByte(), 1], [1, 0], [0, 1], [0, 0]],
                mutableListOf(1),
                1,
                null,
                &[[1u.toUByte()], [1], [1], [0]],
                false,
            );

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-119 defaultaxeskeepdims
            // `np.maximum.reduce(data, axis=null, keepdims=True)`
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                null,
                1,
                null,
                &[[[60.]]],
                false,
            );
            // same as above but with random
            test(
                &[
                    [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
                    [[6.304022, 4.939862], [4.5435624, 3.072864]],
                    [[-2.5058026, 8.008944], [9.587318, -8.794852]],
                ],
                null,
                1,
                null,
                &[[[9.587318]]],
                false,
            );

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-119 defaultaxesdonotkeepdims
            // `np.maximum.reduce(data, axis=null, keepdims=False)`
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                null,
                0,
                null,
                60.,
                false,
            );
            // same as above but with random
            // `np.maximum.reduce(data, axis=null, keepdims=False)`
            test(
                &[
                    [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
                    [[6.304022, 4.939862], [4.5435624, 3.072864]],
                    [[-2.5058026, 8.008944], [9.587318, -8.794852]],
                ],
                null,
                0,
                null,
                9.587318,
                false,
            );

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-119 keepdims
            // `np.maximum.reduce(data, axis=tuple(axes), keepdims=True)`
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(1),
                1,
                null,
                &[[[20., 2.]], [[40., 2.]], [[60., 2.]]],
                false,
            );
            // keepdims with random data
            // `np.maximum.reduce(data, axis=tuple(axes), keepdims=True)`
            test(
                &[
                    [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
                    [[6.304022, 4.939862], [4.5435624, 3.072864]],
                    [[-2.5058026, 8.008944], [9.587318, -8.794852]],
                ],
                mutableListOf(1),
                1,
                null,
                &[
                    [[-7.318765, 7.2374434]],
                    [[6.304022, 4.939862]],
                    [[9.587318, 8.008944]],
                ],
                false,
            );

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-119 negativeaxeskeepdims
            // axes = np.array([-1], dtype=np.int64)
            // `np.maximum.reduce(data, axis=tuple(axes), keepdims=True)`
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(-1),
                1,
                null,
                &[[[5.], [20.]], [[30.], [40.]], [[55.], [60.]]],
                false,
            );
            // axes = np.array([-2], dtype=np.int64)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(-2),
                1,
                null,
                &[[[20., 2.]], [[40., 2.]], [[60., 2.]]],
                false,
            );
            // with random
            test(
                &[
                    [[-4.1676497, -2.7603748], [-4.5138783, -0.762791]],
                    [[-6.3792877, 7.1619177], [-9.958144, 6.3753467]],
                    [[9.046973, 3.4554052], [-5.4674335, 5.4642754]],
                ],
                mutableListOf(-2),
                1,
                null,
                &[
                    [[-4.1676497, -0.762791]],
                    [[-6.3792877, 7.1619177]],
                    [[9.046973, 5.4642754]],
                ],
                false,
            );

            // Multiple axes - keepdims=1 (true)
            // axes = np.array([0, 1], dtype=np.int64)
            // np.maximum.reduce(data, axis=tuple(axes), keepdims=True)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(0, 1),
                1,
                null,
                &[[[60., 2.]]],
                false,
            );
            // axes = np.array([0, 2], dtype=np.int64)
            // np.maximum.reduce(data, axis=tuple(axes), keepdims=True)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(0, 2),
                1,
                null,
                &[[[55.], [60.]]],
                false,
            );
            // axes = np.array([2, 1], dtype=np.int64)
            // np.maximum.reduce(data, axis=tuple(axes), keepdims=True)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(2, 1),
                1,
                null,
                &[[[20.]], [[40.]], [[60.]]],
                false,
            );
            // axes = np.array([2, 0, 1], dtype=np.int64)
            // np.maximum.reduce(data, axis=tuple(axes), keepdims=True)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(2, 0, 1),
                1,
                null,
                &[[[60.]]],
                false,
            );
            // Multiple axes - keepdims=0 (false)
            // axes = np.array([0, 1], dtype=np.int64)
            // np.maximum.reduce(data, axis=tuple(axes), keepdims=False)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(0, 1),
                0,
                null,
                &[60., 2.],
                false,
            );
            // axes = np.array([0, 2], dtype=np.int64)
            // np.maximum.reduce(data, axis=tuple(axes), keepdims=False)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(0, 2),
                0,
                null,
                &[55., 60.],
                false,
            );
            // axes = np.array([2, 1], dtype=np.int64)
            // np.maximum.reduce(data, axis=tuple(axes), keepdims=False)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(2, 1),
                0,
                null,
                &[20., 40., 60.],
                false,
            );
            // axes = np.array([2, 0, 1], dtype=np.int64)
            // np.maximum.reduce(data, axis=tuple(axes), keepdims=False)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(2, 0, 1),
                0,
                null,
                60.,
                false,
            );

            // Multiple axes - negative `axes` - keepdims=1 (true)
            // axes = np.array([-1, 0, 1], dtype=np.int64)
            // np.maximum.reduce(data, axis=tuple(axes), keepdims=True)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(-1, 0, 1),
                1,
                null,
                &[[[60.]]],
                false,
            );
            // Multiple axes - negative `axes` - keepdims=0 (false)
            // axes = np.array([-1, 0, 1], dtype=np.int64)
            // np.maximum.reduce(data, axis=tuple(axes), keepdims=True)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(-1, 0, 1),
                0,
                null,
                60.,
                false,
            );

            // `noopwithemptyaxes = true (1)` should yield tensor equivalent to the input tensor
            test(
                &[
                    [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
                    [[6.304022, 4.939862], [4.5435624, 3.072864]],
                    [[-2.5058026, 8.008944], [9.587318, -8.794852]],
                ],
                null,
                0,
                1,
                &[
                    [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
                    [[6.304022, 4.939862], [4.5435624, 3.072864]],
                    [[-2.5058026, 8.008944], [9.587318, -8.794852]],
                ],
                false,
            );

            // Rank-0 arrays are also valid
            test(42., null, 0, null, 42., false);
            test(42., null, 1, null, 42., false);

            // Negative test - expect error
            // axes = np.array([-2, 0, 1], dtype=np.int64)
            // np.maximum.reduce(data, axis=tuple(axes), keepdims=True)
            // Should error out with `duplicate value in "axes"`
            assertTrue(runCatching { test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(-2, 0, 1),
                1,
                null,
                &[[[60.]]],
                false
            )
             }.isFailure)

            // Negative test - expect error
            // Should error out on empty set
            assertTrue(runCatching { test(&[[1u.toUByte(); 0]], mutableListOf(-2, 0, 1), 1, null, &[0.], false) }.isFailure)

            // Backward compatibility
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(-1, 0, 1),
                0,
                null,
                60.,
                true,
            );

            fun test(
                data: impl NdArray,
                axes: Option<Vec<i64>>,
                keepdims: i64,
                noopWithEmptyAxes: Option<i64>,
                expected: impl NdArray,
                backwardComp: bool,
            ) {
                val hasAxes = axes.isSome();

                val attKeepdims = AttributeProto(
                    name = "keepdims",
                    refAttrName = "keepdims",
                    i = keepdims,
                    docString = "keepdims",
                    type = 2,
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );

                val attribute = mutableListOf(attKeepdims);
                if val noop = noopWithEmptyAxes {
                    if !hasAxes {
                        val attNoOpEmptyAxes = AttributeProto(
                            name = "noopWithEmptyAxes",
                            refAttrName = "noopWithEmptyAxes",
                            i = noop,
                            docString = "noopWithEmptyAxes",
                            type = 2,
                            f = 0.0,
                            s = mutableListOf(),
                            t = null,
                            g = null,
                            sparseTensor = null,
                            tp = null,
                            floats = mutableListOf(),
                            ints = mutableListOf(),
                            strings = mutableListOf(),
                            tensors = mutableListOf(),
                            graphs = mutableListOf(),
                            sparseTensors = mutableListOf(),
                            typeProtos = mutableListOf(),
                        );

                        attribute.push(attNoOpEmptyAxes);
                    }
                }
                if hasAxes && backwardComp {
                    attribute.push(AttributeProto(
                        name = "axes",
                        refAttrName = "axes",
                        i = 0,
                        docString = "axes",
                        type = 7,
                        f = 0.0,
                        s = mutableListOf(),
                        t = null,
                        g = null,
                        sparseTensor = null,
                        tp = null,
                        floats = mutableListOf(),
                        ints = axes.unwrapOrDefault(),
                        strings = mutableListOf(),
                        tensors = mutableListOf(),
                        graphs = mutableListOf(),
                        sparseTensors = mutableListOf(),
                        typeProtos = mutableListOf(),
                    ));
                }

                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "ReduceMax",
                        domain = "",
                        attribute,
                        input = if hasAxes && !backwardComp {
                            mutableListOf(INPUT_X, INPUT_Y)
                        } else {
                            mutableListOf(INPUT_X)
                        },
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                val inputTensor = Tensor::new(data, Device.Cpu);
                val inputDtype = inputTensor.dtype();
                inputs[INPUT_X] = inputTensor
                if !backwardComp {
                    if val a = axes {
                        inputs.insert(INPUT_Y, Tensor::new(a, Device.Cpu)?);
                    }
                }

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

                val expected = Tensor::new(expected, Device.Cpu);

                when (expected.dims().size) {
                    0 -> {
                        if inputDtype == DType::U8 {
                            assertEquals(expected.toVec0<UByte>(), z.toVec0<UByte>())
                        } else {
                            assertEquals(expected.toVec0<Double>(), z.toVec0<Double>())
                        }
                    }
                    1 -> {
                        if inputDtype == DType::U8 {
                            assertEquals(expected.toVec1<UByte>(), z.toVec1<UByte>())
                        } else {
                            assertEquals(expected.toVec1<Double>(), z.toVec1<Double>())
                        }
                    }
                    2 -> {
                        if inputDtype == DType::U8 {
                            assertEquals(expected.toVec2<UByte>(), z.toVec2<UByte>())
                        } else {
                            assertEquals(expected.toVec2<Double>(), z.toVec2<Double>())
                        }
                    }
                    3 -> {
                        if inputDtype == DType::U8 {
                            assertEquals(expected.toVec3<UByte>(), z.toVec3<UByte>())
                        } else {
                            assertEquals(expected.toVec3<Double>(), z.toVec3<Double>())
                        }
                    }
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testReduceMin() {
        // Tests with random data generated with `np.random.uniform`
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-121 boolinputs
            // No special treatment required for bool
            // `np.minimum.reduce(data, axis=axes, keepdims=True)`
            test(
                &[[1u.toUByte(), 1], [1, 0], [0, 1], [0, 0]],
                mutableListOf(1),
                1,
                null,
                &[[1u.toUByte()], [0], [0], [0]],
                false,
            );

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-121 defaultaxeskeepdims
            // `np.minimum.reduce(data, axis=null, keepdims=True)`
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                null,
                1,
                null,
                &[[[1.]]],
                false,
            );
            // same as above but with random
            test(
                &[
                    [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
                    [[6.304022, 4.939862], [4.5435624, 3.072864]],
                    [[-2.5058026, 8.008944], [9.587318, -8.794852]],
                ],
                null,
                1,
                null,
                &[[[-8.794852]]],
                false,
            );

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-121 defaultaxesdonotkeepdims
            // `np.minimum.reduce(data, axis=null, keepdims=False)`
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                null,
                0,
                null,
                1.,
                false,
            );
            // same as above but with random
            // `np.minimum.reduce(data, axis=null, keepdims=False)`
            test(
                &[
                    [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
                    [[6.304022, 4.939862], [4.5435624, 3.072864]],
                    [[-2.5058026, 8.008944], [9.587318, -8.794852]],
                ],
                null,
                0,
                null,
                -8.794852,
                false,
            );

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-121 keepdims
            // `np.minimum.reduce(data, axis=tuple(axes), keepdims=True)`
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(1),
                1,
                null,
                &[[[5., 1.]], [[30., 1.]], [[55., 1.]]],
                false,
            );
            // keepdims with random data
            // `np.minimum.reduce(data, axis=tuple(axes), keepdims=True)`
            test(
                &[
                    [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
                    [[6.304022, 4.939862], [4.5435624, 3.072864]],
                    [[-2.5058026, 8.008944], [9.587318, -8.794852]],
                ],
                mutableListOf(1),
                1,
                null,
                &[
                    [[-7.648377, -5.4018507]],
                    [[4.5435624, 3.072864]],
                    [[-2.5058026, -8.794852]],
                ],
                false,
            );

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-121 negativeaxeskeepdims
            // axes = np.array([-1], dtype=np.int64)
            // `np.minimum.reduce(data, axis=tuple(axes), keepdims=True)`
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(-1),
                1,
                null,
                &[[[1.], [2.]], [[1.], [2.]], [[1.], [2.]]],
                false,
            );
            // axes = np.array([-2], dtype=np.int64)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(-2),
                1,
                null,
                &[[[5., 1.]], [[30., 1.]], [[55., 1.]]],
                false,
            );
            // with random
            test(
                &[
                    [[-4.1676497, -2.7603748], [-4.5138783, -0.762791]],
                    [[-6.3792877, 7.1619177], [-9.958144, 6.3753467]],
                    [[9.046973, 3.4554052], [-5.4674335, 5.4642754]],
                ],
                mutableListOf(-2),
                1,
                null,
                &[
                    [[-4.5138783, -2.7603748]],
                    [[-9.958144, 6.3753467]],
                    [[-5.4674335, 3.4554052]],
                ],
                false,
            );

            // Multiple axes - keepdims=1 (true)
            // axes = np.array([0, 1], dtype=np.int64)
            // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(0, 1),
                1,
                null,
                &[[[5., 1.]]],
                false,
            );
            // axes = np.array([0, 2], dtype=np.int64)
            // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(0, 2),
                1,
                null,
                &[[[1.], [2.]]],
                false,
            );
            // axes = np.array([2, 1], dtype=np.int64)
            // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(2, 1),
                1,
                null,
                &[[[1.]], [[1.]], [[1.]]],
                false,
            );
            // axes = np.array([2, 0, 1], dtype=np.int64)
            // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(2, 0, 1),
                1,
                null,
                &[[[1.]]],
                false,
            );
            // Multiple axes - keepdims=0 (false)
            // axes = np.array([0, 1], dtype=np.int64)
            // np.minimum.reduce(data, axis=tuple(axes), keepdims=False)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(0, 1),
                0,
                null,
                &[5., 1.],
                false,
            );
            // axes = np.array([0, 2], dtype=np.int64)
            // np.minimum.reduce(data, axis=tuple(axes), keepdims=False)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(0, 2),
                0,
                null,
                &[1., 2.],
                false,
            );
            // axes = np.array([2, 1], dtype=np.int64)
            // np.minimum.reduce(data, axis=tuple(axes), keepdims=False)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(2, 1),
                0,
                null,
                &[1., 1., 1.],
                false,
            );
            // axes = np.array([2, 0, 1], dtype=np.int64)
            // np.minimum.reduce(data, axis=tuple(axes), keepdims=False)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(2, 0, 1),
                0,
                null,
                1.,
                false,
            );

            // Multiple axes - negative `axes` - keepdims=1 (true)
            // axes = np.array([-1, 0, 1], dtype=np.int64)
            // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(-1, 0, 1),
                1,
                null,
                &[[[1.]]],
                false,
            );
            // Multiple axes - negative `axes` - keepdims=0 (false)
            // axes = np.array([-1, 0, 1], dtype=np.int64)
            // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(-1, 0, 1),
                0,
                null,
                1.,
                false,
            );

            // `noopwithemptyaxes = true (1)` should yield tensor equivalent to the input tensor
            test(
                &[
                    [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
                    [[6.304022, 4.939862], [4.5435624, 3.072864]],
                    [[-2.5058026, 8.008944], [9.587318, -8.794852]],
                ],
                null,
                0,
                1,
                &[
                    [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
                    [[6.304022, 4.939862], [4.5435624, 3.072864]],
                    [[-2.5058026, 8.008944], [9.587318, -8.794852]],
                ],
                false,
            );

            // Rank-0 tensors are also valid
            test(42., null, 0, null, 42., false);
            test(42., null, 1, null, 42., false);

            // Negative test - expect error
            // axes = np.array([-2, 0, 1], dtype=np.int64)
            // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
            // Should error out with `duplicate value in "axes"`
            assertTrue(runCatching { test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(-2, 0, 1),
                1,
                null,
                &[0.],
                false
            )
             }.isFailure)

            // Negative test - expect error
            // Should error out on empty set
            assertTrue(runCatching { test(&[[1u.toUByte(); 0]], mutableListOf(-2, 0, 1), 1, null, &[0.], false) }.isFailure)

            // Backward compatibility
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(-1, 0, 1),
                0,
                null,
                1.,
                true,
            );

            fun test(
                data: impl NdArray,
                axes: Option<Vec<i64>>,
                keepdims: i64,
                noopWithEmptyAxes: Option<i64>,
                expected: impl NdArray,
                backwardComp: bool,
            ) {
                val hasAxes = axes.isSome();

                val attKeepdims = AttributeProto(
                    name = "keepdims",
                    refAttrName = "keepdims",
                    i = keepdims,
                    docString = "keepdims",
                    type = 2,
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );

                val attribute = mutableListOf(attKeepdims);
                if val noop = noopWithEmptyAxes {
                    if !hasAxes {
                        val attNoOpEmptyAxes = AttributeProto(
                            name = "noopWithEmptyAxes",
                            refAttrName = "noopWithEmptyAxes",
                            i = noop,
                            docString = "noopWithEmptyAxes",
                            type = 2,
                            f = 0.0,
                            s = mutableListOf(),
                            t = null,
                            g = null,
                            sparseTensor = null,
                            tp = null,
                            floats = mutableListOf(),
                            ints = mutableListOf(),
                            strings = mutableListOf(),
                            tensors = mutableListOf(),
                            graphs = mutableListOf(),
                            sparseTensors = mutableListOf(),
                            typeProtos = mutableListOf(),
                        );

                        attribute.push(attNoOpEmptyAxes);
                    }
                }
                if hasAxes && backwardComp {
                    attribute.push(AttributeProto(
                        name = "axes",
                        refAttrName = "axes",
                        i = 0,
                        docString = "axes",
                        type = 7,
                        f = 0.0,
                        s = mutableListOf(),
                        t = null,
                        g = null,
                        sparseTensor = null,
                        tp = null,
                        floats = mutableListOf(),
                        ints = axes.unwrapOrDefault(),
                        strings = mutableListOf(),
                        tensors = mutableListOf(),
                        graphs = mutableListOf(),
                        sparseTensors = mutableListOf(),
                        typeProtos = mutableListOf(),
                    ));
                }

                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "ReduceMin",
                        domain = "",
                        attribute,
                        input = if hasAxes && !backwardComp {
                            mutableListOf(INPUT_X, INPUT_Y)
                        } else {
                            mutableListOf(INPUT_X)
                        },
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                val inputTensor = Tensor::new(data, Device.Cpu);
                val inputDtype = inputTensor.dtype();
                inputs[INPUT_X] = inputTensor
                if !backwardComp {
                    if val a = axes {
                        inputs.insert(INPUT_Y, Tensor::new(a, Device.Cpu)?);
                    }
                }

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

                val expected = Tensor::new(expected, Device.Cpu);

                when (expected.dims().size) {
                    0 -> {
                        if inputDtype == DType::U8 {
                            assertEquals(expected.toVec0<UByte>(), z.toVec0<UByte>())
                        } else {
                            assertEquals(expected.toVec0<Double>(), z.toVec0<Double>())
                        }
                    }
                    1 -> {
                        if inputDtype == DType::U8 {
                            assertEquals(expected.toVec1<UByte>(), z.toVec1<UByte>())
                        } else {
                            assertEquals(expected.toVec1<Double>(), z.toVec1<Double>())
                        }
                    }
                    2 -> {
                        if inputDtype == DType::U8 {
                            assertEquals(expected.toVec2<UByte>(), z.toVec2<UByte>())
                        } else {
                            assertEquals(expected.toVec2<Double>(), z.toVec2<Double>())
                        }
                    }
                    3 -> {
                        if inputDtype == DType::U8 {
                            assertEquals(expected.toVec3<UByte>(), z.toVec3<UByte>())
                        } else {
                            assertEquals(expected.toVec3<Double>(), z.toVec3<Double>())
                        }
                    }
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testReduceMean() {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 defaultaxeskeepdims
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                null,
                1,
                &[[[18.25]]],
            );

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 donokeepdims
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(1),
                0,
                &[[12.5, 1.5], [35.0, 1.5], [57.5, 1.5]],
            );

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 keepdims
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(1),
                1,
                &[[[12.5, 1.5]], [[35.0, 1.5]], [[57.5, 1.5]]],
            );

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 negativeaxeskeepdims
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(-2),
                1,
                &[[[12.5, 1.5]], [[35.0, 1.5]], [[57.5, 1.5]]],
            );

            // All the test data below was generated based on numpy's np.mean
            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(1, 2),
                0,
                &[7.0, 18.25, 29.5],
            );

            test(
                &[
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
                mutableListOf(1, 2),
                1,
                &[[[7.0]], [[18.25]], [[29.5]]],
            );

            test(&[1., 2., 3.], null, 1, &[2.0]);

            fun test(
                data: impl NdArray,
                axes: Option<Vec<i64>>,
                keepdims: i64,
                expected: impl NdArray,
            ) {
                val hasAxes = axes.isSome();

                val attAxes = AttributeProto(
                    name = "axes",
                    refAttrName = "axes",
                    i = 0,
                    docString = "axes",
                    type = 7,
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = axes.unwrapOrDefault(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );

                val attKeepdims = AttributeProto(
                    name = "keepdims",
                    refAttrName = "keepdims",
                    i = keepdims,
                    docString = "keepdims",
                    type = 2,
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );

                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "ReduceMean",
                        domain = "",
                        attribute = if hasAxes {
                            mutableListOf(attAxes, attKeepdims)
                        } else {
                            mutableListOf(attKeepdims)
                        },
                        input = mutableListOf(INPUT_X),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(INPUT_X, Tensor::new(data, Device.Cpu)?);

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

                val expected = Tensor::new(expected, Device.Cpu);
                when (expected.dims().size) {
                    0 -> assertEquals(expected.toVec0<Double>(), z.toVec0<Double>()),
                    1 -> assertEquals(expected.toVec1<Double>(), z.toVec1<Double>()),
                    2 -> assertEquals(expected.toVec2<Double>(), z.toVec2<Double>()),
                    3 -> assertEquals(expected.toVec3<Double>(), z.toVec3<Double>()),
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testSqrt() {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-155
            test(&[1., 4., 9.], &[1., 2., 3.]);

            fun test(data: impl NdArray, expected: impl NdArray) {
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Sqrt",
                        domain = "",
                        attribute = mutableListOf(),
                        input = mutableListOf(INPUT_X),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(INPUT_X, Tensor::new(data, Device.Cpu)?);

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

                val expected = Tensor::new(expected, Device.Cpu);
                when (expected.dims().size) {
                    0 -> assertEquals(expected.toVec0<Double>(), z.toVec0<Double>()),
                    1 -> assertEquals(expected.toVec1<Double>(), z.toVec1<Double>()),
                    2 -> assertEquals(expected.toVec2<Double>(), z.toVec2<Double>()),
                    3 -> assertEquals(expected.toVec3<Double>(), z.toVec3<Double>()),
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testRandomUniform() {
        test(mutableListOf(3, 2, 1, 4), null, null);
            test(mutableListOf(2, 2, 2, 2), -10.0, null);
            test(mutableListOf(2, 2, 2, 2), null, 10.0);
            test(mutableListOf(1, 2, 3, 4), -10.0, 10.0);

            fun test(shape = Vec<i64>, low: Option<f32>, high: Option<f32>) {
                val attLow = AttributeProto(
                    name = "low",
                    refAttrName = "low",
                    i = 0,
                    docString = "low",
                    type = 1, // FLOAT
                    f = low.unwrapOr(0.0),
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );
                val attHigh = AttributeProto(
                    name = "high",
                    refAttrName = "high",
                    i = 0,
                    docString = "high",
                    type = 1, // FLOAT
                    f = high.unwrapOr(1.0),
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );
                val attShape = AttributeProto(
                    name = "shape",
                    refAttrName = "shape",
                    i = 0,
                    docString = "shape",
                    type = 7, // INTS
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = shape,
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );
                val attDtype = AttributeProto(
                    name = "dtype",
                    refAttrName = "dtype",
                    i = 11, // DOUBLE
                    docString = "dtype",
                    type = 2, // INT
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );
                val attrs = {
                    val mutAttrs = mutableListOf(attShape, attDtype);
                    if low.isSome() {
                        mutAttrs.push(attLow);
                    }
                    if high.isSome() {
                        mutAttrs.push(attHigh);
                    }
                    mutAttrs
                };
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "RandomUniform",
                        domain = "",
                        attribute = attrs,
                        input = mutableListOf(),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));
                val eval = simpleEval(manualGraph, mutableMapOf());
                assertEquals(1, eval.size)
                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
                val min = z
                    .flattenAll()
                    .toVec1()
                    .intoIter()
                    .reduce(f64::min)
                    !!;
                val max = z
                    .flattenAll()
                    .toVec1()
                    .intoIter()
                    .reduce(f64::max)
                    !!;
                assertTrue(min >= low.unwrapOr(0.0).into())
                assertTrue(max <= high.unwrapOr(1.0).into())
                assertNe!(min, max);

            }
    }

    @Test
    fun testRandomNormal() {
        test(mutableListOf(3, 2, 1, 4), null, null);
            test(mutableListOf(2, 2, 2, 2), -10.0, null);
            test(mutableListOf(2, 2, 2, 2), null, 10.0);
            test(mutableListOf(1, 2, 3, 4), -10.0, 10.0);

            fun test(shape = Vec<i64>, mean: Option<f32>, scale: Option<f32>) {
                val attMean = AttributeProto(
                    name = "mean",
                    refAttrName = "mean",
                    i = 0,
                    docString = "mean",
                    type = 1, // FLOAT
                    f = mean.unwrapOr(0.0),
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );
                val attScale = AttributeProto(
                    name = "scale",
                    refAttrName = "scale",
                    i = 0,
                    docString = "scale",
                    type = 1, // FLOAT
                    f = scale.unwrapOr(1.0),
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );
                val attShape = AttributeProto(
                    name = "shape",
                    refAttrName = "shape",
                    i = 0,
                    docString = "shape",
                    type = 7, // INTS
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = shape,
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );
                val attDtype = AttributeProto(
                    name = "dtype",
                    refAttrName = "dtype",
                    i = 11, // DOUBLE
                    docString = "dtype",
                    type = 2, // INT
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );
                val attrs = {
                    val mutAttrs = mutableListOf(attShape, attDtype);
                    if mean.isSome() {
                        mutAttrs.push(attMean);
                    }
                    if scale.isSome() {
                        mutAttrs.push(attScale);
                    }
                    mutAttrs
                };
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "RandomNormal",
                        domain = "",
                        attribute = attrs,
                        input = mutableListOf(),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));
                val eval = simpleEval(manualGraph, mutableMapOf());
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
                val data = z.flattenAll()?.toVec1<Double>();

                // test if values are unique
                for (i, a) in data.iter().enumerate() {
                    for (j, b) in data.iter().enumerate() {
                        if i == j {
                            continue;
                        };
                        assertNe!(a, b);
                    }
                }


            }
    }

    @Test
    fun testRange() {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-113
            test(1., 5., 2., &[1., 3.]);

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-113
            test(10i64, 6i64, -3i64, &[10i64, 7i64]);

            fun test(
                start: impl NdArray,
                limit: impl NdArray,
                delta: impl NdArray,
                expected: impl NdArray,
            ) {
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Range",
                        domain = "",
                        attribute = mutableListOf(),
                        input = mutableListOf(
                            INPUT_X,
                            INPUT_Y,
                            INPUT_A,
                        ),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(INPUT_X, Tensor::new(start, Device.Cpu)?);
                inputs.insert(INPUT_Y, Tensor::new(limit, Device.Cpu)?);
                inputs.insert(INPUT_A, Tensor::new(delta, Device.Cpu)?);

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval
                    [OUTPUT_Z]
                     ?: error("Output 'z' not found")
                    .toDtype(DType.F64);

                val expected = Tensor::new(expected, Device.Cpu)?.toDtype(DType.F64);
                when (expected.dims().size) {
                    0 -> assertEquals(expected.toVec0<Double>(), z.toVec0<Double>()),
                    1 -> assertEquals(expected.toVec1<Double>(), z.toVec1<Double>()),
                    2 -> assertEquals(expected.toVec2<Double>(), z.toVec2<Double>()),
                    3 -> assertEquals(expected.toVec3<Double>(), z.toVec3<Double>()),
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testGreater() {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-63
            test(&[1., 2., 3.], &[3., 2., 1.], &[0u.toUByte(), 0, 1]);

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-63
            test(&[1., 2., 3.], 2., &[0u.toUByte(), 0, 1]);

            fun test(a: impl NdArray, b: impl NdArray, expected: impl NdArray) {
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Greater",
                        domain = "",
                        attribute = mutableListOf(),
                        input = mutableListOf(INPUT_X, INPUT_Y),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(INPUT_X, Tensor::new(a, Device.Cpu)?);
                inputs.insert(INPUT_Y, Tensor::new(b, Device.Cpu)?);

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval
                    [OUTPUT_Z]
                     ?: error("Output 'z' not found")
                    .toDtype(DType.F64);

                val expected = Tensor::new(expected, Device.Cpu)?.toDtype(DType.F64);
                when (expected.dims().size) {
                    0 -> assertEquals(expected.toVec0<Double>(), z.toVec0<Double>()),
                    1 -> assertEquals(expected.toVec1<Double>(), z.toVec1<Double>()),
                    2 -> assertEquals(expected.toVec2<Double>(), z.toVec2<Double>()),
                    3 -> assertEquals(expected.toVec3<Double>(), z.toVec3<Double>()),
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testLess() {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-81
            test(&[1., 2., 3.], &[3., 2., 1.], &[1u.toUByte(), 0, 0]);

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-81
            test(&[1., 2., 3.], 2., &[1u.toUByte(), 0, 0]);

            fun test(a: impl NdArray, b: impl NdArray, expected: impl NdArray) {
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Less",
                        domain = "",
                        attribute = mutableListOf(),
                        input = mutableListOf(INPUT_X, INPUT_Y),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(INPUT_X, Tensor::new(a, Device.Cpu)?);
                inputs.insert(INPUT_Y, Tensor::new(b, Device.Cpu)?);

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval
                    [OUTPUT_Z]
                     ?: error("Output 'z' not found")
                    .toDtype(DType.F64);

                val expected = Tensor::new(expected, Device.Cpu)?.toDtype(DType.F64);
                when (expected.dims().size) {
                    0 -> assertEquals(expected.toVec0<Double>(), z.toVec0<Double>()),
                    1 -> assertEquals(expected.toVec1<Double>(), z.toVec1<Double>()),
                    2 -> assertEquals(expected.toVec2<Double>(), z.toVec2<Double>()),
                    3 -> assertEquals(expected.toVec3<Double>(), z.toVec3<Double>()),
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testLog() {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-82
            test(&[1., 10.], &[0., kotlin.math.ln(10.0)]);

            fun test(data: impl NdArray, expected: impl NdArray) {
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Log",
                        domain = "",
                        attribute = mutableListOf(),
                        input = mutableListOf(INPUT_X),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(INPUT_X, Tensor::new(data, Device.Cpu)?);

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

                val expected = Tensor::new(expected, Device.Cpu);
                when (expected.dims().size) {
                    0 -> assertEquals(expected.toVec0<Double>(), z.toVec0<Double>()),
                    1 -> assertEquals(expected.toVec1<Double>(), z.toVec1<Double>()),
                    2 -> assertEquals(expected.toVec2<Double>(), z.toVec2<Double>()),
                    3 -> assertEquals(expected.toVec3<Double>(), z.toVec3<Double>()),
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testMin() {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-94
            test(&[3., 2., 1.], &[1., 4., 4.], &[2., 5., 0.], &[1., 2., 0.]);

            fun test(
                a: impl NdArray,
                b: impl NdArray,
                c: impl NdArray,
                expected: impl NdArray,
            ) {
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Min",
                        domain = "",
                        attribute = mutableListOf(),
                        input = mutableListOf(
                            INPUT_X,
                            INPUT_Y,
                            INPUT_A,
                        ),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(INPUT_X, Tensor::new(a, Device.Cpu)?);
                inputs.insert(INPUT_Y, Tensor::new(b, Device.Cpu)?);
                inputs.insert(INPUT_A, Tensor::new(c, Device.Cpu)?);

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

                val expected = Tensor::new(expected, Device.Cpu);
                when (expected.dims().size) {
                    0 -> assertEquals(expected.toVec0<Double>(), z.toVec0<Double>()),
                    1 -> assertEquals(expected.toVec1<Double>(), z.toVec1<Double>()),
                    2 -> assertEquals(expected.toVec2<Double>(), z.toVec2<Double>()),
                    3 -> assertEquals(expected.toVec3<Double>(), z.toVec3<Double>()),
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testWhere() {
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-173
            test(
                &[[1u.toUByte(), 0], [1, 1]],
                &[[1i64, 2], [3, 4]],
                &[[9i64, 8], [7, 6]],
                &[[1i64, 8], [3, 4]],
            );

            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-173
            test(
                &[[1u.toUByte(), 0], [1, 1]],
                &[[1., 2.], [3., 4.]],
                &[[9., 8.], [7., 6.]],
                &[[1., 8.], [3., 4.]],
            );

            fun test(
                condition: impl NdArray,
                x: impl NdArray,
                y: impl NdArray,
                expected: impl NdArray,
            ) {
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Where",
                        domain = "",
                        attribute = mutableListOf(),
                        input = mutableListOf(
                            INPUT_X,
                            INPUT_Y,
                            INPUT_A,
                        ),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(INPUT_X, Tensor::new(condition, Device.Cpu)?);
                inputs.insert(INPUT_Y, Tensor::new(x, Device.Cpu)?);
                inputs.insert(INPUT_A, Tensor::new(y, Device.Cpu)?);

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval
                    [OUTPUT_Z]
                     ?: error("Output 'z' not found")
                    .toDtype(DType.F64);

                val expected = Tensor::new(expected, Device.Cpu)?.toDtype(DType.F64);
                when (expected.dims().size) {
                    0 -> assertEquals(expected.toVec0<Double>(), z.toVec0<Double>()),
                    1 -> assertEquals(expected.toVec1<Double>(), z.toVec1<Double>()),
                    2 -> assertEquals(expected.toVec2<Double>(), z.toVec2<Double>()),
                    3 -> assertEquals(expected.toVec3<Double>(), z.toVec3<Double>()),
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testFloor() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Floor",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(ValueInfoProto(
                    name = INPUT_X,
                    docString = "",
                    type = null,
                )),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x = Tensor.fromVec(
                // some values taken from https://numpy.org/doc/stable/reference/generated/numpy.floor.html
                mutableListOf(
                    f64::NAN,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    -1.7,
                    -1.5,
                    -0.2,
                    0.2,
                    1.5,
                    1.7,
                    2.0,
                ),
                &[10],
                Device.Cpu,
            );

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            val results = z.toVec1<Double>();

            assertTrue(results[0].isNan())
            assertEquals(mutableListOf(
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    -2.,
                    -2.,
                    -1.,
                    0.,
                    1.,
                    1.,
                    2.
                ), results[1..])
    }

    @Test
    fun testCeil() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Ceil",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(ValueInfoProto(
                    name = INPUT_X,
                    docString = "",
                    type = null,
                )),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));
            val x = Tensor.fromVec(
                // some values taken from https://numpy.org/doc/stable/reference/generated/numpy.ceil.html
                mutableListOf(
                    f64::NAN,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    -1.7,
                    -1.5,
                    -0.2,
                    0.2,
                    1.5,
                    1.7,
                    2.0,
                ),
                &[10],
                Device.Cpu,
            );

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs[INPUT_X] = x

            val eval = simpleEval(manualGraph, inputs);
            assertEquals(1, eval.size)

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

            val results = z.toVec1<Double>();

            assertTrue(results[0].isNan())
            assertEquals(mutableListOf(
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    -1.,
                    -1.,
                    -0.,
                    1.,
                    2.,
                    2.,
                    2.
                ), results[1..])
    }

    @Test
    fun testArgmin() {
        // tests from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-7
            // defaultaxeskeepdims
            test(
                &[[2u32, 1u32], [3u32, 10u32]],
                null,
                1,
                null,
                &[[0i64, 0i64]],
            );
            // keepdims
            test(
                &[[2u32, 1u32], [3u32, 10u32]],
                1,
                1,
                null,
                &[[1i64], [0i64]],
            );
            // // negativeaxiskeepdims
            test(
                &[[2u32, 1u32], [3u32, 10u32]],
                -1,
                1,
                null,
                &[[1i64], [0i64]],
            );
            // nokeepdims
            test(
                &[[2u32, 1u32], [3u32, 10u32]],
                null,
                0,
                null,
                &[0i64, 0i64],
            );
            // tests from https://pytorch.org/docs/stable/generated/torch.argmin.html#torch.argmin
            test(
                &[
                    [0.1139, 0.2254, -0.1381, 0.3687],
                    [1.0100, -1.1975, -0.0102, -0.4732],
                    [-0.9240, 0.1207, -0.7506, -1.0213],
                    [1.7809, -1.2960, 0.9384, 0.1438],
                ],
                1,
                0,
                null,
                &[2i64, 1i64, 3i64, 1i64],
            );
            test(
                &[
                    [0.1139, 0.2254, -0.1381, 0.3687],
                    [1.0100, -1.1975, -0.0102, -0.4732],
                    [-0.9240, 0.1207, -0.7506, -1.0213],
                    [1.7809, -1.2960, 0.9384, 0.1438],
                ],
                1,
                null,
                null,
                &[[2i64], [1i64], [3i64], [1i64]],
            );
            fun test(
                data: impl NdArray,
                axis: Option<i64>,
                keepdims: Option<i64>,
                selectLastIndex: Option<i64>,
                expected: impl NdArray,
            ) {
                val attAxis = AttributeProto(
                    name = "axis",
                    refAttrName = "axis",
                    i = axis.unwrapOr(0),
                    docString = "axis",
                    type = 2, // INT
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );
                val attKeepdims = AttributeProto(
                    name = "keepdims",
                    refAttrName = "keepdims",
                    i = keepdims.unwrapOr(1),
                    docString = "keepdims",
                    type = 2, // INT
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );
                val attSelectLastIndex = AttributeProto(
                    name = "selectLastIndex",
                    refAttrName = "selectLastIndex",
                    i = selectLastIndex.unwrapOr(0),
                    docString = "selectLastIndex",
                    type = 2, // INT
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );
                val attrs = {
                    val mutAttrs = mutableListOf();
                    if axis.isSome() {
                        mutAttrs.push(attAxis);
                    }
                    if keepdims.isSome() {
                        mutAttrs.push(attKeepdims);
                    }
                    if selectLastIndex.isSome() {
                        mutAttrs.push(attSelectLastIndex);
                    }
                    mutAttrs
                };
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "ArgMin",
                        domain = "",
                        attribute = attrs,
                        input = mutableListOf(INPUT_X),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));
                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(INPUT_X, Tensor::new(data, Device.Cpu)?);
                val eval = simpleEval(manualGraph, inputs);
                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

                val expected = Tensor::new(expected, Device.Cpu);
                when (expected.dims().size) {
                    1 -> assertEquals(expected.toVec1<Long>(), z.toVec1<Long>()),
                    2 -> assertEquals(expected.toVec2<Long>(), z.toVec2<Long>()),
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testArgmax() {
        // tests from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-6
            // defaultaxeskeepdims
            test(
                &[[2u32, 1u32], [3u32, 10u32]],
                null,
                1,
                null,
                &[[1i64, 1i64]],
            );
            // keepdims
            test(
                &[[2u32, 1u32], [3u32, 10u32]],
                1,
                1,
                null,
                &[[0i64], [1i64]],
            );
            // // negativeaxiskeepdims
            test(
                &[[2u32, 1u32], [3u32, 10u32]],
                -1,
                1,
                null,
                &[[0i64], [1i64]],
            );
            // nokeepdims
            test(
                &[[2u32, 1u32], [3u32, 10u32]],
                null,
                0,
                null,
                &[1i64, 1i64],
            );
            // tests from https://pytorch.org/docs/stable/generated/torch.argmax.html
            test(
                &[
                    [1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195],
                ],
                1,
                0,
                null,
                &[0i64, 2i64, 0i64, 1i64],
            );
            test(
                &[
                    [1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195],
                ],
                1,
                null,
                null,
                &[[0i64], [2i64], [0i64], [1i64]],
            );
            fun test(
                data: impl NdArray,
                axis: Option<i64>,
                keepdims: Option<i64>,
                selectLastIndex: Option<i64>,
                expected: impl NdArray,
            ) {
                val attAxis = AttributeProto(
                    name = "axis",
                    refAttrName = "axis",
                    i = axis.unwrapOr(0),
                    docString = "axis",
                    type = 2, // INT
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );
                val attKeepdims = AttributeProto(
                    name = "keepdims",
                    refAttrName = "keepdims",
                    i = keepdims.unwrapOr(1),
                    docString = "keepdims",
                    type = 2, // INT
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );
                val attSelectLastIndex = AttributeProto(
                    name = "selectLastIndex",
                    refAttrName = "selectLastIndex",
                    i = selectLastIndex.unwrapOr(0),
                    docString = "selectLastIndex",
                    type = 2, // INT
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );
                val attrs = {
                    val mutAttrs = mutableListOf();
                    if axis.isSome() {
                        mutAttrs.push(attAxis);
                    }
                    if keepdims.isSome() {
                        mutAttrs.push(attKeepdims);
                    }
                    if selectLastIndex.isSome() {
                        mutAttrs.push(attSelectLastIndex);
                    }
                    mutAttrs
                };
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "ArgMax",
                        domain = "",
                        attribute = attrs,
                        input = mutableListOf(INPUT_X),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));
                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(INPUT_X, Tensor::new(data, Device.Cpu)?);
                val eval = simpleEval(manualGraph, inputs);
                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

                val expected = Tensor::new(expected, Device.Cpu);
                when (expected.dims().size) {
                    1 -> assertEquals(expected.toVec1<Long>(), z.toVec1<Long>()),
                    2 -> assertEquals(expected.toVec2<Long>(), z.toVec2<Long>()),
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testLeakyrelu() {
        // tests from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-80
            // leakyrelu
            test(&[-1.0, 0.0, 1.0], 0.1, &[-0.1, 0.0, 1.0]);
            fun test(data: impl NdArray, alpha: Option<f32>, expected: impl NdArray) {
                val attAlpha = AttributeProto(
                    name = "alpha",
                    refAttrName = "alpha",
                    i = 0,
                    docString = "alpha",
                    type = 1, // FLOAT
                    f = alpha.unwrapOr(0.01),
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );
                val attrs = {
                    val mutAttrs = mutableListOf();
                    if alpha.isSome() {
                        mutAttrs.push(attAlpha);
                    }
                    mutAttrs
                };
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "LeakyRelu",
                        domain = "",
                        attribute = attrs,
                        input = mutableListOf(INPUT_X),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));
                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(INPUT_X, Tensor::new(data, Device.Cpu)?);
                val eval = simpleEval(manualGraph, inputs);
                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

                val expected = Tensor::new(expected, Device.Cpu);
                for both in z
                    .toVec1<Double>()
                    .iter()
                    .zip(expected.toVec1<Double>().iter())
                {
                    val (act, exp) = both;
                    assertTrue(f64::abs(act - exp) < f32::EPSILON.into())
                }


            }
    }

    @Test
    fun testIf() {
        val x = mutableListOf(1.0, 2.0, 3.0, 4.0, 5.0);
            val y = mutableListOf(5.0, 4.0, 3.0, 2.0, 1.0);
            val outputTypeProto = TypeProto(
                value = typeProto::Value::TensorType(typeProto::Tensor {
                    elemType = TensorProto.DataType.FLOAT,
                    shape = TensorShapeProto(
                        dim = mutableListOf(Dimension(
                            denotation: "",
                            value = dimension::Value::DimValue(5),
                        )),
                    ),
                }),
                denotation: "",
            );
            val thenBranch = GraphProto(
                output = mutableListOf(ValueInfoProto(
                    name = "thenOut",
                    type = outputTypeProto,
                    docString = "",
                )),
                node = mutableListOf(NodeProto(
                    opType = "Constant",
                    input = mutableListOf(),
                    output = mutableListOf("thenOut"),
                    attribute = mutableListOf(AttributeProto(
                        name = "value",
                        type = AttributeProto.AttributeType.TENSOR,
                        t = TensorProto(
                            dims = mutableListOf(x.size as i64),
                            floatData = x,
                            dataType = TensorProto.DataType.FLOAT,

                        ),

                    )),

                )),

            );
            val elseBranch = GraphProto(
                output = mutableListOf(ValueInfoProto(
                    name = "elseOut",
                    type = outputTypeProto,
                    docString = "",
                )),
                node = mutableListOf(NodeProto(
                    opType = "Constant",
                    input = mutableListOf(),
                    output = mutableListOf("elseOut"),
                    attribute = mutableListOf(AttributeProto(
                        name = "value",
                        type = AttributeProto.AttributeType.TENSOR,
                        t = TensorProto(
                            dims = mutableListOf(y.size as i64),
                            floatData = y,
                            dataType = TensorProto.DataType.FLOAT,

                        ),

                    )),

                )),

            );
            val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "If",
                    attribute = mutableListOf(
                        AttributeProto(
                            name = "thenBranch",
                            type = AttributeType::Graph.into(),
                            g = thenBranch,

                        ),
                        AttributeProto(
                            name = "elseBranch",
                            type = AttributeType::Graph.into(),
                            g = elseBranch,

                        ),
                    ),
                    input = mutableListOf("cond"),
                    output = mutableListOf("res"),

                )),
                input = mutableListOf(),
                output = mutableListOf(ValueInfoProto(
                    name = "res",
                    docString = "",
                    type = outputTypeProto,
                )),

            ));

            for cond in [1u.toUByte(), 0] {
                val inputs =
                    mutableMapOf(("cond", Tensor::full(cond, (1,), Device.Cpu)?)]);
                val outputs = simpleEval(manualGraph, inputs);
                val expected = if cond != 0 { &x } else { &y };
                val res = outputs["res"] else {
                    candle::bail!("outputs didn't contain expected key `res`: {outputs:?}");
                };
                assertEquals(expected, &res.toVec1<Float>())
            }
    }

    @Test
    fun testPad() {
        val data = Tensor.fromVec(
                mutableListOf(
                    1.0, 2.0, 3.0, //
                    4.0, 5.0, 6.0, //
                ),
                (2, 3),
                Device.Cpu,
            );
            val pads = Tensor.fromVec(mutableListOf(0i64, 1, 0, 0), (4,), Device.Cpu);
            val mode = "reflect";

            val expected = Tensor.fromVec(
                mutableListOf(
                    2.0, 1.0, 2.0, 3.0, //
                    5.0, 4.0, 5.0, 6.0, //
                ),
                (2, 4),
                Device.Cpu,
            );

            val model = createModelProtoWithGraph(GraphProto(
                input = mutableListOf(
                    ValueInfoProto(
                        name = "data",

                    ),
                    ValueInfoProto(
                        name = "pads",

                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = "output",

                )),
                node = mutableListOf(NodeProto(
                    opType = "Pad",
                    input = mutableListOf("data", "pads"),
                    output = mutableListOf("output"),
                    attribute = mutableListOf(AttributeProto(
                        name = "mode",
                        type = AttributeProto.AttributeType.STRING,
                        s = mode.asBytes(),

                    )),

                )),

            ));

            val inputs = mutableMapOf(("data", data), ("pads", pads)]);
            val res = simpleEval(model, inputs);
            val actual = res["output"] else {
                candle::bail!("outputs didn't contain expected key `output`: {res:?}");
            };

            assertEquals(expected.toVec2<Double>(), actual.toVec2<Double>())
    }

    @Test
    fun testSlice() {
        val model = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Slice",
                    input = mutableListOf(
                        "data",
                        "starts",
                        "ends",
                        "axes",
                        "steps",
                    ),
                    output = mutableListOf("result"),

                )),
                input = ["data", "starts", "ends", "axes", "steps"]
                    .intoIter()
                    .map(|name| ValueInfoProto(
                        name = name,
                        type = null,
                        docString = "",
                    ))
                    .collect(),
                output = ["result"]
                    .intoIter()
                    .map(|name| ValueInfoProto(
                        name = name,
                        type = null,
                        docString = "",
                    ))
                    .collect(),

            ));



            val outputs = simpleEval(
                model,
                mutableMapOf((
                        "data",
                        Tensor.fromVec(mutableListOf(1i64, 2, 3, 4, 5, 6, 7, 8), (2, 4), Device.Cpu)?,
                    ),
                    (
                        "starts",
                        Tensor.fromVec(mutableListOf(1i64, 0), (2,), Device.Cpu)?,
                    ),
                    (
                        "ends",
                        Tensor.fromVec(mutableListOf(2i64, 3), (2,), Device.Cpu)?,
                    ),
                    (
                        "axes",
                        Tensor.fromVec(mutableListOf(0i64, 1), (2,), Device.Cpu)?,
                    ),
                    (
                        "steps",
                        Tensor.fromVec(mutableListOf(1i64, 2), (2,), Device.Cpu)?,
                    ),
                ]),
            );
            val actual = outputs["result"]!!.toVec2<Long>();
            assertEquals(mutableListOf(mutableListOf(5i64, 7)), actual)


            val model = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Slice",
                    input = mutableListOf("data", "starts", "ends"),
                    output = mutableListOf("result"),

                )),
                input = ["data", "starts", "ends"]
                    .intoIter()
                    .map(|name| ValueInfoProto(
                        name = name,
                        type = null,
                        docString = "",
                    ))
                    .collect(),
                output = ["result"]
                    .intoIter()
                    .map(|name| ValueInfoProto(
                        name = name,
                        type = null,
                        docString = "",
                    ))
                    .collect(),

            ));
            val outputs = simpleEval(
                model,
                mutableMapOf((
                        "data",
                        Tensor.fromVec(mutableListOf(1i64, 2, 3, 4, 5, 6, 7, 8), (2, 4), Device.Cpu)?,
                    ),
                    (
                        "starts",
                        Tensor.fromVec(mutableListOf(0i64, 1), (2,), Device.Cpu)?,
                    ),
                    (
                        "ends",
                        Tensor.fromVec(mutableListOf(-1i64, 1000), (2,), Device.Cpu)?,
                    ),
                ]),
            );
            val actual = outputs["result"]!!.toVec2<Long>();
            assertEquals(mutableListOf(mutableListOf(2i64, 3, 4)), actual)
    }

    @Test
    fun testLstm() {
        // values generated from pytorch, so at least it's close enough to what pytorch does

            val inputSize = 3;
            val hiddenSize = 5;
            val batchSize = 1;
            val sequenceLength = 4;
            val numberDirections = 1;
            val weightIhL0 = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    -1.525_595_9,
                    -0.750_231_8,
                    -0.653_980_9,
                    -1.609_484_8,
                    -0.100_167_18,
                    -0.609_188_9,
                    -0.979_772_27,
                    -1.609_096_3,
                    -0.712_144_6,
                    0.303_722,
                    -0.777_314_3,
                    -0.251_455_25,
                    -0.222_270_49,
                    1.687_113_4,
                    0.228_425_17,
                    0.467_635_5,
                    -0.696_972_4,
                    -1.160_761_5,
                    0.699_542_4,
                    0.199_081_63,
                    0.865_692_4,
                    0.244_403_9,
                    -0.662_911_36,
                    0.807_308_26,
                    1.101_680_6,
                    -0.175_936_04,
                    -2.245_557_8,
                    -1.446_458,
                    0.061_155_282,
                    -0.617_744_45,
                    -0.798_069_83,
                    -0.131_623_21,
                    1.879_345_8,
                    -0.072_131_78,
                    0.157_770_6,
                    -0.773_454_9,
                    0.199_056_5,
                    0.045_702_778,
                    0.152_956_92,
                    -0.475_678_8,
                    -0.111_019_83,
                    0.292_735_25,
                    -0.157_845_15,
                    -0.028_787_14,
                    0.453_254_58,
                    1.142_161_1,
                    0.248_610_7,
                    -1.775_400_8,
                    -0.025_502_462,
                    -1.023_330_6,
                    -0.596_185_15,
                    -1.005_530_7,
                    0.428_542_3,
                    1.476_077_8,
                    -1.786_867_9,
                    1.610_317_6,
                    -0.703_956_66,
                    -0.185_265_8,
                    -0.996_235_1,
                    -0.831_255_26,
                ),
                (20, 3),
                Device.Cpu,
            );
            val weightHhL0 = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    0.409_972_43,
                    0.408_450_66,
                    0.257_865_4,
                    1.095_021_4,
                    -0.506_486_6,
                    0.099_775_404,
                    -0.653_973_4,
                    0.731_693_7,
                    -1.456_733,
                    1.608_935_4,
                    0.093_769_975,
                    -1.259_749,
                    0.254_633_5,
                    -0.501_957_3,
                    -1.041_2,
                    0.732_267_2,
                    1.307_535_5,
                    -1.162_798_8,
                    0.119_636_11,
                    -0.163_135_33,
                    0.661_445_3,
                    1.189_920_5,
                    0.816_533_9,
                    -0.913_523_6,
                    -0.353_806_53,
                    0.763_927_04,
                    -0.588_950_7,
                    -0.763_597_37,
                    1.335_205_7,
                    0.604_273_6,
                    -0.103_442_08,
                    -0.151_216_92,
                    1.246_568_3,
                    0.505_721_4,
                    0.950_511_2,
                    1.296_648_3,
                    0.873_796_3,
                    -0.560_259_4,
                    1.285_784_5,
                    0.816_823_84,
                    -1.464_799_4,
                    -1.262_928_4,
                    1.122_018_8,
                    1.566_334_1,
                    2.558_138_4,
                    -0.233_363_88,
                    -0.013_472_13,
                    1.860_634_8,
                    1.549_620_5,
                    0.347_629_25,
                    0.093_008_03,
                    0.614_740_3,
                    0.712_364_55,
                    -1.776_507_3,
                    0.353_864_58,
                    1.199_613_2,
                    -0.712_258_93,
                    -0.620_034_4,
                    -0.228_134_95,
                    -0.789_274_63,
                    -1.611_111_8,
                    -1.871_612_9,
                    0.543_083_6,
                    0.660_678_6,
                    0.270_527_72,
                    0.559_691_97,
                    -0.318_396_3,
                    1.511_720_7,
                    -1.363_267_2,
                    -0.983_219_6,
                    1.511_266_7,
                    0.641_870_74,
                    -0.747_445_9,
                    -0.923_438_55,
                    0.573_398_4,
                    -0.109_299_51,
                    0.518_112_1,
                    0.106_535_35,
                    0.269_240_77,
                    1.324_768,
                    0.037_456_9,
                    -0.637_839_3,
                    -0.814_755_44,
                    -0.689_506_53,
                    0.843_654_3,
                    1.165_701_3,
                    0.526_932_2,
                    1.619_253_3,
                    -0.963_976_26,
                    0.141_520_38,
                    -0.163_660_96,
                    -0.358_222_57,
                    1.722_279_3,
                    -0.303_575_6,
                    0.238_874_2,
                    1.344_001_2,
                    0.103_225_69,
                    1.100_354_2,
                    -0.341_680_2,
                    0.947_338_9,
                ),
                (20, 5),
                Device.Cpu,
            );
            val biasIhL0 = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    -0.568_515_96,
                    0.837_596_2,
                    1.783_660_7,
                    -0.195_424_66,
                    0.235_193_13,
                    1.914_243_3,
                    1.836_411_1,
                    1.324_532_4,
                    -0.070_514_58,
                    0.346_979_4,
                    -0.653_679_6,
                    1.558_620_2,
                    0.218_566_15,
                    -0.574_307_26,
                    1.457_125_1,
                    1.770_955_7,
                    -2.017_3,
                    0.423_503_2,
                    0.573_022,
                    -1.796_243,
                ),
                (20,),
                Device.Cpu,
            );
            val biasHhL0 = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    1.247_040_4,
                    1.273_851_2,
                    0.390_949_25,
                    0.387_210_5,
                    0.144_403_95,
                    0.777_168_45,
                    -2.338_112_6,
                    -0.829_120_4,
                    1.166_139_1,
                    1.478_657_5,
                    0.267_608_73,
                    0.756_119_85,
                    -0.587_336_1,
                    -2.061_920_6,
                    0.430_473_48,
                    0.337_656_62,
                    -0.343_785_35,
                    -0.617_226_06,
                    1.252_969_3,
                    -0.051_417_42,
                ),
                (20,),
                Device.Cpu,
            );
            val input = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    0.647_212_8,
                    -0.041_167_17,
                    -0.177_493_08,
                    -0.500_039_3,
                    0.867_274_94,
                    -0.273_192_23,
                    -0.460_768_13,
                    -0.099_093_71,
                    0.472_844_8,
                    1.004_948_5,
                    -0.287_142_04,
                    -1.161_862_1,
                ),
                (4, 1, 3),
                Device.Cpu,
            );
            val h0 = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    0.027_581_785,
                    0.565_238_24,
                    -0.011_487_379,
                    0.670_640_05,
                    -0.492_925_05,
                ),
                (1, 1, 5),
                Device.Cpu,
            );
            val c0 = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    1.505_028_5,
                    -2.326_355,
                    1.616_89,
                    -0.902_623_8,
                    0.173_668_24,
                ),
                (1, 1, 5),
                Device.Cpu,
            );
            val output = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    0.595_601_7,
                    -0.017_232_792,
                    0.110_355_72,
                    -0.493_231_74,
                    0.047_632_16,
                    0.635_845_2,
                    0.040_328_12,
                    -0.378_861_16,
                    -0.746_434,
                    0.200_809_09,
                    0.584_026_5,
                    0.145_328_82,
                    -0.734_529_85,
                    -0.521_430_43,
                    0.219_038_17,
                    0.742_045_16,
                    0.319_438_8,
                    -0.047_266_465,
                    -0.282_384_96,
                    0.271_313_4,
                ),
                (4, 1, 5),
                Device.Cpu,
            );
            val hn = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    0.742_045_16,
                    0.319_438_8,
                    -0.047_266_465,
                    -0.282_384_96,
                    0.271_313_4,
                ),
                (1, 1, 5),
                Device.Cpu,
            );
            val cn = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    0.963_055_85,
                    1.003_307,
                    -1.754_899,
                    -1.596_712_2,
                    0.825_292_47,
                ),
                (1, 1, 5),
                Device.Cpu,
            );
            // end of generated values

            val model = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "LSTM",
                    name = "LSTM_test",
                    attribute = mutableListOf(AttributeProto(
                        name = "hiddenSize",
                        type = AttributeProto.AttributeType.INT,
                        i = hiddenSize as i64,

                    )),
                    input = mutableListOf(
                        "input",
                        "w",
                        "r",
                        "b", // b
                        "",  // seqlens
                        "h",
                        "c",
                    ),
                    output = mutableListOf("output", "hn", "cn"),

                )),
                input = ["input", "w", "r", "b", "h", "c"]
                    .intoIter()
                    .map(|name| ValueInfoProto(
                        name = name,

                    ))
                    .collect(),
                output = ["output", "hn", "cn"]
                    .intoIter()
                    .map(|name| ValueInfoProto(
                        name = name,

                    ))
                    .collect(),

            ));
            // pytorch stores weight and bias as [ifco] but we want it as [iofc]
            // so we need to re-arrange the tensors a bit
            val idxIofc = {
                val stride = hiddenSize as i64;
                val dev = weightIhL0.device();
                val idxI = Tensor::arange(0, stride, dev);
                val idxF = Tensor::arange(stride, 2 * stride, dev);
                val idxG = Tensor::arange(2 * stride, 3 * stride, dev);
                val idxO = Tensor::arange(3 * stride, 4 * stride, dev);

                Tensor::cat(&[&idxI, &idxO, &idxF, &idxG], 0)
            };
            val w = weightIhL0.indexSelect(&idxIofc, 0);
            val w = w.reshape((numberDirections, 4 * hiddenSize, inputSize));
            val r = weightHhL0.indexSelect(&idxIofc, 0);
            val r = r.reshape((numberDirections, 4 * hiddenSize, hiddenSize));
            val wb = biasIhL0.indexSelect(&idxIofc, 0);
            val rb = biasHhL0.indexSelect(&idxIofc, 0);
            val b = Tensor::cat(&[wb, rb], 0)?.reshape((numberDirections, 8 * hiddenSize));
            val output = output.reshape((sequenceLength, numberDirections, batchSize, hiddenSize));
            val result = simpleEval(
                model,
                mutableMapOf(("input", input),
                    ("w", w),
                    ("r", r),
                    ("b", b),
                    ("h", h0),
                    ("c", c0),
                ]),
            );
            val actualOutput = result["output"]!!;
            assertEquals(actualOutput.dims(), output.dims())
            val actualHn = result["hn"]!!;
            assertEquals(actualHn.dims(), hn.dims())
            val actualCn = result["cn"]!!;
            assertEquals(actualCn.dims(), cn.dims())
            val diffCloseEnough = |a: &Tensor, b| -> Result<_> {
                val diffs = a.sub(b)?.flattenAll()?.toVec1<Float>();
                Ok(diffs.iter().all(|f| f.abs() < 0.0001))
            };
            assertTrue(diffCloseEnough(&output, actualOutput)?,
                "output did not when (expected\n) {actualOutput}\n{output}",)
            assertTrue(diffCloseEnough(&hn, actualHn)?,
                "hn did not when (expected\n) {actualHn}\n{hn}",)
            assertTrue(diffCloseEnough(&cn, actualCn)?,
                "cn did not when (expected\n) {actualCn}\n{cn}",)
    }

    @Test
    fun testRnn() {
        // values generated from pytorch, so at least it's close enough to what pytorch does


            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#RNN
            val model = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "RNN",
                    name = "RNN_test",
                    attribute = mutableListOf(AttributeProto(
                        name = "hiddenSize",
                        type = AttributeProto.AttributeType.INT,
                        i = 5,

                    )),
                    input = mutableListOf(
                        "input",
                        "w",
                        "r",
                        "b", // b
                        "",  // seqlens
                        "h",
                    ),
                    output = mutableListOf("output", "hn"),

                )),
                input = ["input", "w", "r", "b", "h"]
                    .intoIter()
                    .map(|name| ValueInfoProto(
                        name = name,

                    ))
                    .collect(),
                output = ["output", "hn"]
                    .intoIter()
                    .map(|name| ValueInfoProto(
                        name = name,

                    ))
                    .collect(),

            ));

            val inputSize = 3;
            val hiddenSize = 5;
            val batchSize = 1;
            val sequenceLength = 4;
            val numberDirections = 1;
            val weightIhL0 = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    0.33669036626815796,
                    0.12880940735340118,
                    0.23446236550807953,
                    0.23033303022384644,
                    -1.1228563785552979,
                    -0.18632829189300537,
                    2.2082014083862305,
                    -0.637997031211853,
                    0.46165722608566284,
                    0.2673508822917938,
                    0.5349046587944031,
                    0.809357225894928,
                    1.110290288925171,
                    -1.6897989511489868,
                    -0.9889599084854126,
                ),
                (5, 3),
                Device.Cpu,
            );
            val weightHhL0 = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    -1.3846737146377563,
                    -0.8712361454963684,
                    -0.223365917801857,
                    1.7173614501953125,
                    0.3188803195953369,
                    -0.42451897263526917,
                    0.3057209253311157,
                    -0.7745925188064575,
                    -1.5575724840164185,
                    -0.9223900437355042,
                    1.811317801475525,
                    0.16056492924690247,
                    0.36724865436553955,
                    0.17541083693504333,
                    1.3851605653762817,
                    -0.44585201144218445,
                    1.4451338052749634,
                    0.7078122496604919,
                    -1.0758858919143677,
                    0.5356546640396118,
                    1.1753677129745483,
                    0.5611738562583923,
                    -0.45274803042411804,
                    -0.771777868270874,
                    -0.1721901297569275,
                ),
                (5, 5),
                Device.Cpu,
            );
            val biasIhL0 = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    0.9579718112945557,
                    -0.6381967663764954,
                    -1.9187371730804443,
                    -0.6441153287887573,
                    -0.6060903072357178,
                ),
                (5,),
                Device.Cpu,
            );
            val biasHhL0 = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    -0.1425034999847412,
                    0.972653865814209,
                    2.0037777423858643,
                    0.6621911525726318,
                    0.5332217216491699,
                ),
                (5,),
                Device.Cpu,
            );
            val input = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    2.748873233795166,
                    -0.3840780258178711,
                    -1.962258219718933,
                    -0.30899786949157715,
                    -0.4268203377723694,
                    0.4503966271877289,
                    -0.0022214562632143497,
                    -0.19801591336727142,
                    1.775763750076294,
                    -1.6059082746505737,
                    0.48799338936805725,
                    -0.17943637073040009,
                ),
                (4, 1, 3),
                Device.Cpu,
            );
            val hx = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    1.4753035306930542,
                    -1.353177547454834,
                    0.16822677850723267,
                    -0.8245629668235779,
                    -0.060138583183288574,
                ),
                (1, 1, 5),
                Device.Cpu,
            );
            val output = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    -0.8023818135261536,
                    0.9590549468994141,
                    0.9999996423721313,
                    -0.9906406402587891,
                    0.9999986886978149,
                    -0.5140700936317444,
                    0.8138962388038635,
                    0.16080257296562195,
                    0.9994772672653198,
                    -0.38456836342811584,
                    0.992118239402771,
                    -0.5608834624290466,
                    -0.07238662987947464,
                    0.9196381568908691,
                    -0.9843823313713074,
                    0.5993185043334961,
                    -0.9232994914054871,
                    -0.9976708292961121,
                    -0.9960790276527405,
                    -0.973706841468811,
                ),
                (4, 1, 5),
                Device.Cpu,
            );
            val hn = Tensor::fromVec::<_, f32>(
                mutableListOf(
                    0.5993185043334961,
                    -0.9232994914054871,
                    -0.9976708292961121,
                    -0.9960790276527405,
                    -0.973706841468811,
                ),
                (1, 1, 5),
                Device.Cpu,
            );

            val w = weightIhL0.reshape((numberDirections, hiddenSize, inputSize));
            val r = weightHhL0.reshape((numberDirections, hiddenSize, hiddenSize));
            val wb = biasIhL0.reshape((numberDirections, hiddenSize));
            val rb = biasHhL0.reshape((numberDirections, hiddenSize));
            val b = Tensor::cat(&[wb, rb], 0)?.reshape((numberDirections, 2 * hiddenSize));
            val h = hx.reshape((numberDirections, batchSize, hiddenSize));
            val output = output.reshape((sequenceLength, numberDirections, batchSize, hiddenSize));
            val hn = hn.reshape((numberDirections, batchSize, hiddenSize));

            val diffCloseEnough = |a: &Tensor, b| -> Result<_> {
                val diffs = a.sub(b)?.flattenAll()?.toVec1<Float>();
                Ok(diffs.iter().all(|f| f.abs() < 0.0001))
            };
            val result = simpleEval(
                model,
                mutableMapOf(("input", input),
                    ("w", w),
                    ("r", r),
                    ("b", b),
                    ("h", h),
                ]),
            );
            val actualOutput = result["output"]!!;
            assertEquals(actualOutput.dims(), output.dims())
            val actualHn = result["hn"]!!;
            assertEquals(actualHn.dims(), hn.dims())
            assertTrue(diffCloseEnough(&output, actualOutput)?,
                "output did not when (expected\n) {actualOutput}\n{output}",)
            assertTrue(diffCloseEnough(&hn, actualHn)?,
                "hn did not when (expected\n) {actualHn}\n{hn}",)
    }

    @Test
    fun testExpandDimChanged() {
        // Create a manual graph for the Expand operation
            val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Expand",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf("data", "newShape"),
                    output = mutableListOf("expanded"),
                    name = "",
                    docString = "",
                )),
                input = mutableListOf(
                    ValueInfoProto(
                        name = "data",
                        docString = "",
                        type = null,
                    ),
                    ValueInfoProto(
                        name = "newShape",
                        docString = "",
                        type = null,
                    ),
                ),
                output = mutableListOf(ValueInfoProto(
                    name = "expanded",
                    docString = "",
                    type = null,
                )),

            ));

            // Input tensor with shape [3, 1]
            val data = Tensor.fromVec(mutableListOf(1.0f32, 2.0f32, 3.0f32), (3, 1), Device.Cpu);

            // New shape tensor: [2, 1, 6]
            val newShape = Tensor.fromVec(mutableListOf(2i64, 1, 6), (3,), Device.Cpu);

            // Expected output after expansion
            val expected = Tensor.fromVec(
                mutableListOf(
                    1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32,
                    2.0f32, 3.0f32, 3.0f32, 3.0f32, 3.0f32, 3.0f32, 3.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32,
                    1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 3.0f32, 3.0f32, 3.0f32,
                    3.0f32, 3.0f32, 3.0f32,
                ),
                (2, 3, 6),
                Device.Cpu,
            );

            // Execute the model evaluation
            val inputs = mutableMapOf(("data", data),
                ("newShape", newShape),
            ]);
            val result = simpleEval(manualGraph, inputs);

            // Retrieve and compare the result
            val expanded = result["expanded"] ?: error("Output 'expanded' not found");

            assertEquals(expected.toVec3<Float>(), expanded.toVec3<Float>())
    }

    @Test
    fun testExpandDimUnchanged() {
        // Create a manual graph for the Expand operation
            val manualGraph = makeGraphHelper("Expand", &["data", "newShape"], &["expanded"], mutableListOf());

            // Input tensor with shape [3, 1] and dtype f32
            val data = Tensor.fromVec(mutableListOf(1.0f32, 2.0f32, 3.0f32), (3, 1), Device.Cpu);

            // New shape tensor: [3, 4]
            val newShape = Tensor.fromVec(mutableListOf(3i64, 4), (2,), Device.Cpu);

            // Expected output after expansion, dtype f32
            val expected = Tensor.fromVec(
                mutableListOf(
                    1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 3.0f32, 3.0f32, 3.0f32,
                    3.0f32,
                ),
                (3, 4),
                Device.Cpu,
            );

            // Execute the model evaluation
            val inputs = mutableMapOf(("data", data),
                ("newShape", newShape),
            ]);
            val result = simpleEval(manualGraph, inputs);

            // Retrieve and compare the result
            val expanded = result["expanded"] ?: error("Output 'expanded' not found");
            assertEquals(expected.toVec2<Float>(), expanded.toVec2<Float>())
    }

    @Test
    fun testSplitEqualParts1dOpset13() {
        val input = Tensor.fromVec(
                mutableListOf(1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32),
                (6,),
                Device.Cpu,
            );
            val inputs = mutableMapOf();
            inputs["input"] = input

            {
                val manualGraph =
                    makeSplitGraphHelper(&["input"], &["output1", "output2", "output3"], 0);
                val eval = simpleEval(manualGraph, inputs);
                assertEquals(3, eval.size)

                val out1 = eval["output1"] ?: error("Output 'output1' not found");
                val out2 = eval["output2"] ?: error("Output 'output2' not found");
                val out3 = eval["output3"] ?: error("Output 'output3' not found");

                assertEquals(mutableListOf(1.0f32, 2.0f32), out1.toVec1<Float>())
                assertEquals(mutableListOf(3.0f32, 4.0f32), out2.toVec1<Float>())
                assertEquals(mutableListOf(5.0f32, 6.0f32), out3.toVec1<Float>())
            }

            {
                val splits = Tensor.fromVec(mutableListOf(2i64, 4), (2,), Device.Cpu);
                inputs["split"] = splits

                val manualGraph =
                    makeSplitGraphHelper(&["input", "split"], &["output1", "output2"], 0);
                val eval = simpleEval(manualGraph, inputs);
                assertEquals(2, eval.size)

                val out1 = eval["output1"] ?: error("Output 'output1' not found");
                val out2 = eval["output2"] ?: error("Output 'output2' not found");

                assertEquals(mutableListOf(1.0f32, 2.0f32), out1.toVec1<Float>())
                assertEquals(mutableListOf(3.0f32, 4.0f32, 5.0f32, 6.0f32), out2.toVec1<Float>())
            }
    }

    @Test
    fun testReduceSumDefaultAxesKeepdims() {
        val manualGraph = makeReduceSumGraphHelper(&["data", "axes"], &["reduced"], 1, null);

            // Test with example data
            {
                val data = Tensor.fromVec(
                    mutableListOf(
                        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    ),
                    (3, 2, 2),
                    Device.Cpu,
                );
                // val axes = Tensor::fromvec(Vec::<i64>::new(), (0,), Device.Cpu);

                val inputs = mutableMapOf();
                inputs["data"] = data
                // inputs["axes".tostring()] = axes

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val reduced = eval["reduced"] ?: error("Output 'reduced' not found");
                val expected = Tensor.fromVec(mutableListOf(78.0f32), (1, 1, 1), Device.Cpu);

                assertEquals(expected.toVec3<Float>(), reduced.toVec3<Float>())
            }

            {
                val data = Tensor.fromVec(
                    mutableListOf(
                        -5.2f32, 7.8, -3.1, 9.4, 2.6, -8.7, 4.3, -1.9, 6.5, -0.8, -7.2, 3.6,
                    ),
                    (3, 2, 2),
                    Device.Cpu,
                );

                val inputs = mutableMapOf();
                inputs.insert("data", data);

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val reduced = eval["reduced"] ?: error("Output 'reduced' not found");
                val expected = data.sumAll()?.reshape((1, 1, 1));

                assertEquals(expected.toVec3<Float>(), reduced.toVec3<Float>())
            }
    }

    @Test
    fun testReduceSumDoNotKeepDims() {
        val manualGraph = makeReduceSumGraphHelper(&["data", "axes"], &["reduced"], 0, null);

            // Test with example data
            {
                val data = Tensor.fromVec(
                    mutableListOf(
                        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    ),
                    (3, 2, 2),
                    Device.Cpu,
                );
                val axes = Tensor.fromVec(mutableListOf(1i64), (1,), Device.Cpu);

                val inputs = mutableMapOf();
                inputs["data"] = data
                inputs["axes"] = axes

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val reduced = eval["reduced"] ?: error("Output 'reduced' not found");
                val expected = Tensor.fromVec(
                    mutableListOf(4.0f32, 6.0, 12.0, 14.0, 20.0, 22.0),
                    (3, 2),
                    Device.Cpu,
                );

                assertEquals(expected.toVec2<Float>(), reduced.toVec2<Float>())
            }

            // Test with random data
            {
                val _shape = (3, 2, 2);
                val data = Tensor.fromVec(
                    mutableListOf(
                        -5.2f32, 7.8, -3.1, 9.4, 2.6, -8.7, 4.3, -1.9, 6.5, -0.8, -7.2, 3.6,
                    ),
                    (3, 2, 2),
                    Device.Cpu,
                );
                val axes = Tensor.fromVec(mutableListOf(1i64), (1,), Device.Cpu);

                val inputs = mutableMapOf();
                inputs.insert("data", data);
                inputs["axes"] = axes

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val reduced = eval["reduced"] ?: error("Output 'reduced' not found");

                // Calculate expected result
                val expected = data.sum(1);

                assertEquals(expected.toVec2<Float>(), reduced.toVec2<Float>())
            }
    }

    @Test
    fun testXor() {
        // tests based on: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Xor xor

            // 2d
            test(
                &[[0u.toUByte(), 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 1]],
                &[[1u.toUByte(), 1, 0, 0], [1, 0, 0, 1], [1, 1, 1, 0]],
                &[[1u.toUByte(), 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]],
            );

            // 3d
            test(
                &[
                    [
                        [0u.toUByte(), 1, 1, 1, 1],
                        [0, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 1],
                    ],
                    [
                        [0, 0, 1, 1, 1],
                        [1, 0, 1, 1, 1],
                        [1, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0],
                    ],
                    [
                        [1, 0, 0, 1, 1],
                        [1, 1, 1, 0, 0],
                        [1, 1, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                    ],
                ],
                &[
                    [
                        [1u.toUByte(), 0, 0, 1, 1],
                        [0, 0, 1, 0, 1],
                        [1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    [
                        [1, 0, 0, 1, 1],
                        [1, 0, 1, 1, 1],
                        [0, 1, 0, 1, 1],
                        [1, 1, 1, 0, 0],
                    ],
                    [
                        [0, 1, 1, 1, 0],
                        [1, 1, 0, 1, 0],
                        [0, 1, 1, 1, 0],
                        [1, 1, 0, 1, 0],
                    ],
                ],
                &[
                    [
                        [1u.toUByte(), 1, 1, 0, 0],
                        [0, 1, 0, 0, 1],
                        [0, 1, 1, 0, 1],
                        [0, 0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 1, 0],
                        [0, 1, 1, 1, 0],
                    ],
                    [
                        [1, 1, 1, 0, 1],
                        [0, 0, 1, 1, 0],
                        [1, 0, 1, 1, 1],
                        [0, 1, 0, 1, 1],
                    ],
                ],
            );

            // 4d
            test(
                &[
                    [
                        [[0u.toUByte(), 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 1]],
                        [[1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                    ],
                    [
                        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]],
                        [[1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1]],
                    ],
                ],
                &[
                    [
                        [[1u.toUByte(), 0, 1, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
                        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
                    ],
                    [
                        [[1, 1, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                        [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]],
                    ],
                ],
                &[
                    [
                        [[1u.toUByte(), 1, 0, 0], [1, 0, 1, 1], [0, 1, 1, 1]],
                        [[1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0]],
                    ],
                    [
                        [[0, 0, 1, 0], [1, 0, 1, 1], [1, 0, 1, 0]],
                        [[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0]],
                    ],
                ],
            );

            // tests based on: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Xor xorbroadcast
            // 3d vs 1d
            test(
                // Shape (3, 4, 5)
                &[
                    [
                        [0u.toUByte(), 0, 0, 0, 1],
                        [0, 1, 0, 1, 1],
                        [1, 0, 0, 1, 1],
                        [0, 0, 1, 0, 1],
                    ],
                    [
                        [0, 1, 0, 1, 1],
                        [1, 1, 0, 0, 1],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 1],
                    ],
                    [
                        [1, 1, 0, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 1, 1, 0, 1],
                        [1, 1, 0, 1, 1],
                    ],
                ],
                // shape (5)
                &[1u.toUByte(), 0, 0, 1, 1],
                // shape (3, 4, 5)
                &[
                    [
                        [1u.toUByte(), 0, 0, 1, 0],
                        [1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 0],
                    ],
                    [
                        [1, 1, 0, 0, 0],
                        [0, 1, 0, 1, 0],
                        [1, 1, 1, 0, 1],
                        [1, 0, 0, 1, 0],
                    ],
                    [
                        [0, 1, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0],
                        [0, 1, 0, 0, 0],
                    ],
                ],
            );

            // 3d vs 2d
            test(
                // Shape (3, 4, 5)
                &[
                    [
                        [0u.toUByte(), 0, 0, 0, 1],
                        [0, 1, 0, 1, 1],
                        [1, 0, 0, 1, 1],
                        [0, 0, 1, 0, 1],
                    ],
                    [
                        [0, 1, 0, 1, 1],
                        [1, 1, 0, 0, 1],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 1],
                    ],
                    [
                        [1, 1, 0, 1, 1],
                        [0, 0, 0, 1, 1],
                        [0, 1, 1, 0, 1],
                        [1, 1, 0, 1, 1],
                    ],
                ],
                // shape (4, 5)
                &[
                    [0u.toUByte(), 1, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [1, 1, 0, 1, 1],
                    [1, 1, 0, 1, 0],
                ],
                // shape (3, 4, 5)
                &[
                    [
                        [0u.toUByte(), 1, 0, 1, 1],
                        [0, 1, 1, 1, 1],
                        [0, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                    ],
                    [
                        [0, 0, 0, 0, 1],
                        [1, 1, 1, 0, 1],
                        [1, 0, 1, 0, 1],
                        [1, 1, 0, 1, 1],
                    ],
                    [
                        [1, 0, 0, 0, 1],
                        [0, 0, 1, 1, 1],
                        [1, 0, 1, 1, 0],
                        [0, 0, 0, 0, 1],
                    ],
                ],
            );

            // 4d vs 2d
            test(
                // Shape (2, 3, 3, 4)
                &[
                    [
                        [[1u.toUByte(), 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 0]],
                        [[1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1]],
                        [[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]],
                    ],
                    [
                        [[0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1]],
                        [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1]],
                        [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1]],
                    ],
                ],
                // shape (3, 4)
                &[[0u.toUByte(), 0, 1, 1], [1, 1, 1, 1], [0, 1, 0, 1]],
                // shape (2, 3, 3, 4)
                &[
                    [
                        [[1u.toUByte(), 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                        [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0]],
                        [[1, 0, 1, 1], [0, 0, 0, 1], [0, 1, 1, 0]],
                    ],
                    [
                        [[0, 1, 1, 0], [0, 0, 1, 0], [1, 1, 1, 0]],
                        [[1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 0]],
                        [[1, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]],
                    ],
                ],
            );

            // 4d vs 3d
            test(
                // Shape (2, 3, 3, 4)
                &[
                    [
                        [[1u.toUByte(), 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 0]],
                        [[1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1]],
                        [[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]],
                    ],
                    [
                        [[0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1]],
                        [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1]],
                        [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1]],
                    ],
                ],
                // shape (3, 3, 4)
                &[
                    [[1u.toUByte(), 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]],
                    [[0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]],
                    [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]],
                ],
                // shape (2, 3, 3, 4)
                &[
                    [
                        [[0u.toUByte(), 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]],
                        [[1, 0, 0, 1], [0, 1, 0, 0], [1, 1, 0, 0]],
                        [[1, 1, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0]],
                    ],
                    [
                        [[1, 0, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]],
                        [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 1, 0]],
                        [[1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]],
                    ],
                ],
            );

            // 4d vs 4d
            test(
                // Shape (1, 4, 1, 2)
                &[[[[1u.toUByte(), 0]], [[1, 0]], [[1, 0]], [[1, 1]]]],
                // shape (2, 1, 4, 2)
                &[
                    [[[0u.toUByte(), 0], [1, 1], [1, 1], [1, 1]]],
                    [[[0, 1], [1, 0], [0, 1], [0, 0]]],
                ],
                // shape (2, 4, 4, 2)
                &[
                    [
                        [[1u.toUByte(), 0], [0, 1], [0, 1], [0, 1]],
                        [[1, 0], [0, 1], [0, 1], [0, 1]],
                        [[1, 0], [0, 1], [0, 1], [0, 1]],
                        [[1, 1], [0, 0], [0, 0], [0, 0]],
                    ],
                    [
                        [[1, 1], [0, 0], [1, 1], [1, 0]],
                        [[1, 1], [0, 0], [1, 1], [1, 0]],
                        [[1, 1], [0, 0], [1, 1], [1, 0]],
                        [[1, 0], [0, 1], [1, 0], [1, 1]],
                    ],
                ],
            );

            fun test(input = impl NdArray, other: impl NdArray, expected: impl NdArray) {
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Xor",
                        domain = "",
                        attribute = mutableListOf(),
                        input = mutableListOf(INPUT_X, INPUT_Y),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = HashMap::from([
                    (INPUT_X, Tensor::new(input, Device.Cpu)?),
                    (INPUT_Y, Tensor::new(other, Device.Cpu)?),
                ]);

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

                val expected = Tensor::new(expected, Device.Cpu);

                when (expected.dims().size) {
                    0 -> {
                        assertEquals(expected.toVec0<UByte>(), z.toVec0<UByte>())
                    }
                    1 -> {
                        assertEquals(expected.toVec1<UByte>(), z.toVec1<UByte>())
                    }
                    2 -> {
                        assertEquals(expected.toVec2<UByte>(), z.toVec2<UByte>())
                    }
                    3 -> {
                        assertEquals(expected.toVec3<UByte>(), z.toVec3<UByte>())
                    }
                    4 -> {
                        // Candle has no method equivalent to `tovec4()`
                        // So, as a hack, we flatten it to a single dim vec to test the results
                        assertEquals(expected.flattenAll()?.toVec1<UByte>(), z.flattenAll()?.toVec1<UByte>())
                    }
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testSignOperation() {
        val manualGraph = createModelProtoWithGraph(GraphProto(
                node = mutableListOf(NodeProto(
                    opType = "Sign",
                    domain = "",
                    attribute = mutableListOf(),
                    input = mutableListOf(INPUT_X),
                    output = mutableListOf(OUTPUT_Z),
                    name = "",
                    docString = "",
                )),
                name = "",
                initializer = mutableListOf(),
                input = mutableListOf(),
                output = mutableListOf(ValueInfoProto(
                    name = OUTPUT_Z,
                    docString = "",
                    type = null,
                )),
                valueInfo = mutableListOf(),
                docString = "",
                sparseInitializer = mutableListOf(),
                quantizationAnnotation = mutableListOf(),
            ));

            val inputs: MutableMap<String, Tensor> = mutableMapOf();
            inputs.insert(
                INPUT_X,
                Tensor::new(mutableListOf(-2f32, -1., 0., 1., 2.), Device.Cpu)?,
            );
            val eval = simpleEval(manualGraph, inputs);

            val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
            assertEquals(mutableListOf(-1, -1, 0, 1, 1), z.toDtype(candle::DType::I64)?.toVec1<Long>())
    }

    @Test
    fun testSeluOperator() {
        {
                // Test 1: Default alpha and gamma
                val defaultGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Selu",
                        domain = "",
                        input = mutableListOf("input"),
                        output = mutableListOf("output"),

                    )),
                    input = mutableListOf(ValueInfoProto(
                        name = "input",

                    )),
                    output = mutableListOf(ValueInfoProto(
                        name = "output",
                        type = null,

                    )),

                ));

                val input = Tensor.fromVec(mutableListOf(-1.0f32, 0.0, 1.0, 2.0), (2, 2), Device.Cpu);
                val inputs = mutableMapOf();
                inputs["input"] = input

                val eval = simpleEval(&defaultGraph, inputs);
                val output = eval["output"]!!;
                val outVec = toVec2Round(output, 4);
                assertEquals(mutableListOf(mutableListOf(-1.1113, 0.0), mutableListOf(1.0507, 2.1014)), outVec)
            }

            {
                // Test 2: Change alpha and gamma
                val customGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Selu",
                        attribute = mutableListOf(
                            AttributeProto(
                                name = "alpha",
                                type = AttributeType::Float as i32,
                                f = 2.0,

                            ),
                            AttributeProto(
                                name = "gamma",
                                type = AttributeType::Float as i32,
                                f = 0.5,

                            ),
                        ),
                        input = mutableListOf("input"),
                        output = mutableListOf("output"),

                    )),
                    input = mutableListOf(ValueInfoProto(
                        name = "input",

                    )),
                    output = mutableListOf(ValueInfoProto(
                        name = "output",

                    )),

                ));

                val input = Tensor.fromVec(mutableListOf(-1.0f32, 0.0, 1.0, 2.0), (2, 2), Device.Cpu);
                val inputs = mutableMapOf();
                inputs["input"] = input
                val eval = simpleEval(&customGraph, inputs);
                val output = eval["output"]!!;
                val outVec = toVec2Round(output, 4);
                assertEquals(mutableListOf(mutableListOf(-0.6321, 0.0), mutableListOf(0.5, 1.0)), outVec)
            }

            {
                // Test 3: Different input values
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Selu",
                        domain = "",
                        input = mutableListOf("input"),
                        output = mutableListOf("output"),

                    )),
                    input = mutableListOf(ValueInfoProto(
                        name = "input",

                    )),
                    output = mutableListOf(ValueInfoProto(
                        name = "output",

                    )),

                ));

                val expected = mutableListOf(-1.758, -1.7463, 0.0, 10.507);

                val input = Tensor.fromVec(mutableListOf(-10.0f32, -5.0, 0.0, 10.0), (2, 2), Device.Cpu);
                val inputs = mutableMapOf();
                inputs["input"] = input
                val eval = simpleEval(manualGraph, inputs);
                val output = eval["output"]!!;
                val outVec = toVec2Round(output, 4);
                assertEquals(mutableListOf(
                        mutableListOf(expected[0], expected[1]),
                        mutableListOf(expected[2], expected[3])
                    ), outVec)
            }

            {
                // Test 4: Test based on https://github.com/onnx/onnx/blob/main/docs/Operators.md#Selu
                val graph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Selu",
                        input = mutableListOf("input"),
                        output = mutableListOf("output"),
                        attribute = mutableListOf(
                            AttributeProto(
                                name = "alpha",
                                type = AttributeType::Float as i32,
                                f = 2.0,

                            ),
                            AttributeProto(
                                name = "gamma",
                                type = AttributeType::Float as i32,
                                f = 3.0,

                            ),
                        ),

                    )),
                    input = mutableListOf(ValueInfoProto(
                        name = "input",

                    )),
                    output = mutableListOf(ValueInfoProto(
                        name = "output",

                    )),

                ));

                val input = Tensor.fromVec(mutableListOf(-1.0f32, 0.0, 1.0), (3,), Device.Cpu);
                val inputs = mutableMapOf();
                inputs["input"] = input

                val eval = simpleEval(&graph, inputs);
                val output = eval["output"]!!;
                val outVec = output.toVec1<Float>();
                val expected = mutableListOf(-3.7927232, 0.0, 3.0);

                for (o, e) in outVec.iter().zip(expected.iter()) {
                    assertTrue((o - e).abs() < 1e-5, "Got {o}, expected {e}")
                }
            }

            {
                // Test 5: Empty tensor
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Selu",
                        domain = "",
                        input = mutableListOf("input"),
                        output = mutableListOf("output"),

                    )),
                    input = mutableListOf(ValueInfoProto(
                        name = "input",

                    )),
                    output = mutableListOf(ValueInfoProto(
                        name = "output",

                    )),

                ));

                val input = Tensor.fromVec(mutableListOf() as Vec<f32>, (0, 2), Device.Cpu);
                val inputs = mutableMapOf();
                inputs["input"] = input
                val eval = simpleEval(manualGraph, inputs);
                val output = eval["output"]!!;
                assertEquals(&[0, 2], output.dims())
            }
    }

    @Test
    fun testHardSwish() {
        {
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "HardSwish",
                        input = mutableListOf(INPUT_X),
                        output = mutableListOf(OUTPUT_Z),

                    )),
                    input = mutableListOf(ValueInfoProto(
                        name = INPUT_X,

                    )),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,

                    )),

                ));
                val inputData = mutableListOf(-4.0f32, -3.0, 0.0, 2.0, 3.0, 5.0);
                val inputTensor = Tensor.fromVec(inputData, (inputData.size,), Device.Cpu);
                val inputs = mutableMapOf();
                inputs[INPUT_X] = inputTensor

                val outputs = simpleEval(manualGraph, inputs);
                val output = outputs[OUTPUT_Z] ?: error("missing output Z");
                val outputVec = output.toVec1<Float>();

                val expected = mutableListOf(0.0, 0.0, 0.0, 1.6666666, 3.0, 5.0);

                for (i, (got, exp)) in outputVec.iter().zip(expected.iter()).enumerate() {
                    val diff = (got - exp).abs();
                    assertTrue(diff < 1e-4,
                        "Mismatch at index {i}: got {got}, expected {exp}, diff={diff}")
                }
            }
            {
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "HardSwish",
                        input = mutableListOf(INPUT_X),
                        output = mutableListOf(OUTPUT_Z),

                    )),
                    input = mutableListOf(ValueInfoProto(
                        name = INPUT_X,

                    )),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,

                    )),

                ));
                val inputData = mutableListOf(-4.0f32, -2.0, 0.0, 2.0, 4.0);
                val inputTensor = Tensor.fromVec(inputData, (inputData.size,), Device.Cpu);
                val inputs = mutableMapOf();
                inputs[INPUT_X] = inputTensor

                val outputs = simpleEval(manualGraph, inputs);
                val output = outputs[OUTPUT_Z] ?: error("missing output Z");
                val outputVec = output.toVec1<Float>();

                val expected = mutableListOf(0.0, -0.33333334, 0.0, 1.6666667, 4.0);

                for (i, (got, exp)) in outputVec.iter().zip(expected.iter()).enumerate() {
                    val diff = (got - exp).abs();
                    assertTrue(diff < 1e-4,
                        "Mismatch at index {i}: got {got}, expected {exp}, diff={diff}")
                }
            }
    }

    @Test
    fun testScatterndOperation() {
        // Example 1 based on ONNX documentation
            test(
                &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &[[4i64], [3], [1], [7]],
                &[9.0f32, 10.0, 11.0, 12.0],
                &[1.0f32, 11.0, 3.0, 10.0, 9.0, 6.0, 7.0, 12.0],
            );

            // A more complex example with 2D data
            test(
                &[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]],
                &[[0i64, 1], [1, 0]],
                &[10.0f32, 20.0],
                &[[1.0f32, 10.0], [20.0, 4.0], [5.0, 6.0]],
            );

            // 3D example with indices pointing to specific locations
            test(
                &[
                    [[1.0f32, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[9.0, 10.0], [11.0, 12.0]],
                ],
                &[[0i64, 0, 1], [1, 1, 0]],
                &[100.0f32, 200.0],
                &[
                    [[1.0f32, 100.0], [3.0, 4.0]],
                    [[5.0, 6.0], [200.0, 8.0]],
                    [[9.0, 10.0], [11.0, 12.0]],
                ],
            );

            fun test(
                data: impl NdArray,
                indices: impl NdArray,
                updates: impl NdArray,
                expected: impl NdArray,
            ) {
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "ScatterND",
                        domain = "",
                        attribute = mutableListOf(),
                        input = mutableListOf(
                            INPUT_X,
                            INPUT_Y,
                            INPUT_A,
                        ),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(INPUT_X, Tensor::new(data, Device.Cpu)?);
                inputs.insert(INPUT_Y, Tensor::new(indices, Device.Cpu)?);
                inputs.insert(INPUT_A, Tensor::new(updates, Device.Cpu)?);

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
                val expected = Tensor::new(expected, Device.Cpu);

                when (expected.dims().size) {
                    1 -> assertEquals(expected.toVec1<Float>(), z.toVec1<Float>()),
                    2 -> assertEquals(expected.toVec2<Float>(), z.toVec2<Float>()),
                    3 -> assertEquals(expected.toVec3<Float>(), z.toVec3<Float>()),
                    else -> error("unreachable"),
                };


            }
    }

    @Test
    fun testTriluOperation() {
        // Test 1: Upper triangular matrix (default behavior with upper=true)
            {
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Trilu",
                        domain = "",
                        attribute = mutableListOf(), // empty attribute means default upper=true
                        input = mutableListOf(INPUT_X),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(ValueInfoProto(
                        name = INPUT_X,
                        docString = "",
                        type = null,
                    )),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val x = Tensor.fromVec(
                    mutableListOf(
                        4i64, 7, 3, 7, 9, 1, 2, 8, 6, 9, 9, 4, 0, 8, 7, 4, 3, 4, 2, 4,
                    ),
                    &[4, 5],
                    Device.Cpu,
                );

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs[INPUT_X] = x

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
                val results = z.toVec2<Long>();

                assertEquals(mutableListOf(
                        mutableListOf(4, 7, 3, 7, 9),
                        mutableListOf(0, 2, 8, 6, 9),
                        mutableListOf(0, 0, 0, 8, 7),
                        mutableListOf(0, 0, 0, 2, 4)
                    ), results)
            }

            // Test 2: Upper triangular with positive k=1 (diagonal above main)
            {
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Trilu",
                        domain = "",
                        attribute = mutableListOf(),
                        input = mutableListOf(INPUT_X, INPUT_Y),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(
                        ValueInfoProto(
                            name = INPUT_X,
                            docString = "",
                            type = null,
                        ),
                        ValueInfoProto(
                            name = INPUT_Y,
                            docString = "",
                            type = null,
                        ),
                    ),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val x = Tensor.fromVec(
                    mutableListOf(1i64, 4, 9, 7, 1, 9, 2, 8, 8, 4, 3, 9, 7, 4, 2),
                    &[3, 5],
                    Device.Cpu,
                );

                val k = Tensor.fromVec(mutableListOf(1i64), (), Device.Cpu);

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs[INPUT_X] = x
                inputs[INPUT_Y] = k

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
                val results = z.toVec2<Long>();

                assertEquals(mutableListOf(
                        mutableListOf(0, 4, 9, 7, 1),
                        mutableListOf(0, 0, 8, 8, 4),
                        mutableListOf(0, 0, 0, 4, 2)
                    ), results)
            }

            // Test 3: Upper triangular with negative k=-1 (one diagonal below main)
            {
                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Trilu",
                        domain = "",
                        attribute = mutableListOf(),
                        input = mutableListOf(INPUT_X, INPUT_Y),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val x = Tensor.fromVec(
                    mutableListOf(
                        4i64, 7, 3, 7, 9, 1, 2, 8, 6, 9, 9, 4, 0, 8, 7, 4, 3, 4, 2, 4,
                    ),
                    &[4, 5],
                    Device.Cpu,
                );

                val k = Tensor.fromVec(mutableListOf(-1i64), (), Device.Cpu);

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs[INPUT_X] = x
                inputs[INPUT_Y] = k

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
                val results = z.toVec2<Long>();

                assertEquals(mutableListOf(
                        mutableListOf(4, 7, 3, 7, 9),
                        mutableListOf(1, 2, 8, 6, 9),
                        mutableListOf(0, 4, 0, 8, 7),
                        mutableListOf(0, 0, 4, 2, 4)
                    ), results)
            }

            // Test 4: Lower triangular matrix (upper=0)
            {
                val attUpper = AttributeProto(
                    name = "upper",
                    refAttrName = "upper",
                    i = 0, // 0 means false, use lower triangular
                    docString = "upper",
                    type = 2,
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );

                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Trilu",
                        domain = "",
                        attribute = mutableListOf(attUpper),
                        input = mutableListOf(INPUT_X),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val x = Tensor.fromVec(
                    mutableListOf(
                        4i64, 7, 3, 7, 9, 1, 2, 8, 6, 9, 9, 4, 1, 8, 7, 4, 3, 4, 2, 4,
                    ),
                    &[4, 5],
                    Device.Cpu,
                );

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs[INPUT_X] = x

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
                val results = z.toVec2<Long>();

                // Lower triangular matrix (default k=0)
                assertEquals(mutableListOf(
                        mutableListOf(4, 0, 0, 0, 0),
                        mutableListOf(1, 2, 0, 0, 0),
                        mutableListOf(9, 4, 1, 0, 0),
                        mutableListOf(4, 3, 4, 2, 0)
                    ), results)
            }

            // Test 5: Lower triangular with negative k=-1
            {
                val attUpper = AttributeProto(
                    name = "upper",
                    refAttrName = "upper",
                    i = 0,
                    docString = "upper",
                    type = 2,
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );

                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Trilu",
                        domain = "",
                        attribute = mutableListOf(attUpper),
                        input = mutableListOf(INPUT_X, INPUT_Y),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val x = Tensor.fromVec(
                    mutableListOf(
                        4i64, 7, 3, 7, 9, 1, 2, 8, 6, 9, 9, 4, 1, 8, 7, 4, 3, 4, 2, 4,
                    ),
                    &[4, 5],
                    Device.Cpu,
                );

                val k = Tensor.fromVec(mutableListOf(-1i64), (), Device.Cpu);

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs[INPUT_X] = x
                inputs[INPUT_Y] = k

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
                val results = z.toVec2<Long>();

                assertEquals(mutableListOf(
                        mutableListOf(0, 0, 0, 0, 0),
                        mutableListOf(1, 0, 0, 0, 0),
                        mutableListOf(9, 4, 0, 0, 0),
                        mutableListOf(4, 3, 4, 0, 0)
                    ), results)
            }

            // Test 6: Lower triangular with positive k=2
            {
                val attUpper = AttributeProto(
                    name = "upper",
                    refAttrName = "upper",
                    i = 0,
                    docString = "upper",
                    type = 2,
                    f = 0.0,
                    s = mutableListOf(),
                    t = null,
                    g = null,
                    sparseTensor = null,
                    tp = null,
                    floats = mutableListOf(),
                    ints = mutableListOf(),
                    strings = mutableListOf(),
                    tensors = mutableListOf(),
                    graphs = mutableListOf(),
                    sparseTensors = mutableListOf(),
                    typeProtos = mutableListOf(),
                );

                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "Trilu",
                        domain = "",
                        attribute = mutableListOf(attUpper),
                        input = mutableListOf(INPUT_X, INPUT_Y),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val x = Tensor.fromVec(
                    mutableListOf(
                        4i64, 7, 3, 7, 9, 1, 2, 8, 6, 9, 9, 4, 1, 8, 7, 4, 3, 4, 2, 4,
                    ),
                    &[4, 5],
                    Device.Cpu,
                );

                val k = Tensor.fromVec(mutableListOf(2i64), (), Device.Cpu);

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs[INPUT_X] = x
                inputs[INPUT_Y] = k

                val eval = simpleEval(manualGraph, inputs);
                assertEquals(1, eval.size)

                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");
                val results = z.toVec2<Long>();

                assertEquals(mutableListOf(
                        mutableListOf(4, 7, 3, 0, 0),
                        mutableListOf(1, 2, 8, 6, 0),
                        mutableListOf(9, 4, 1, 8, 7),
                        mutableListOf(4, 3, 4, 2, 4)
                    ), results)
            }
    }

    @Test
    fun testOneHot() {
        // Tests based on: https://github.com/onnx/onnx/blob/main/docs/Operators.md#OneHot
            {
                val depthValue = Tensor::new(3i64, Device.Cpu); // depth = 3
                val valuesTensor = Tensor.fromVec(mutableListOf(0.0f32, 1.0), (2,), Device.Cpu); // off = 0.0, on = 1.0

                val manualGraph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "OneHot",
                        domain = "",
                        attribute = mutableListOf(AttributeProto(
                            name = "axis",
                            type = AttributeType::Int as i32,
                            i = -1,

                        )),
                        input = mutableListOf(
                            INPUT_X,  // indices
                            "depth",  // depth
                            "values", // values
                        ),
                        output = mutableListOf(OUTPUT_Z),
                        name = "",
                        docString = "",
                    )),
                    name = "",
                    initializer = mutableListOf(),
                    input = mutableListOf(),
                    output = mutableListOf(ValueInfoProto(
                        name = OUTPUT_Z,
                        docString = "",
                        type = null,
                    )),
                    valueInfo = mutableListOf(),
                    docString = "",
                    sparseInitializer = mutableListOf(),
                    quantizationAnnotation = mutableListOf(),
                ));

                val inputs: MutableMap<String, Tensor> = mutableMapOf();
                inputs.insert(
                    INPUT_X,
                    Tensor::new(mutableListOf(0i64, 1, 2), Device.Cpu)?,
                );
                inputs["depth"] = depthValue
                inputs["values"] = valuesTensor

                val eval = simpleEval(manualGraph, inputs);
                val z = eval[OUTPUT_Z] ?: error("Output 'z' not found");

                val expected = mutableListOf(
                    mutableListOf(1.0, 0.0, 0.0),
                    mutableListOf(0.0, 1.0, 0.0),
                    mutableListOf(0.0, 0.0, 1.0),
                );

                val zReshaped = z.toDtype(DType.F32).reshape((3, 3))?.toVec2<Float>();
                assertEquals(expected, zReshaped)
            }
            {
                // Test with axis
                val indices = Tensor.fromVec(mutableListOf(1i64, 9, 2, 4), (2, 2), Device.Cpu);
                val depth = Tensor::new(10i64, Device.Cpu);
                val values = Tensor.fromVec(mutableListOf(1.0f32, 3.0), (2,), Device.Cpu);

                val graph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "OneHot",
                        input = mutableListOf("indices".into(), "depth".into(), "values".into()),
                        output = mutableListOf("y".into()),
                        attribute = mutableListOf(AttributeProto(
                            name = "axis".into(),
                            type = AttributeType::Int as i32,
                            i = 1,

                        )),

                    )),
                    output = mutableListOf(ValueInfoProto(
                        name = "y".into(),

                    )),

                ));

                val inputs = mutableMapOf();
                inputs["indices".into()] = indices
                inputs["depth".into()] = depth
                inputs["values".into()] = values

                val eval = simpleEval(&graph, inputs);
                val y = eval["y"]!!;
                assertEquals(&[2, 10, 2], y.dims())
            }
            {
                // Test with negative axis
                val indices = Tensor.fromVec(mutableListOf(1i64, 9, 2, 4), (2, 2), Device.Cpu);
                val depth = Tensor::new(10i64, Device.Cpu);
                val values = Tensor.fromVec(mutableListOf(1.0f32, 3.0), (2,), Device.Cpu);

                val graph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "OneHot",
                        input = mutableListOf("indices".into(), "depth".into(), "values".into()),
                        output = mutableListOf("y".into()),
                        attribute = mutableListOf(AttributeProto(
                            name = "axis".into(),
                            type = AttributeType::Int as i32,
                            i = -2,

                        )),

                    )),
                    output = mutableListOf(ValueInfoProto(
                        name = "y".into(),

                    )),

                ));

                val inputs = mutableMapOf();
                inputs["indices".into()] = indices
                inputs["depth".into()] = depth
                inputs["values".into()] = values

                val eval = simpleEval(&graph, inputs);
                val y = eval["y"]!!;
                assertEquals(&[2, 10, 2], y.dims())
            }
            {
                // Test with negative indices
                val indices = Tensor.fromVec(mutableListOf(0i64, -7, -8), (3,), Device.Cpu);
                val depth = Tensor::new(10i64, Device.Cpu);
                val values = Tensor.fromVec(mutableListOf(1.0f32, 3.0), (2,), Device.Cpu);

                val graph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "OneHot",
                        input = mutableListOf("indices".into(), "depth".into(), "values".into()),
                        output = mutableListOf("y".into()),
                        attribute = mutableListOf(AttributeProto(
                            name = "axis".into(),
                            type = AttributeType::Int as i32,
                            i = 1,

                        )),

                    )),
                    output = mutableListOf(ValueInfoProto(
                        name = "y".into(),

                    )),

                ));

                val inputs = mutableMapOf();
                inputs["indices".into()] = indices
                inputs["depth".into()] = depth
                inputs["values".into()] = values

                val eval = simpleEval(&graph, inputs);
                val y = eval["y"]!!;
                assertEquals(&[3, 10], y.dims())
            }
            {
                // Test without axis
                val indices = Tensor.fromVec(mutableListOf(0i64, 7, 8), (3,), Device.Cpu);
                val depth = Tensor::new(12i64, Device.Cpu);
                val values = Tensor.fromVec(mutableListOf(2f32, 5.0), (2,), Device.Cpu);

                val graph = createModelProtoWithGraph(GraphProto(
                    node = mutableListOf(NodeProto(
                        opType = "OneHot",
                        input = mutableListOf("indices".into(), "depth".into(), "values".into()),
                        output = mutableListOf("y".into()),

                    )),
                    output = mutableListOf(ValueInfoProto(
                        name = "y".into(),

                    )),

                ));

                val inputs = mutableMapOf();
                inputs["indices".into()] = indices
                inputs["depth".into()] = depth
                inputs["values".into()] = values

                val eval = simpleEval(&graph, inputs);
                val y = eval["y"]!!;
                assertEquals(&[3, 12], y.dims())
            }
    }
}
