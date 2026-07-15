package com.example.rag;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Embeds text with multilingual-e5-small via ONNX Runtime, reproducing the
 * sentence-transformers pipeline exactly: tokenize (max 512, padded),
 * mask-weighted mean pooling over the last hidden state, L2 normalize.
 *
 * E5 is asymmetric: passages and queries take different prefixes. All
 * prefixing lives here so it cannot drift between Ingest and Search.
 */
public final class Embedder implements AutoCloseable {

    public static final String PASSAGE_PREFIX = "passage: ";
    public static final String QUERY_PREFIX = "query: ";
    public static final int DIMENSIONS = 384;

    private final HuggingFaceTokenizer tokenizer;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final Set<String> inputNames;

    public Embedder(Path modelDir) throws Exception {
        tokenizer = HuggingFaceTokenizer.builder()
                .optTokenizerPath(modelDir.resolve("tokenizer.json"))
                .optPadding(true)
                .optTruncation(true)
                .optMaxLength(512)
                .build();
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelDir.resolve("model.onnx").toString(),
                new OrtSession.SessionOptions());
        // Bind only what the graph declares — this export may or may not
        // take token_type_ids, so never hardcode the input set.
        inputNames = session.getInputNames();
    }

    public float[][] embedDocuments(List<String> texts) throws Exception {
        String[] prefixed = texts.stream().map(t -> PASSAGE_PREFIX + t).toArray(String[]::new);
        return embedBatch(prefixed);
    }

    public float[] embedQuery(String text) throws Exception {
        return embedBatch(new String[]{QUERY_PREFIX + text})[0];
    }

    /** No prefix — for the vector parity test, which supplies one itself. */
    public float[] embedRaw(String text) throws Exception {
        return embedBatch(new String[]{text})[0];
    }

    private float[][] embedBatch(String[] texts) throws Exception {
        Encoding[] encodings = tokenizer.batchEncode(texts);
        int batch = encodings.length;
        int seq = encodings[0].getIds().length; // uniform: padding is on

        long[] ids = new long[batch * seq];
        long[] mask = new long[batch * seq];
        long[] types = new long[batch * seq];
        for (int i = 0; i < batch; i++) {
            System.arraycopy(encodings[i].getIds(), 0, ids, i * seq, seq);
            System.arraycopy(encodings[i].getAttentionMask(), 0, mask, i * seq, seq);
            System.arraycopy(encodings[i].getTypeIds(), 0, types, i * seq, seq);
        }

        long[] shape = {batch, seq};
        Map<String, OnnxTensor> inputs = new HashMap<>();
        try {
            inputs.put("input_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape));
            inputs.put("attention_mask", OnnxTensor.createTensor(env, LongBuffer.wrap(mask), shape));
            if (inputNames.contains("token_type_ids")) {
                inputs.put("token_type_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(types), shape));
            }

            try (OrtSession.Result result = session.run(inputs)) {
                float[][][] hidden = (float[][][]) result.get(0).getValue(); // [batch, seq, 384]
                float[][] out = new float[batch][];
                for (int i = 0; i < batch; i++) {
                    out[i] = pool(hidden[i], encodings[i].getAttentionMask());
                }
                return out;
            }
        } finally {
            inputs.values().forEach(OnnxTensor::close);
        }
    }

    /**
     * Mean pooling weighted by the attention mask, then L2 normalization.
     * Averaging over padding tokens would corrupt the vector — only real
     * tokens contribute.
     */
    private static float[] pool(float[][] hidden, long[] mask) {
        float[] sum = new float[DIMENSIONS];
        long count = 0;
        for (int t = 0; t < hidden.length; t++) {
            if (mask[t] == 0) continue;
            count++;
            for (int d = 0; d < DIMENSIONS; d++) {
                sum[d] += hidden[t][d];
            }
        }
        double norm = 0;
        for (int d = 0; d < DIMENSIONS; d++) {
            sum[d] /= count;
            norm += (double) sum[d] * sum[d];
        }
        norm = Math.sqrt(norm);
        for (int d = 0; d < DIMENSIONS; d++) {
            sum[d] /= (float) norm;
        }
        return sum;
    }

    @Override
    public void close() throws Exception {
        session.close();
        tokenizer.close();
    }

    public static Path defaultModelDir() {
        // works from repo root or from java/
        Path local = Path.of("models/multilingual-e5-small");
        if (local.resolve("model.onnx").toFile().exists()) return local;
        return Path.of("java/models/multilingual-e5-small");
    }

    /** Vector parity check: prints all 384 floats for "passage: hello world".
     *  Joins all args because exec:java splits the argument string on spaces. */
    public static void main(String[] args) throws Exception {
        String text = args.length > 0 ? String.join(" ", args) : "passage: hello world";
        try (Embedder e = new Embedder(defaultModelDir())) {
            float[] v = e.embedRaw(text);
            StringBuilder sb = new StringBuilder();
            for (float f : v) sb.append(String.format("%.6f%n", f));
            System.out.print(sb);
        }
    }
}
