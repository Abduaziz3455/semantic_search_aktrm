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
 * Embeds text with BAAI/bge-m3 via ONNX Runtime, reproducing the
 * sentence-transformers pipeline exactly: tokenize (max 512, padded),
 * take the [CLS] token from the last hidden state, L2 normalize.
 *
 * bge-m3 dense retrieval is symmetric — no query/document prefixes.
 */
public final class Embedder implements AutoCloseable {

    public static final int DIMENSIONS = 1024;

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
        return embedBatch(texts.toArray(new String[0]));
    }

    public float[] embedQuery(String text) throws Exception {
        return embedBatch(new String[]{text})[0];
    }

    /** Entry point for the cross-runtime vector parity test. */
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
                float[][][] hidden = (float[][][]) result.get(0).getValue(); // [batch, seq, 1024]
                float[][] out = new float[batch][];
                for (int i = 0; i < batch; i++) {
                    out[i] = pool(hidden[i]);
                }
                return out;
            }
        } finally {
            inputs.values().forEach(OnnxTensor::close);
        }
    }

    /**
     * bge-m3 dense pooling: take the [CLS] token (position 0) of the last
     * hidden state, then L2 normalize.
     */
    private static float[] pool(float[][] hidden) {
        float[] cls = hidden[0].clone();
        double norm = 0;
        for (int d = 0; d < DIMENSIONS; d++) {
            norm += (double) cls[d] * cls[d];
        }
        norm = Math.sqrt(norm);
        for (int d = 0; d < DIMENSIONS; d++) {
            cls[d] /= (float) norm;
        }
        return cls;
    }

    @Override
    public void close() throws Exception {
        session.close();
        tokenizer.close();
    }

    public static Path defaultModelDir() {
        // docker compose sets MODEL_DIR; otherwise resolve relative to the repo.
        String env = System.getenv("MODEL_DIR");
        if (env != null && !env.isBlank()) return Path.of(env);
        Path local = Path.of("models/bge-m3");
        if (local.resolve("model.onnx").toFile().exists()) return local;
        return Path.of("java/models/bge-m3");
    }

    /** Vector parity check: prints all 1024 floats for "hello world".
     *  Joins all args because exec:java splits the argument string on spaces. */
    public static void main(String[] args) throws Exception {
        String text = args.length > 0 ? String.join(" ", args) : "hello world";
        try (Embedder e = new Embedder(defaultModelDir())) {
            float[] v = e.embedRaw(text);
            StringBuilder sb = new StringBuilder();
            for (float f : v) sb.append(String.format("%.6f%n", f));
            System.out.print(sb);
        }
    }
}
