package com.example.rag;

import io.qdrant.client.QdrantClient;
import io.qdrant.client.QdrantGrpcClient;
import io.qdrant.client.grpc.Collections.Distance;
import io.qdrant.client.grpc.Collections.VectorParams;
import io.qdrant.client.grpc.Points.PointStruct;
import io.qdrant.client.grpc.Points.QueryPoints;
import io.qdrant.client.grpc.Points.ScoredPoint;
import io.qdrant.client.grpc.JsonWithInt.Value;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static io.qdrant.client.PointIdFactory.id;
import static io.qdrant.client.QueryFactory.nearest;
import static io.qdrant.client.ValueFactory.value;
import static io.qdrant.client.ValueFactory.list;
import static io.qdrant.client.VectorsFactory.vectors;

/** Qdrant lifecycle, upsert, and search over the shared 'faq' collection (gRPC, port 6334). */
public final class QdrantStore implements AutoCloseable {

    public static final String COLLECTION = "faq";
    public static final int DIMENSIONS = 1024;
    public static final int DEFAULT_LIMIT = 3;
    // bge-m3 scores: relevant ~0.60-0.71, off-topic <=0.43 on this corpus.
    public static final float DEFAULT_THRESHOLD = 0.50f;

    private final QdrantClient client;

    public QdrantStore() {
        // localhost for a native run; docker compose sets QDRANT_HOST=qdrant.
        String host = System.getenv().getOrDefault("QDRANT_HOST", "localhost");
        int port = Integer.parseInt(System.getenv().getOrDefault("QDRANT_PORT", "6334"));
        client = new QdrantClient(
                QdrantGrpcClient.newBuilder(host, port, false).build());
    }

    public void recreateCollection() throws Exception {
        if (client.collectionExistsAsync(COLLECTION).get()) {
            client.deleteCollectionAsync(COLLECTION).get();
        }
        client.createCollectionAsync(COLLECTION,
                VectorParams.newBuilder()
                        .setSize(DIMENSIONS)
                        .setDistance(Distance.Cosine)
                        .build()
        ).get();
    }

    public void upsert(List<QaEntry> entries, float[][] vectors) throws Exception {
        List<PointStruct> points = new ArrayList<>();
        for (int i = 0; i < entries.size(); i++) {
            QaEntry e = entries.get(i);
            List<Value> tagValues = new ArrayList<>();
            for (String t : e.tags) tagValues.add(value(t));
            Map<String, Value> payload = Map.of(
                    "question", value(e.question),
                    "answer", value(e.answer),
                    "tags", list(tagValues)
            );
            points.add(PointStruct.newBuilder()
                    .setId(id(e.id))
                    .setVectors(vectors(toFloatList(vectors[i])))
                    .putAllPayload(payload)
                    .build());
        }
        client.upsertAsync(COLLECTION, points).get();
    }

    public List<Hit> search(float[] queryVector, int limit, float threshold) throws Exception {
        List<ScoredPoint> results = client.queryAsync(
                QueryPoints.newBuilder()
                        .setCollectionName(COLLECTION)
                        .setQuery(nearest(toFloatList(queryVector)))
                        .setLimit(limit)
                        .setScoreThreshold(threshold)
                        .setWithPayload(io.qdrant.client.WithPayloadSelectorFactory.enable(true))
                        .build()
        ).get();

        List<Hit> hits = new ArrayList<>();
        for (ScoredPoint p : results) {
            Map<String, Value> payload = p.getPayloadMap();
            List<String> tags = new ArrayList<>();
            if (payload.containsKey("tags")) {
                for (Value v : payload.get("tags").getListValue().getValuesList()) {
                    tags.add(v.getStringValue());
                }
            }
            hits.add(new Hit(
                    p.getScore(),
                    payload.get("question").getStringValue(),
                    payload.get("answer").getStringValue(),
                    tags));
        }
        return hits;
    }

    private static List<Float> toFloatList(float[] v) {
        List<Float> out = new ArrayList<>(v.length);
        for (float f : v) out.add(f);
        return out;
    }

    @Override
    public void close() {
        client.close();
    }
}
