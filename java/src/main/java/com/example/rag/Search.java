package com.example.rag;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Search the shared index.
 * Usage: mvn exec:java -Dexec.mainClass=com.example.rag.Search -Dexec.args="how do I undo a release?"
 * With no query, drops into a REPL.
 */
public final class Search {

    public static void main(String[] args) throws Exception {
        int limit = QdrantStore.DEFAULT_LIMIT;
        float threshold = QdrantStore.DEFAULT_THRESHOLD;

        // Parse: positional query words plus optional --limit / --threshold.
        StringBuilder queryBuilder = new StringBuilder();
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--limit") && i + 1 < args.length) {
                limit = Integer.parseInt(args[++i]);
            } else if (args[i].equals("--threshold") && i + 1 < args.length) {
                threshold = Float.parseFloat(args[++i]);
            } else {
                if (queryBuilder.length() > 0) queryBuilder.append(' ');
                queryBuilder.append(args[i]);
            }
        }
        String query = queryBuilder.toString().trim();

        try (Embedder embedder = new Embedder(Embedder.defaultModelDir());
             QdrantStore store = new QdrantStore()) {

            if (!query.isEmpty()) {
                runQuery(embedder, store, query, limit, threshold);
                return;
            }

            System.out.println("Semantic search REPL — empty line to exit.");
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(System.in, StandardCharsets.UTF_8));
            while (true) {
                System.out.print("query> ");
                String line = reader.readLine();
                if (line == null || line.trim().isEmpty()) break;
                runQuery(embedder, store, line.trim(), limit, threshold);
            }
        }
    }

    private static void runQuery(Embedder embedder, QdrantStore store,
                                 String query, int limit, float threshold) throws Exception {
        float[] vector = embedder.embedQuery(query);
        List<Hit> hits = store.search(vector, limit, threshold);
        if (hits.isEmpty()) {
            System.out.println("No confident match found.");
            return;
        }
        int rank = 1;
        for (Hit h : hits) {
            String answer = h.answer();
            String snippet = answer.length() <= 120 ? answer : answer.substring(0, 117) + "...";
            System.out.printf("%d. [%.4f] %s%n", rank++, h.score(), h.question());
            System.out.printf("   %s%n", snippet);
        }
    }
}
