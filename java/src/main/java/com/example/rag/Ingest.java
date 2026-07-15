package com.example.rag;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Recreate the 'faq' collection and index qa.jsonl.
 * Usage: mvn exec:java -Dexec.mainClass=com.example.rag.Ingest [-Dexec.args="path/to/qa.jsonl"]
 */
public final class Ingest {

    public static void main(String[] args) throws Exception {
        Path path = args.length > 0 ? Path.of(args[0]) : defaultQaPath();
        ObjectMapper mapper = new ObjectMapper();

        List<QaEntry> entries = new ArrayList<>();
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            if (line.isBlank()) continue;
            entries.add(mapper.readValue(line, QaEntry.class));
        }

        // Embed question + answer as one chunk, matching the Python ingest.
        List<String> chunks = new ArrayList<>();
        for (QaEntry e : entries) chunks.add(e.question + "\n" + e.answer);

        try (Embedder embedder = new Embedder(Embedder.defaultModelDir());
             QdrantStore store = new QdrantStore()) {
            float[][] vectors = embedder.embedDocuments(chunks);
            store.recreateCollection();
            store.upsert(entries, vectors);
            System.out.printf("Indexed %d entries from %s into '%s'.%n",
                    entries.size(), path, QdrantStore.COLLECTION);
        }
    }

    private static Path defaultQaPath() {
        Path local = Path.of("qa.jsonl");
        if (local.toFile().exists()) return local;
        return Path.of("../qa.jsonl"); // when run from java/
    }
}
