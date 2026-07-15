package com.example.rag;

import java.util.List;

/** One search result. */
public record Hit(float score, String question, String answer, List<String> tags) {}
