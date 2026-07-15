package com.example.rag;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.List;

/** One line of qa.jsonl. */
@JsonIgnoreProperties(ignoreUnknown = true)
public final class QaEntry {
    public long id;
    public String question;
    public String answer;
    public List<String> tags = List.of();
}
