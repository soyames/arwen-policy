# Retrieval

The retrieval layer turns released Arwen Policy Corpus records into evidence candidates for policy analysis.

Current implementation: deterministic, dependency-light BM25-style lexical retrieval with metadata filters and explicit provenance references.

Production extensions planned:

- vector retrieval and reranking;
- multilingual embeddings;
- corpus snapshot loading from Hugging Face;
- stakeholder-aware diversification;
- argument/evidence graph retrieval;
- hybrid lexical + semantic ranking.

The retrieval layer must never detach an answer from its source, document and segment identifiers.
