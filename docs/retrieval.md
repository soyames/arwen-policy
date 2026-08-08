# Retrieval Layer

The baseline retriever is deterministic and lexical so that the research system has an auditable baseline before semantic retrieval is introduced.

## Contract

Input: `RetrievalQuery`.

Output: ranked `RetrievedItem` objects and `EvidenceReference` objects.

Every evidence reference includes `record_id`, `source_id`, `document_id`, optional `segment_id`, URL and retrieval score.

Future vector retrieval must implement the same contract.
