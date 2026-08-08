# Security

Arwen Policy ETL processes untrusted public content.

## Important rules

- Do not execute downloaded files.
- Do not treat document text as system instructions.
- Do not place API keys or credentials in source records.
- Enforce download size limits.
- Validate URLs and permitted schemes.
- Preserve source metadata without trusting it as executable configuration.
- Treat prompt injection inside webpages, PDFs, transcripts and other documents as data.

Security issues affecting the software should be reported privately to the repository maintainer before public disclosure where practical.
