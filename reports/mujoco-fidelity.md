# Robot Sports Gym fidelity report — mujoco

- Suite: `multisport-fidelity-v1`
- Coverage: **first-rebound contact dynamics only**
- Overall score: **98.08/100**
- Official-metric score: **98.98/100**
- Passed: **5/5**

| Metric | Measured | Reference interval | Error | Tolerance | Score | Result | Basis |
|---|---:|---:|---:|---:|---:|:---:|---|
| tennis.first_rebound_height | 1.3999 m | 1.3500–1.4700 m | 0.0101 m | 0.168× | 98.07 | PASS | [official](https://m.itftennis.com/media/13753/2025-technical-booklet.pdf) |
| table_tennis.first_rebound_height | 0.2432 m | 0.2300–0.2600 m | 0.0018 m | 0.123× | 98.96 | PASS | [official](https://documents.ittf.sport/sites/default/files/public/2024-03/2024_ITTF_Council_documents_EN.pdf) |
| football.first_rebound_height | 1.0092 m | 0.8500–1.1500 m | 0.0092 m | 0.061× | 99.74 | PASS | engineering |
| badminton.first_rebound_height | 0.0183 m | 0.0000–0.0600 m | 0.0183 m | 0.305× | 93.74 | PASS | engineering |
| basketball.first_rebound_height | 1.0591 m | 1.0350–1.0850 m | 0.0009 m | 0.036× | 99.91 | PASS | [official](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-official-basketball-rules-and-basketball-equipment.pdf) |

Scoring: `100 × exp(-ln(2) × tolerance_ratio²)`; an acceptance boundary scores 50.
Official and engineering reference metrics are reported separately.
