# Robot Sports Gym fidelity report — isaacsim

- Suite: `multisport-fidelity-v1`
- Coverage: **first-rebound contact dynamics only**
- Overall score: **98.67/100**
- Official-metric score: **99.93/100**
- Passed: **5/5**

| Metric | Measured | Reference interval | Error | Tolerance | Score | Result | Basis |
|---|---:|---:|---:|---:|---:|:---:|---|
| badminton.first_rebound_height | 0.0184 m | 0.0000–0.0600 m | 0.0184 m | 0.307× | 93.66 | PASS | engineering |
| basketball.first_rebound_height | 1.0593 m | 1.0350–1.0850 m | 0.0007 m | 0.027× | 99.95 | PASS | [official](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-official-basketball-rules-and-basketball-equipment.pdf) |
| football.first_rebound_height | 0.9941 m | 0.8500–1.1500 m | 0.0059 m | 0.039× | 99.89 | PASS | engineering |
| table_tennis.first_rebound_height | 0.2444 m | 0.2300–0.2600 m | 0.0006 m | 0.038× | 99.90 | PASS | [official](https://documents.ittf.sport/sites/default/files/public/2024-03/2024_ITTF_Council_documents_EN.pdf) |
| tennis.first_rebound_height | 1.4082 m | 1.3500–1.4700 m | 0.0018 m | 0.030× | 99.94 | PASS | [official](https://m.itftennis.com/media/13753/2025-technical-booklet.pdf) |

Scoring: `100 × exp(-ln(2) × tolerance_ratio²)`; an acceptance boundary scores 50.
Official and engineering reference metrics are reported separately.
