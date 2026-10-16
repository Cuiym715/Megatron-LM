# DSPP microbatch construction and ordering

- Mode: `warmup-short-steady-long`
- Cost source: `metadata-proxy`
- Physical microbatches: 10
- Warmup boundary: first 7 tasks
- Warmup short-only tasks: 4
- Real schedule group sizes: 1, 1, 1, 1, 3, 3
- Group sizes including communication-only padding: 1, 1, 1, 1, 3, 3

## Model layout

- Stage 0: chunk0 transformer_layers=[1] + embedding; chunk1 transformer_layers=[6] + output-head/loss
- Stage 1: chunk0 transformer_layers=[2]; chunk1 transformer_layers=[5]
- Stage 2: chunk0 transformer_layers=[3]; chunk1 transformer_layers=[4]

## Constructed physical microbatches (before ordering)

| Input position | TaskId | Valid/padding tokens | Composition |
| ---: | --- | ---: | --- |
| 0 | L0:P0 | 2048/0 | L0/S1 long seg 1/3 q[0:2048] len=6144 |
| 1 | L0:P1 | 2048/0 | L0/S1 long seg 2/3 q[2048:4096] len=6144 |
| 2 | L0:P2 | 2048/0 | L0/S1 long seg 3/3 q[4096:6144] len=6144 |
| 3 | L0:P3 | 1794/254 | L0/S0 short seg 1/1 q[0:1537] len=1537<br>L0/S3 short seg 1/1 q[0:257] len=257 |
| 4 | L0:P4 | 1537/511 | L0/S2 short seg 1/1 q[0:1537] len=1537 |
| 5 | L1:P0 | 2048/0 | L1/S2 long seg 1/3 q[0:2048] len=6144 |
| 6 | L1:P1 | 2048/0 | L1/S2 long seg 2/3 q[2048:4096] len=6144 |
| 7 | L1:P2 | 2048/0 | L1/S2 long seg 3/3 q[4096:6144] len=6144 |
| 8 | L1:P3 | 1537/511 | L1/S3 short seg 1/1 q[0:1537] len=1537 |
| 9 | L1:P4 | 514/1534 | L1/S0 short seg 1/1 q[0:257] len=257<br>L1/S1 short seg 1/1 q[0:257] len=257 |

## Final pipeline entrance order

| Order | Phase | Schedule group | TaskId | Workload | Composition |
| ---: | --- | ---: | --- | ---: | --- |
| 0 | warmup | 0 | L1:P4 | 132098.0000 | L1/S0 short seg 1/1 q[0:257] len=257<br>L1/S1 short seg 1/1 q[0:257] len=257 |
| 1 | warmup | 1 | L0:P4 | 2362369.0000 | L0/S2 short seg 1/1 q[0:1537] len=1537 |
| 2 | warmup | 2 | L1:P3 | 2362369.0000 | L1/S3 short seg 1/1 q[0:1537] len=1537 |
| 3 | warmup | 3 | L0:P3 | 2428418.0000 | L0/S0 short seg 1/1 q[0:1537] len=1537<br>L0/S3 short seg 1/1 q[0:257] len=257 |
| 4 | warmup | 4 | L0:P0 | 4194304.0000 | L0/S1 long seg 1/3 q[0:2048] len=6144 |
| 5 | warmup | 4 | L0:P1 | 12582912.0000 | L0/S1 long seg 2/3 q[2048:4096] len=6144 |
| 6 | warmup | 4 | L0:P2 | 20971520.0000 | L0/S1 long seg 3/3 q[4096:6144] len=6144 |
| 7 | steady | 5 | L1:P0 | 4194304.0000 | L1/S2 long seg 1/3 q[0:2048] len=6144 |
| 8 | steady | 5 | L1:P1 | 12582912.0000 | L1/S2 long seg 2/3 q[2048:4096] len=6144 |
| 9 | steady | 5 | L1:P2 | 20971520.0000 | L1/S2 long seg 3/3 q[4096:6144] len=6144 |

Workload unit: metadata proxy units. It is used only for ordering; it is not simulated runtime.
Long-sequence segments are consecutive in the entrance F order. The warmup boundary may fall inside a chain; it neither inserts a fence nor prevents B/W or unrelated work from interleaving.
