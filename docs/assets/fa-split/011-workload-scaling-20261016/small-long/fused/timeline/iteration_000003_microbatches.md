# DSPP microbatch construction and ordering

- Mode: `warmup-short-steady-long`
- Cost source: `metadata-proxy`
- Physical microbatches: 18
- Warmup boundary: first 11 tasks
- Warmup short-only tasks: 4
- Real schedule group sizes: 1, 1, 1, 1, 7, 7
- Group sizes including communication-only padding: 1, 1, 1, 1, 7, 7

## Model layout

- Stage 0: chunk0 transformer_layers=[1] + embedding; chunk1 transformer_layers=[6] + output-head/loss
- Stage 1: chunk0 transformer_layers=[2]; chunk1 transformer_layers=[5]
- Stage 2: chunk0 transformer_layers=[3]; chunk1 transformer_layers=[4]

## Constructed physical microbatches (before ordering)

| Input position | TaskId | Valid/padding tokens | Composition |
| ---: | --- | ---: | --- |
| 0 | L0:P0 | 2048/0 | L0/S1 long seg 1/7 q[0:2048] len=14336 |
| 1 | L0:P1 | 2048/0 | L0/S1 long seg 2/7 q[2048:4096] len=14336 |
| 2 | L0:P2 | 2048/0 | L0/S1 long seg 3/7 q[4096:6144] len=14336 |
| 3 | L0:P3 | 2048/0 | L0/S1 long seg 4/7 q[6144:8192] len=14336 |
| 4 | L0:P4 | 2048/0 | L0/S1 long seg 5/7 q[8192:10240] len=14336 |
| 5 | L0:P5 | 2048/0 | L0/S1 long seg 6/7 q[10240:12288] len=14336 |
| 6 | L0:P6 | 2048/0 | L0/S1 long seg 7/7 q[12288:14336] len=14336 |
| 7 | L0:P7 | 1794/254 | L0/S0 short seg 1/1 q[0:1537] len=1537<br>L0/S3 short seg 1/1 q[0:257] len=257 |
| 8 | L0:P8 | 1537/511 | L0/S2 short seg 1/1 q[0:1537] len=1537 |
| 9 | L1:P0 | 2048/0 | L1/S2 long seg 1/7 q[0:2048] len=14336 |
| 10 | L1:P1 | 2048/0 | L1/S2 long seg 2/7 q[2048:4096] len=14336 |
| 11 | L1:P2 | 2048/0 | L1/S2 long seg 3/7 q[4096:6144] len=14336 |
| 12 | L1:P3 | 2048/0 | L1/S2 long seg 4/7 q[6144:8192] len=14336 |
| 13 | L1:P4 | 2048/0 | L1/S2 long seg 5/7 q[8192:10240] len=14336 |
| 14 | L1:P5 | 2048/0 | L1/S2 long seg 6/7 q[10240:12288] len=14336 |
| 15 | L1:P6 | 2048/0 | L1/S2 long seg 7/7 q[12288:14336] len=14336 |
| 16 | L1:P7 | 1537/511 | L1/S3 short seg 1/1 q[0:1537] len=1537 |
| 17 | L1:P8 | 514/1534 | L1/S0 short seg 1/1 q[0:257] len=257<br>L1/S1 short seg 1/1 q[0:257] len=257 |

## Final pipeline entrance order

| Order | Phase | Schedule group | TaskId | Workload | Composition |
| ---: | --- | ---: | --- | ---: | --- |
| 0 | warmup | 0 | L1:P8 | 132098.0000 | L1/S0 short seg 1/1 q[0:257] len=257<br>L1/S1 short seg 1/1 q[0:257] len=257 |
| 1 | warmup | 1 | L0:P8 | 2362369.0000 | L0/S2 short seg 1/1 q[0:1537] len=1537 |
| 2 | warmup | 2 | L1:P7 | 2362369.0000 | L1/S3 short seg 1/1 q[0:1537] len=1537 |
| 3 | warmup | 3 | L0:P7 | 2428418.0000 | L0/S0 short seg 1/1 q[0:1537] len=1537<br>L0/S3 short seg 1/1 q[0:257] len=257 |
| 4 | warmup | 4 | L0:P0 | 4194304.0000 | L0/S1 long seg 1/7 q[0:2048] len=14336 |
| 5 | warmup | 4 | L0:P1 | 12582912.0000 | L0/S1 long seg 2/7 q[2048:4096] len=14336 |
| 6 | warmup | 4 | L0:P2 | 20971520.0000 | L0/S1 long seg 3/7 q[4096:6144] len=14336 |
| 7 | warmup | 4 | L0:P3 | 29360128.0000 | L0/S1 long seg 4/7 q[6144:8192] len=14336 |
| 8 | warmup | 4 | L0:P4 | 37748736.0000 | L0/S1 long seg 5/7 q[8192:10240] len=14336 |
| 9 | warmup | 4 | L0:P5 | 46137344.0000 | L0/S1 long seg 6/7 q[10240:12288] len=14336 |
| 10 | warmup | 4 | L0:P6 | 54525952.0000 | L0/S1 long seg 7/7 q[12288:14336] len=14336 |
| 11 | steady | 5 | L1:P0 | 4194304.0000 | L1/S2 long seg 1/7 q[0:2048] len=14336 |
| 12 | steady | 5 | L1:P1 | 12582912.0000 | L1/S2 long seg 2/7 q[2048:4096] len=14336 |
| 13 | steady | 5 | L1:P2 | 20971520.0000 | L1/S2 long seg 3/7 q[4096:6144] len=14336 |
| 14 | steady | 5 | L1:P3 | 29360128.0000 | L1/S2 long seg 4/7 q[6144:8192] len=14336 |
| 15 | steady | 5 | L1:P4 | 37748736.0000 | L1/S2 long seg 5/7 q[8192:10240] len=14336 |
| 16 | steady | 5 | L1:P5 | 46137344.0000 | L1/S2 long seg 6/7 q[10240:12288] len=14336 |
| 17 | steady | 5 | L1:P6 | 54525952.0000 | L1/S2 long seg 7/7 q[12288:14336] len=14336 |

Workload unit: metadata proxy units. It is used only for ordering; it is not simulated runtime.
Long-sequence segments are consecutive in the entrance F order. The warmup boundary may fall inside a chain; it neither inserts a fence nor prevents B/W or unrelated work from interleaving.
