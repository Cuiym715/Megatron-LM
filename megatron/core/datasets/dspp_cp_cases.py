"""Small, explicit CP acceptance inputs; not a cached training schedule."""
CP_LAYOUT_CASES = (
    ('whole-short', (800, 224, 600, 424), 2048),
    ('oversize-short', (1500, 548), 2048),
    ('fragmented-short', (700, 700, 648), 2048),
    ('history-tail-short', (4608, 768, 768), 2048),
    ('multi-history', (6401, 2305, 97, 31), 2048),
    ('empty-local-query', (1,), 64),
)
