import numpy as np
connections = {1: [2, 3], 2: [3, 4], 3: [1, 5], 4: [1, 3, 5],
         5: [2, 6, 7], 6: [6], 7: [8], 8: [7]}

p = np.zeros((8, 8))
for i, outs in connections.items():
    for j in outs:
        p[i-1, j-1] = 1 / len(outs)      #Uniform over outward connections

assert np.allclose(p.sum(axis=1), 1)

q = np.ones(8) / 8
for n in range(1, 102):
    q = q @ p
    if n in (100, 101):
        print(f"q{n} =", np.round(q, 6))