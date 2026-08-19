"""PRNG device (XORShift 5-int) — 1:1 z TspGAKernel.random() (int32 wrapping)."""
from numba import cuda, int32


@cuda.jit(device=True, inline=True)
def rnd_int(states, gid):
    s0 = states[gid, 0]
    t = int32(s0 ^ (s0 >> 7))
    states[gid, 0] = states[gid, 1]
    states[gid, 1] = states[gid, 2]
    states[gid, 2] = states[gid, 3]
    states[gid, 3] = states[gid, 4]
    s4 = states[gid, 4]
    states[gid, 4] = int32((s4 ^ (s4 << 6)) ^ (t ^ (t << 13)))
    return int32((states[gid, 1] + states[gid, 1] + 1) * states[gid, 4])


@cuda.jit(device=True, inline=True)
def rnd01(states, gid):
    v = rnd_int(states, gid)
    if v < 0:
        return -v / 2147483648.0
    elif v > 0:
        return v / 2147483647.0
    return 0.0
