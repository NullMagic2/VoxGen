import math
import struct
from pathlib import Path

N = 2048
base = [0.35 * math.sin(i * 0.017) + 0.08 * math.cos(i * 0.071) for i in range(N)]
current = [0.22 * math.cos(i * 0.013) - 0.11 * math.sin(i * 0.043) for i in range(N)]
for name, values in [("test_base_hidden.f32", base), ("test_current_embed.f32", current)]:
    Path(name).write_bytes(b"".join(struct.pack("<f", x) for x in values))
    print(f"wrote {name}: {len(values)} f32 / {len(values)*4} bytes")
