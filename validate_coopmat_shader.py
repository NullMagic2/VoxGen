from pathlib import Path

root = Path(__file__).resolve().parent
shader = (root / "shaders" / "seq_linear_bias_coopmat_xtx7900.comp").read_text()
root_cargo = (root / "Cargo.toml").read_text()
demo_cargo = (root / "demo" / "Cargo.toml").read_text()

def need(ok, msg):
    if not ok:
        raise SystemExit(f"FAIL: {msg}")

need('version = "0.7.37"' in root_cargo, 'root version')
need('version = "0.7.37"' in demo_cargo, 'demo version')
mem = '#extension GL_KHR_memory_scope_semantics : require'
coop = '#extension GL_KHR_cooperative_matrix : require'
need(mem in shader, 'GL_KHR_memory_scope_semantics extension missing')
need(coop in shader, 'GL_KHR_cooperative_matrix extension missing')
need(shader.index(mem) < shader.index('gl_ScopeSubgroup'), 'memory scope extension must be requested before gl_ScopeSubgroup use')
need(shader.count('gl_ScopeSubgroup') >= 3, 'cooperative matrix subgroup scope unexpectedly absent')
print('OK: cooperative-matrix GLSL explicitly requests memory-scope semantics before subgroup scope constants')
