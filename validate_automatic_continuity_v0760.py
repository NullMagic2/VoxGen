from pathlib import Path

root = Path(__file__).resolve().parent
http = (root/'src/http.rs').read_text(encoding='utf-8')
demo = (root/'demo/src/main.rs').read_text(encoding='utf-8')

def need(c, m):
    if not c:
        raise SystemExit('FAIL: ' + m)

for token in [
    'const CONTINUITY_TTL: Duration = Duration::from_secs(30 * 60);',
    'const MAX_CONTINUITY_SESSIONS: usize = 256;',
    'global_generation: u64', 'id_generations: HashMap<String, u64>',
    'store.global_generation = store.global_generation.wrapping_add(1);',
    'let generation = store.id_generations.entry(continuity_id.to_owned()).or_insert(0);',
    'prior.speaker_key == speaker_key',
    'store.global_generation != plan.global_generation',
    'store.sessions.len() >= MAX_CONTINUITY_SESSIONS',
    'min_by_key(|(_, session)| session.updated_at)',
    'fn commit_continuity(',
    'state.clear_continuity()?',
    'state.reset_continuity_id(continuity_id)?',
]:
    need(token in http, 'continuity state safety: ' + token)

# Both successful paths commit only after synthesis/cancellation checks.
need(http.count('commit_continuity(&state, plan)?') >= 2, 'streaming and buffered success commits')
stream_cancel = http.index('// A cancelled stream is a normal control-flow event')
stream_commit = http.index('commit_continuity(&state, plan)?', stream_cancel)
need(stream_cancel < stream_commit, 'stream commit occurs after cancel gate')
nonstream_cancel = http.index('Err(_err) if state.cancel_speech.load(Ordering::Acquire)')
nonstream_commit = http.index('commit_continuity(&state, plan)?', stream_commit + 1)
need(nonstream_cancel < nonstream_commit, 'buffered commit occurs after cancel gate')

# Response diagnostics expose what the engine actually used.
for header in [
    'X-VoxGen-Previous-Style', 'X-VoxGen-Effective-Style',
    'X-VoxGen-Previous-Intensity', 'X-VoxGen-Effective-Intensity',
    'X-VoxGen-Previous-Pace-Percent', 'X-VoxGen-Effective-Pace-Percent',
    'X-VoxGen-Requested-Pace-Percent', 'X-VoxGen-Boundary',
]:
    need(header in http, 'continuity response header: ' + header)

# Demo preserves a stable speaker-conditioning anchor during managed style changes.
need('let reference_preset = if clone_mode != "ultimate" && is_managed_style_preset(&preset)' in demo, 'managed reference selection branch')
need('"neutral"' in demo[demo.index('let reference_preset = if clone_mode'):demo.index('let reference_preset = if clone_mode')+400], 'neutral managed reference anchor')
print('v0.7.60 automatic continuity state-safety validation passed')
