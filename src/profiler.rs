use serde::Serialize;
use std::{collections::BTreeMap, sync::Mutex, time::Instant};

#[derive(Default)]
pub struct Profiler { stages: Mutex<BTreeMap<String, f64>> }

pub struct Scope<'a> { profiler: &'a Profiler, name: &'static str, start: Instant }
impl Profiler {
    pub fn scope(&self, name: &'static str) -> Scope<'_> { Scope { profiler: self, name, start: Instant::now() } }
    pub fn snapshot(&self) -> ProfileSnapshot { ProfileSnapshot { stages_ms: self.stages.lock().unwrap().clone() } }
}
impl Drop for Scope<'_> {
    fn drop(&mut self) { self.profiler.stages.lock().unwrap().insert(self.name.into(), self.start.elapsed().as_secs_f64() * 1000.0); }
}
#[derive(Debug, Clone, Serialize)]
pub struct ProfileSnapshot { pub stages_ms: BTreeMap<String, f64> }
