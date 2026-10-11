use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Capture,
    Decode,
    DocumentDetection,
    PerspectiveCorrection,
    Preprocessing,
    AiInference,
    StructuredParsing,
    DomainValidation,
    Persistence,
    Total,
}

impl Stage {
    pub const ALL: [Self; 10] = [
        Self::Capture,
        Self::Decode,
        Self::DocumentDetection,
        Self::PerspectiveCorrection,
        Self::Preprocessing,
        Self::AiInference,
        Self::StructuredParsing,
        Self::DomainValidation,
        Self::Persistence,
        Self::Total,
    ];

    const fn index(self) -> usize {
        match self {
            Self::Capture => 0,
            Self::Decode => 1,
            Self::DocumentDetection => 2,
            Self::PerspectiveCorrection => 3,
            Self::Preprocessing => 4,
            Self::AiInference => 5,
            Self::StructuredParsing => 6,
            Self::DomainValidation => 7,
            Self::Persistence => 8,
            Self::Total => 9,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageMetrics {
    pub elapsed_nanos: u64,
    pub samples: u64,
}

struct Metrics {
    elapsed_nanos: [AtomicU64; 10],
    samples: [AtomicU64; 10],
}

impl Metrics {
    fn new() -> Self {
        Self {
            elapsed_nanos: std::array::from_fn(|_| AtomicU64::new(0)),
            samples: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

pub struct InstrumentationGuard {
    state: GuardState,
}

enum GuardState {
    Noop,
    Developer {
        stage: Stage,
        started: Instant,
        metrics: Arc<Metrics>,
    },
}

impl InstrumentationGuard {
    fn noop() -> Self {
        Self {
            state: GuardState::Noop,
        }
    }
}

impl Drop for InstrumentationGuard {
    fn drop(&mut self) {
        if let GuardState::Developer {
            stage,
            started,
            metrics,
        } = &self.state
        {
            let elapsed = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            metrics.elapsed_nanos[stage.index()].fetch_add(elapsed, Ordering::Relaxed);
            metrics.samples[stage.index()].fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub trait Instrumentation: Send + Sync {
    fn begin(&self, stage: Stage) -> InstrumentationGuard;
    fn snapshot(&self) -> [StageMetrics; 10];
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoopInstrumentation;

impl Instrumentation for NoopInstrumentation {
    fn begin(&self, _stage: Stage) -> InstrumentationGuard {
        InstrumentationGuard::noop()
    }

    fn snapshot(&self) -> [StageMetrics; 10] {
        [StageMetrics {
            elapsed_nanos: 0,
            samples: 0,
        }; 10]
    }
}

#[derive(Clone)]
pub struct DeveloperInstrumentation {
    metrics: Arc<Metrics>,
}

impl Default for DeveloperInstrumentation {
    fn default() -> Self {
        Self {
            metrics: Arc::new(Metrics::new()),
        }
    }
}

impl Instrumentation for DeveloperInstrumentation {
    fn begin(&self, stage: Stage) -> InstrumentationGuard {
        InstrumentationGuard {
            state: GuardState::Developer {
                stage,
                started: Instant::now(),
                metrics: Arc::clone(&self.metrics),
            },
        }
    }

    fn snapshot(&self) -> [StageMetrics; 10] {
        std::array::from_fn(|index| StageMetrics {
            elapsed_nanos: self.metrics.elapsed_nanos[index].load(Ordering::Relaxed),
            samples: self.metrics.samples[index].load(Ordering::Relaxed),
        })
    }
}
