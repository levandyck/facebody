import os
from dataclasses import dataclass
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

# ---------------------------- Parallel processing --------------------------- #
@dataclass(frozen=True)
class ParallelConfig:
    max_proc: int = 64
    batch_size: int = 8

def run_parallel(func, tasks, desc: str, cfg: ParallelConfig):
    """Joblib wrapper with a single tqdm bar, process backend, and single-thread inner workers."""
    if not tasks:
        return []

    n_workers = min(cfg.max_proc, len(tasks), os.cpu_count() or 1)
    with parallel_backend("loky", inner_max_num_threads=1):
        with tqdm_joblib(tqdm(total=len(tasks), desc=desc)):
            return Parallel(n_jobs=n_workers, batch_size=cfg.batch_size)(
                delayed(func)(*t) for t in tasks
            )
