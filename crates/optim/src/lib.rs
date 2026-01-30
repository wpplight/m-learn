use rayon::prelude::*;

pub mod executor;
pub mod pool;
pub mod task;

pub use executor::{Executor, ExecutorHandle, ThreadPoolExecutor};
pub use pool::{GlobalThreadPool, PoolConfig, PoolInfo, RayonPool, ThreadPool};
pub use task::{FnTask, Task, TaskResult, TaskStatus};

/// 并行map
pub fn par_map<T, U, F>(iter: T, f: F) -> Vec<U>
where
    T: IntoParallelIterator,
    F: Fn(T::Item) -> U + Sync + Send,
    U: Send,
{
    GlobalThreadPool::par_map(iter, f)
}

/// 并行filter
pub fn par_filter<T, P>(iter: T, predicate: P) -> Vec<T::Item>
where
    T: IntoParallelIterator,
    T::Item: Send,
    P: Fn(&T::Item) -> bool + Sync + Send,
{
    GlobalThreadPool::par_filter(iter, predicate)
}

/// 并行执行单个任务
pub fn execute<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    GlobalThreadPool::execute(f)
}

/// 并行处理迭代器
pub fn par_iter<T, F, R>(iter: T, f: F) -> Vec<R>
where
    T: IntoParallelIterator,
    F: Fn(T::Item) -> R + Sync + Send,
    R: Send,
{
    iter.into_par_iter().map(f).collect()
}

/// 批量并行处理
pub fn par_batches<F, R>(items: Vec<F>, mapper: impl Fn(F) -> R + Sync + Send) -> Vec<R>
where
    F: Send + 'static,
    R: Send,
{
    items.into_par_iter().map(mapper).collect()
}

/// 获取线程池信息
pub fn pool_info() -> PoolInfo {
    GlobalThreadPool::pool_info()
}
