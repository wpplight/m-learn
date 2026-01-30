use once_cell::sync::Lazy;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

/// 线程池配置
#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub num_threads: Option<usize>,
    pub thread_name: Option<String>,
    pub stack_size: Option<usize>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            num_threads: None,
            thread_name: None,
            stack_size: None,
        }
    }
}

impl PoolConfig {
    pub fn with_num_threads(mut self, n: usize) -> Self {
        self.num_threads = Some(n);
        self
    }

    pub fn with_thread_name(mut self, name: impl Into<String>) -> Self {
        self.thread_name = Some(name.into());
        self
    }

    pub fn with_stack_size(mut self, size: usize) -> Self {
        self.stack_size = Some(size);
        self
    }
}

/// 线程池信息
#[derive(Debug, Clone)]
pub struct PoolInfo {
    pub num_threads: usize,
    pub current_threads: usize,
    pub idle_threads: usize,
}

/// 线程池trait
pub trait ThreadPool: Send + Sync {
    fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static;

    fn pool_info(&self) -> PoolInfo;
}

/// Rayon线程池实现
pub struct RayonPool {
    inner: rayon::ThreadPool,
}

impl RayonPool {
    pub fn new(config: PoolConfig) -> Result<Self, rayon::ThreadPoolBuildError> {
        let mut builder = ThreadPoolBuilder::new();

        if let Some(num_threads) = config.num_threads {
            builder = builder.num_threads(num_threads);
        }

        if let Some(thread_name) = config.thread_name {
            builder = builder.thread_name(move |i| format!("{}-{}", thread_name, i));
        }

        if let Some(stack_size) = config.stack_size {
            builder = builder.stack_size(stack_size);
        }

        Ok(Self {
            inner: builder.build()?,
        })
    }

    pub fn from_global() -> Self {
        Self {
            inner: rayon::ThreadPoolBuilder::new().build().unwrap(),
        }
    }
}

impl ThreadPool for RayonPool {
    fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.inner.install(f)
    }

    fn pool_info(&self) -> PoolInfo {
        PoolInfo {
            num_threads: self.inner.current_num_threads(),
            current_threads: self.inner.current_num_threads(),
            idle_threads: self.inner.current_num_threads(),
        }
    }
}

/// 全局线程池（单例模式）
pub struct GlobalThreadPool;

static GLOBAL_POOL: Lazy<RayonPool> = Lazy::new(|| RayonPool::from_global());

impl GlobalThreadPool {
    /// 执行单个任务
    pub fn execute<F, R>(f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        GLOBAL_POOL.inner.install(f)
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
        GLOBAL_POOL.pool_info()
    }

    /// 并行map
    pub fn par_map<T, U, F>(iter: T, f: F) -> Vec<U>
    where
        T: IntoParallelIterator,
        F: Fn(T::Item) -> U + Sync + Send,
        U: Send,
    {
        iter.into_par_iter().map(f).collect()
    }

    /// 并行filter
    pub fn par_filter<T, P>(iter: T, predicate: P) -> Vec<T::Item>
    where
        T: IntoParallelIterator,
        T::Item: Send,
        P: Fn(&T::Item) -> bool + Sync + Send,
    {
        iter.into_par_iter().filter(predicate).collect()
    }

    /// 并行for_each
    pub fn par_for_each<T, F>(iter: T, f: F)
    where
        T: IntoParallelIterator,
        F: Fn(T::Item) + Sync + Send,
    {
        iter.into_par_iter().for_each(f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rayon_pool() {
        let pool = RayonPool::new(PoolConfig::default()).unwrap();
        let result = pool.execute(|| 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_global_pool() {
        let result = GlobalThreadPool::execute(|| 10);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_par_iter() {
        let data: Vec<i32> = (0..100).collect();
        let squared = GlobalThreadPool::par_iter(&data, |x| x * x);
        assert_eq!(squared.len(), 100);
        assert_eq!(squared[50], 2500);
    }

    #[test]
    fn test_par_filter() {
        let data: Vec<i32> = (0..20).collect();
        let evens = GlobalThreadPool::par_filter(&data, |&x| x % 2 == 0);
        assert_eq!(evens.len(), 10);
    }
}
