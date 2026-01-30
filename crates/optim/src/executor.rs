use crate::pool::ThreadPool;
use crate::task::{Task, TaskResult};
use std::sync::Arc;

/// 执行器句柄
#[derive(Debug)]
pub struct ExecutorHandle {
    task_id: String,
}

impl ExecutorHandle {
    pub fn task_id(&self) -> &str {
        &self.task_id
    }
}

/// 执行器trait
pub trait Executor: Send + Sync {
    fn spawn<T>(&self, task: T) -> ExecutorHandle
    where
        T: Task + 'static,
        T::Output: Send;

    fn spawn_blocking<T>(&self, task: T) -> ExecutorHandle
    where
        T: Task + 'static,
        T::Output: Send;

    fn spawn_many<T>(&self, tasks: Vec<T>) -> Vec<ExecutorHandle>
    where
        T: Task + 'static,
        T::Output: Send;

    fn wait_all<T>(&self, handles: Vec<ExecutorHandle>) -> Vec<TaskResult<T::Output>>
    where
        T: Task + 'static,
        T::Output: Send;
}

/// 基于线程池的执行器
pub struct ThreadPoolExecutor<P>
where
    P: ThreadPool + 'static,
{
    pool: Arc<P>,
}

impl<P> ThreadPoolExecutor<P>
where
    P: ThreadPool + 'static,
{
    pub fn new(pool: Arc<P>) -> Self {
        Self { pool }
    }

    pub fn from_pool(pool: Arc<P>) -> Self {
        Self { pool }
    }
}

impl<P> Executor for ThreadPoolExecutor<P>
where
    P: ThreadPool + 'static,
{
    fn spawn<T>(&self, task: T) -> ExecutorHandle
    where
        T: Task + 'static,
        T::Output: Send,
    {
        let task_id = task.name();
        let _ = self.pool.execute(move || task.execute());

        ExecutorHandle { task_id }
    }

    fn spawn_blocking<T>(&self, task: T) -> ExecutorHandle
    where
        T: Task + 'static,
        T::Output: Send,
    {
        let task_id = task.name();
        let _ = self.pool.execute(move || task.execute());

        ExecutorHandle { task_id }
    }

    fn spawn_many<T>(&self, tasks: Vec<T>) -> Vec<ExecutorHandle>
    where
        T: Task + 'static,
        T::Output: Send,
    {
        tasks.into_iter().map(|task| self.spawn(task)).collect()
    }

    fn wait_all<T>(&self, _handles: Vec<ExecutorHandle>) -> Vec<TaskResult<T::Output>>
    where
        T: Task + 'static,
        T::Output: Send,
    {
        unimplemented!("wait_all requires task storage and result retrieval")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_spawn() {
        use crate::pool::RayonPool;

        let pool = RayonPool::from_global();
        let executor = ThreadPoolExecutor::new(Arc::new(pool));

        let task = crate::task::FnTask::new("test-task", || 42);
        let handle = executor.spawn(task);
        assert_eq!(handle.task_id(), "test-task");
    }
}
