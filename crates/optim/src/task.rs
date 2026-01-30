use std::fmt;

/// 任务执行结果
pub type TaskResult<T> = Result<T, TaskError>;

/// 任务执行错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskError {
    Panic(String),
    Timeout,
    Cancelled,
}

impl std::error::Error for TaskError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl fmt::Display for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskError::Panic(msg) => write!(f, "Task panicked: {}", msg),
            TaskError::Timeout => write!(f, "Task timed out"),
            TaskError::Cancelled => write!(f, "Task was cancelled"),
        }
    }
}

/// 任务状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed(TaskError),
}

/// 通用任务抽象
pub trait Task: Send + 'static {
    type Output: Send;

    fn name(&self) -> String;
    fn execute(self) -> Self::Output;
}

/// 函数任务包装器
pub struct FnTask<F, O>
where
    F: FnOnce() -> O + Send + 'static,
    O: Send + 'static,
{
    name: String,
    func: F,
    _phantom: std::marker::PhantomData<O>,
}

impl<F, O> FnTask<F, O>
where
    F: FnOnce() -> O + Send + 'static,
    O: Send + 'static,
{
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            func,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F, O> Task for FnTask<F, O>
where
    F: FnOnce() -> O + Send + 'static,
    O: Send + 'static,
{
    type Output = O;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn execute(self) -> Self::Output {
        (self.func)()
    }
}

/// 闭包任务包装器（支持捕获）
pub struct ClosureTask<O>
where
    O: Send + 'static,
{
    name: String,
    func: Box<dyn FnOnce() -> O + Send + 'static>,
}

impl<O> ClosureTask<O>
where
    O: Send + 'static,
{
    pub fn new(name: impl Into<String>, func: Box<dyn FnOnce() -> O + Send + 'static>) -> Self {
        Self {
            name: name.into(),
            func,
        }
    }
}

impl<O> Task for ClosureTask<O>
where
    O: Send + 'static,
{
    type Output = O;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn execute(self) -> Self::Output {
        (self.func)()
    }
}
