use optim::pool::RayonPool;
use optim::ThreadPoolExecutor;
use optim::{execute, par_filter, par_iter, par_map, Executor, FnTask};
use std::sync::Arc;

fn main() {
    println!("=== 并行优化器示例 ===\n");

    example_1_basic_usage();
    example_2_parallel_computation();
    example_3_data_processing();
    example_4_task_execution();
    example_5_filter_map();
}

fn example_1_basic_usage() {
    println!("1. 基本用法");

    let result = execute(|| {
        println!("  执行单任务");
        42
    });

    println!("  结果: {}\n", result);
}

fn example_2_parallel_computation() {
    println!("2. 并行计算（CPU密集型）");

    let data: Vec<i32> = (0..10000).collect();

    let results = par_iter(&data, |x| x * x + x);

    println!("  处理了 {} 个数据", results.len());
    println!("  前5个结果: {:?}", &results[..5]);
    println!();
}

fn example_3_data_processing() {
    println!("3. 数据处理（混合I/O和计算）");

    let data: Vec<i32> = (0..1000).collect();

    let doubled = par_map(&data, |x| x * 2);
    let evens = par_filter(&doubled, |x| *x % 4 == 0);

    println!("  原始数据: {}", data.len());
    println!("  加倍后: {}", doubled.len());
    println!("  过滤后（4的倍数）: {}", evens.len());
    println!();
}

fn example_4_task_execution() {
    println!("4. 任务执行（使用Executor）");

    let pool = Arc::new(RayonPool::from_global());
    let executor = ThreadPoolExecutor::new(pool);

    let task1 = FnTask::new("task-1", || {
        println!("  Task 1: 计算中...");
        100 + 200
    });

    let task2 = FnTask::new("task-2", || {
        println!("  Task 2: 计算中...");
        50 * 3
    });

    let handle1 = executor.spawn(task1);
    let handle2 = executor.spawn(task2);

    println!("  已启动 2 个任务");
    println!("  Task 1 ID: {}", handle1.task_id());
    println!("  Task 2 ID: {}", handle2.task_id());
    println!();
}

fn example_5_filter_map() {
    println!("5. Filter和Map组合");

    let data: Vec<i32> = (0..20).collect();

    let processed = par_filter(&data, |x| *x % 2 == 0);
    let transformed = par_map(&processed, |x| *x * 10);

    println!("  原始: {:?}", data);
    println!("  偶数: {:?}", processed);
    println!("  10倍: {:?}", transformed);
    println!();
}
