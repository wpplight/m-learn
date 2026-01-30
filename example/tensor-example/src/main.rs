use formula;
use std::time::Instant;
use tensor::TensorError;
use tensor::{tensor, Tensor};

fn main() -> Result<(), TensorError> {
    let mut test_tensor = Tensor::new(vec![2, 3], 0.2)?;
    println!("{:?}", test_tensor.numel());
    test_tensor.display();

    test_tensor.reshape(vec![3, 2])?;
    test_tensor.display();

    // 测试均匀分布 [0, 1)
    println!("\n=== Test rand (uniform [0, 1)) ===");
    let rand_tensor = Tensor::rand(vec![2, 3])?;
    rand_tensor.display();

    // 测试标准正态分布 (mean=0, std=1)
    println!("\n=== Test randn (standard normal) ===");
    let randn_tensor = Tensor::randn(vec![2, 3])?;
    randn_tensor.display();

    println!("\n=== Test tensor! macro ===");
    // 测试直接构筑
    let t = tensor!([1.0, 2.0, 3.0])?;
    t.display();
    let t2 = Tensor::build(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2, 6],
    )?;
    t2.display();

    // 测试运算符
    println!("\n=== Test operators ===");
    let x = Tensor::build(vec![1.0, 2.0, 4.0, 8.0], vec![4])?;
    let y = Tensor::build(vec![2.0, 2.0, 2.0, 2.0], vec![4])?;

    print!("x: ");
    x.display();
    print!("y: ");
    y.display();
    println!();

    print!("x + y: ");
    (x.clone() + y.clone())?.display();
    print!("x - y: ");
    (x.clone() - y.clone())?.display();
    print!("x * y: ");
    (x.clone() * y.clone())?.display();
    print!("x / y: ");
    (x.clone() / y.clone())?.display();
    print!("x ** 2: ");
    x.pow(2.0).display();

    println!();
    print!("x + 2.0: ");
    (x.clone() + 2.0).display();
    print!("x - 1.0: ");
    (x.clone() - 1.0).display();
    print!("x * 3.0: ");
    (x.clone() * 3.0).display();
    print!("x / 2.0: ");
    (x / 2.0).display();

    println!();
    println!("=== Test indexing ===");
    let t2d = Tensor::build(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    println!("2D tensor [2,2]:");
    print!("  t2d[[0, 0]] = {}\n", t2d[[0, 0]]);
    print!("  t2d[[0, 1]] = {}\n", t2d[[0, 1]]);
    print!("  t2d[[1, 0]] = {}\n", t2d[[1, 0]]);
    print!("  t2d[[1, 1]] = {}\n", t2d[[1, 1]]);

    println!();
    let t3d = Tensor::build(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])?;
    println!("3D tensor [2, 2, 2]:");
    print!("  t3d[[0, 0, 0]] = {}\n", t3d[[0, 0, 0]]);
    print!("  t3d[[0, 0, 1]] = {}\n", t3d[[0, 0, 1]]);
    print!("  t3d[[1, 1, 1]] = {}\n", t3d[[1, 1, 1]]);

    println!();
    println!("=== Test slice ===");
    let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let t_slice = Tensor::build(data, vec![3, 4])?;
    print!("  t[1..3]: ");
    t_slice.slice(1..3)?.display();

    // 测试sum
    println!("======test sum ========");
    let z = tensor!([1.0, 2.0, 3.0])?;
    z.display();
    print!("z.sum(): ");
    println!("{}", z.sum());

    // 测试4D索引
    println!("\n=== Test 4D indexing ===");
    let data_4d: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let t4d = Tensor::build(data_4d, vec![2, 2, 2, 2])?;
    println!("4D tensor [2, 2, 2, 2]:");
    print!("  t4d[[0, 0, 0, 0]] = {}\n", t4d[[0, 0, 0, 0]]);
    print!("  t4d[[0, 0, 0, 1]] = {}\n", t4d[[0, 0, 0, 1]]);
    print!("  t4d[[0, 0, 1, 0]] = {}\n", t4d[[0, 0, 1, 0]]);
    print!("  t4d[[0, 0, 1, 1]] = {}\n", t4d[[0, 0, 1, 1]]);
    print!("  t4d[[1, 1, 1, 1]] = {}\n", t4d[[1, 1, 1, 1]]);

    // 测试可变索引
    println!("\n=== Test mutable indexing ===");
    let mut t_mut = Tensor::build(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    println!("Before mutation:");
    t_mut.display();
    t_mut[[0, 1]] = 10.0;
    t_mut[[1, 0]] = 20.0;
    println!("After mutation:");
    t_mut.display();
    println!("  t_mut[[0, 1]] = {}", t_mut[[0, 1]]);
    println!("  t_mut[[1, 0]] = {}", t_mut[[1, 0]]);

    // 测试1D切片
    println!("\n=== Test 1D slice ===");
    let t_1d = Tensor::build(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5])?;
    println!("Original 1D tensor:");
    t_1d.display();
    print!("  t[1..4]: ");
    t_1d.slice(1..4)?.display();

    // 测试错误处理
    println!("\n=== Test error handling ===");
    let t_err = Tensor::build(vec![1.0, 2.0, 3.0], vec![3])?;
    match t_err.slice(0..10) {
        Ok(_) => println!("Slice succeeded (unexpected!)"),
        Err(e) => println!("Slice error (expected): {:?}", e),
    }

    // 测试组合运算
    println!("\n=== Test chained operations ===");
    let a = Tensor::build(vec![1.0, 2.0, 3.0], vec![3])?;
    let b = Tensor::build(vec![4.0, 5.0, 6.0], vec![3])?;
    let c = Tensor::build(vec![7.0, 8.0, 9.0], vec![3])?;
    print!("a: ");
    a.display();
    print!("b: ");
    b.display();
    print!("c: ");
    c.display();
    let result = (a.clone() + b.clone())? * 2.0;
    print!("(a + b) * 2: ");
    result.display();
    let sum = result / 2.0;
    let power = c.pow(2.0);
    let chained = (sum / power)?;
    print!("(a + b) / (c ** 2): ");
    chained.display();

    // 测试向量化操作 vs 手动循环的性能对比
    println!("\n=== Test performance: manual loop vs tensor operations ===");

    let t1 = Tensor::arange(10000)?;
    let t2 = t1.clone();
    let mut t3_manual = Tensor::zeros(vec![10000])?;

    let start = Instant::now();
    for i in 0..10000 {
        t3_manual[i] = t1[i] + t2[i];
    }
    let duration_manual = start.elapsed();

    let start = Instant::now();
    let _t4_tensor_small = (t1.clone() + t2.clone())?;
    let duration_tensor_small = start.elapsed();

    println!("\nSmall data (10,000 elements):");
    println!("Manual loop: {:?}", duration_manual);
    println!("Tensor op:   {:?}", duration_tensor_small);
    if duration_manual < duration_tensor_small {
        let speedup = duration_tensor_small.as_nanos() as f64 / duration_manual.as_nanos() as f64;
        println!("Manual loop is {:.2}x faster", speedup);
    } else {
        let speedup = duration_manual.as_nanos() as f64 / duration_tensor_small.as_nanos() as f64;
        println!("Tensor operation is {:.2}x faster", speedup);
    }

    let t1_large = Tensor::arange(1_000_000)?;
    let t2_large = t1_large.clone();
    let mut t3_manual_large = Tensor::zeros(vec![1_000_000]).unwrap();

    let start = Instant::now();
    for i in 0..1_000_000 {
        t3_manual_large[i] = t1_large[i] + t2_large[i];
    }
    let duration_manual_large = start.elapsed();

    let start = Instant::now();
    let _t4_tensor_large = (t1_large.clone() + t2_large.clone())?;
    let duration_tensor_large = start.elapsed();

    println!("\nLarge data (1,000,000 elements):");
    println!("Manual loop: {:?}", duration_manual_large);
    println!("Tensor op:   {:?}", duration_tensor_large);
    if duration_manual_large < duration_tensor_large {
        let speedup =
            duration_tensor_large.as_nanos() as f64 / duration_manual_large.as_nanos() as f64;
        println!("Manual loop is {:.2}x faster", speedup);
    } else {
        let speedup =
            duration_manual_large.as_nanos() as f64 / duration_tensor_large.as_nanos() as f64;
        println!("Tensor operation is {:.2}x faster", speedup);
    }

    println!("\n=== Test map operation ===");
    println!("1. 1D Tensor - Double each element");
    let t1 = Tensor::build(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5])?;
    print!("  Original: ");
    t1.display();
    let result1 = t1.map(|_indices, value| value * 2.0);
    print!("  Mapped (x2): ");
    result1.display();

    println!("2. 2D Tensor - Scale by row index");
    let t2 = Tensor::build(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
    print!("  Original: ");
    t2.display();
    let result2 = t2.map(|indices, value| value * (indices[0] as f32 + 1.0));
    print!("  Mapped (x[row_index+1]): ");
    result2.display();

    println!("3. 3D Tensor - Sum indices and add to value");
    let t3 = Tensor::build(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])?;
    print!("  Original: ");
    t3.display();
    let result3 = t3.map(|indices, value| {
        let idx_sum: usize = indices.iter().sum();
        value + idx_sum as f32
    });
    print!("  Mapped (value + sum_of_indices): ");
    result3.display();

    println!("4. Large tensor (parallel execution, >100k elements)");
    let large_data: Vec<f32> = (0..150_000).map(|i| i as f32).collect();
    let t4 = Tensor::build(large_data, vec![150_000])?;
    println!("  Original size: {} elements", t4.data().len());
    let result4 = t4.map(|_indices, value| value.powf(2.0));
    println!("  Mapped (x^2) size: {} elements", result4.data().len());
    println!("  First 5 values: {:?}", &result4.data()[..5]);

    println!("  First 5 values: {:?}", &result4.data()[..5]);

    println!("\n=== Test formula + map ===");
    println!("5. Generate normal distribution values using formula");
    let mu = 0.0;
    let sigma = 1.0;
    println!("  Parameters: mu={}, sigma={}", mu, sigma);

    let x_range: Vec<f32> = (-30..=31).map(|i| i as f32 / 10.0).collect();
    println!(
        "  X range: [{:.1}, {:.1}, ...] with {} points",
        x_range[0],
        x_range[1],
        x_range.len()
    );

    let x_tensor = Tensor::build(x_range.clone(), vec![x_range.len()])?;
    let normal_values = x_tensor.map(|_indices, x| formula::normal_distribution(x, mu, sigma));

    println!("  Normal distribution computed with map + formula");
    print!(
        "  Peak value: {:.6} (at x={})\n",
        normal_values.data()[30],
        mu
    );
    println!("  Sample values at x=-3, -2, -1, 0, 1, 2, 3:");
    let indices_to_check = vec![0, 10, 20, 30, 40, 50, 60];
    for &idx in &indices_to_check {
        let x_val = x_tensor[idx];
        let normal_val = normal_values[idx];
        println!("    x={:.1} -> {:.6}", x_val, normal_val);
    }

    Ok(())
}
