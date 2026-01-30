use draw::{plot, plot_pairs, plot_series, ImageType, PlotConfig};
use tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Single (x, y) pair
    println!("Example 1: Single (x, y) pair with Tensor");
    let x = Tensor::build(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5])?;
    let y = Tensor::build(vec![2.0, 4.0, 6.0, 8.0, 10.0], vec![5])?;

    let config = PlotConfig::new()
        .title("Linear Function")
        .xlabel("X")
        .ylabel("Y")
        .x_range(0.0, 6.0)
        .y_range(0.0, 12.0)
        .x_ticks(7)
        .y_ticks(7)
        .show_window(false)
        .export("output/example1_linear.svg", ImageType::Svg);

    plot(&config, &x, &y)?;
    println!("Exported to example1_linear.svg\n");

    // Example 2: One x with multiple y series
    println!("Example 2: One x with multiple y series");
    let x = Tensor::build(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5])?;
    let y1 = Tensor::build(vec![0.0, 1.0, 4.0, 9.0, 16.0], vec![5])?;
    let y2 = Tensor::build(vec![0.0, 2.0, 8.0, 18.0, 32.0], vec![5])?;
    let y3 = Tensor::build(vec![0.0, 0.5, 2.0, 4.5, 8.0], vec![5])?;

    let config = PlotConfig::new()
        .title("Multiple Y Series (Same X)")
        .xlabel("X")
        .ylabel("Y")
        .legends(vec![
            "y = x^2".to_string(),
            "y = 2x^2".to_string(),
            "y = 0.5x^2".to_string(),
        ])
        .show_window(false)
        .export("output/example2_multi_y.svg", ImageType::Svg);

    plot_series(&config, &x, vec![&y1, &y2, &y3])?;
    println!("Exported to example2_multi_y.svg\n");

    // Example 3: Multiple (x, y) pairs
    println!("Example 3: Multiple (x, y) pairs");
    let x1 = Tensor::build(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5])?;
    let y1 = Tensor::build(vec![0.0, 1.0, 4.0, 9.0, 16.0], vec![5])?;

    let x2 = Tensor::build(vec![0.0, 0.5, 1.0, 1.5, 2.0], vec![5])?;
    let y2 = Tensor::build(vec![0.0, 2.0, 8.0, 18.0, 32.0], vec![5])?;

    let x3 = Tensor::build(vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![5])?;
    let y3 = Tensor::build(vec![0.0, 0.5, 2.0, 4.5, 8.0], vec![5])?;

    let config = PlotConfig::new()
        .title("Multiple (X, Y) Pairs")
        .xlabel("X")
        .ylabel("Y")
        .legends(vec![
            "Pair 1: x^2".to_string(),
            "Pair 2: 2x^2 (dense)".to_string(),
            "Pair 3: 0.5x^2 (sparse)".to_string(),
        ])
        .show_window(false)
        .export("output/example3_multi_pairs.svg", ImageType::Svg);

    plot_pairs(&config, vec![&x1, &x2, &x3], vec![&y1, &y2, &y3])?;
    println!("Exported to example3_multi_pairs.svg\n");

    Ok(())
}
