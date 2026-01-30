use draw::{plot, ImageType, PlotConfig};
use formula;
use tensor::{tensor, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Formula + Map + Draw Example ===\n");

    println!("1. Create x range tensor: [-5.0, 5.0] with step 0.1");
    let x_tensor = Tensor::range(-5.0, 5.0, 0.1)?;
    println!("   Created tensor with {} elements", x_tensor.data().len());
    println!("   First 3 values: {:?}", &x_tensor.data()[..3]);
    println!(
        "   Last 3 values: {:?}",
        &x_tensor.data()[x_tensor.data().len() - 3..]
    );

    println!("\n2. Compute normal distribution using map + formula");
    let mu = 0.0;
    let sigma = 1.0;
    println!("   Parameters: mu={}, sigma={}", mu, sigma);

    let y_tensor = x_tensor.map(|_indices, x| formula::normal_distribution(x, mu, sigma));
    println!(
        "   Computed y tensor with {} elements",
        y_tensor.data().len()
    );
    println!("   Peak value at x=0: {:.6}", y_tensor.data()[50]);

    println!("\n3. Plot normal distribution curve");
    let config = PlotConfig::new()
        .title("Normal Distribution N(0, 1)")
        .xlabel("x")
        .ylabel("Probability Density")
        .x_range(-5.0, 5.0)
        .y_range(0.0, 0.45)
        .x_ticks(11)
        .y_ticks(6)
        .legends(vec!["f(x) = N(0,1)".to_string()])
        .show_window(true)
        .export("output/normal_distribution.svg", ImageType::Svg);

    plot(&config, &x_tensor, &y_tensor)?;
    println!("   Saved to: output/normal_distribution.svg");

    println!("\n=== Example completed ===");
    Ok(())
}
