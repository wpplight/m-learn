# Formula Crate Documentation

Formula crate provides mathematical functions for tensor operations and data analysis.

## Overview

- **Normal distribution**: Probability density function
- **Extensible design**: Easy to add new mathematical functions
- **f32-based**: All operations work with f32 precision
- **Pure functions**: No side effects, deterministic outputs

---

## API Reference

### Probability Distribution

#### `normal_distribution(x, mu, sigma)`

Computes the probability density of the standard normal distribution at point x.

**Formula:**
```
f(x) = (1 / (σ * √(2π))) * e^(-0.5 * ((x - μ) / σ)^2)
```

**Where:**
- `x: f32` - Input value
- `mu: f32` - Mean (μ), center of distribution
- `sigma: f32` - Standard deviation (σ), spread of distribution
- `π` - Mathematical constant (~3.14159)

**Parameters:**
- `x: f32` - Point to evaluate
- `mu: f32` - Mean of distribution
- `sigma: f32` - Standard deviation (must be > 0)

**Returns:**
- `f32` - Probability density value

**Properties:**
- Peak at x = μ with value: `1 / (σ * √(2π))`
- Symmetric around mean: f(μ + δ) = f(μ - δ)
- Inflection points at μ ± σ
- Area under curve = 1.0

**Example 1: Standard normal distribution**
```rust
use formula;

let result = formula::normal_distribution(0.0, 0.0, 1.0);
// result ≈ 0.3989 (peak value)
```

**Example 2: Shifted mean**
```rust
use formula;

let result = formula::normal_distribution(1.0, 2.0, 1.0);
// Peak at x=2 with value ≈ 0.1995
```

**Example 3: Different sigma (spread)**
```rust
use formula;

let result = formula::normal_distribution(0.0, 0.0, 2.0);
// Flatter curve, lower peak ≈ 0.1995
```

---

## Mathematical Properties

### Normal Distribution

The standard normal distribution has the following characteristics:

**Parameters:**
- Mean (μ): Controls the center
- Standard deviation (σ): Controls the spread (width)

**Key Values:**
- **Peak**: `1 / (σ * √(2π))` - Occurs at x = μ
- **Inflection**: μ ± σ - Points where curvature changes
- **68% interval**: μ ± σ - Contains ~68% of probability mass
- **95% interval**: μ ± 1.96σ - Contains ~95% of probability mass
- **99.7% interval**: μ ± 2.58σ - Contains ~99.7% of probability mass

**Special Cases:**
- **σ → 0**: Approaches Dirac delta (spike at μ)
- **σ → ∞**: Approaches uniform distribution

---

## Usage with Tensor

The formula crate integrates seamlessly with tensor operations through the `map()` function.

### Element-wise calculation

Apply formula to each tensor element:

```rust
use tensor::{tensor, Tensor};
use formula;

let x = tensor!([-5.0, -4.0, ..., 5.0])?;
let pdf = x.map(|_indices, x_val| {
    formula::normal_distribution(x_val, 0.0, 1.0)
});
```

### Batch processing

Apply formula to entire tensor using thread pool:

```rust
use tensor::Tensor;
use formula;

let large_x = Tensor::arange(100000)?;
let pdf = large_x.map(|_indices, x_val| {
    formula::normal_distribution(x_val, 0.0, 1.0)
});
// Automatically uses parallel execution (>= 100,000 elements)
```

### Visualization

Combine with draw crate:

```rust
use tensor::Tensor;
use draw::{plot, PlotConfig, ImageType};
use formula;

let x = tensor!([-5.0, -4.0, ..., 5.0])?;
let y = x.map(|_indices, x_val| {
    formula::normal_distribution(x_val, 0.0, 1.0)
});

let config = PlotConfig::new()
    .title("Normal Distribution N(0, 1)")
    .xlabel("x")
    .ylabel("Probability Density")
    .export("output/normal.svg", ImageType::Svg);

plot(&config, &x, &y)?;
```

---

## Best Practices

1. **Parameter validation**: Ensure sigma > 0
2. **Performance**: Uses f32 operations, optimized for vectorization
3. **Integration**: Works seamlessly with tensor::map() for element-wise operations
4. **Parallelization**: Large tensors automatically processed in parallel via thread pool

---

## Future Extensions

Possible additions to the formula crate:
- **Other distributions**: Exponential, Gamma, Beta, Uniform
- **Statistical functions**: Mean, variance, covariance
- **Special functions**: Sigmoid, ReLU, Softmax
- **Trigonometric**: sin, cos, tan, etc.

---

## Implementation Details

The normal distribution function uses:
- `std::f32::consts::PI` - π constant
- `f32::sqrt()` - Square root
- `f32::exp()` - Exponential
- `f32::powi()` - Integer power

All operations are IEEE 754 floating-point compliant.

---

## Dependencies

- **std**: Core library (for constants, math functions)

---

## License

MIT
