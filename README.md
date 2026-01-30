# M-Learn - Machine Learning Library in Rust

A modular Rust library for machine learning with tensor operations, parallel computation, mathematical formulas, and visualization.

## Project Structure

```
m-learn/
├── crates/
│   ├── tensor/       # Tensor operations and data structures
│   ├── optim/        # Global thread pool and parallel execution
│   ├── draw/         # Plotting and visualization
│   └── formula/      # Mathematical formulas (distributions, etc.)
├── example/
│   ├── tensor-example/           # Tensor usage examples
│   ├── draw-example/             # Plotting examples
│   └── formula-draw-example/     # Formula + Tensor + Plot examples
└── src/
    └── global_thread_pool.rs  # Global thread pool initialization
```

---

## Crates Overview

### Tensor (`crates/tensor/`)

**Purpose**: Multi-dimensional array operations for ML and scientific computing

**Key Features:**
- Create tensors from data or using macros
- Element-wise mathematical operations (+, -, *, /, pow)
- Multi-dimensional indexing (1D, 2D, 3D, 4D)
- Slicing and mutation
- Element-wise mapping with position awareness (`map()`)
- Utility functions (zeros, arange, rand, randn, sum, reshape, display)

**Documentation**: [crates/tensor/README.md](crates/tensor/README.md)

---

### Optim (`crates/optim/`)

**Purpose**: High-performance parallel execution with thread pool management

**Key Features:**
- Global singleton Rayon thread pool
- Parallel iterator operations (par_iter, par_iter_mut)
- Batch processing (par_batches)
- Task execution (execute)
- Thread pool information (pool_info)

**Performance**: Automatic speedup (3-5x) for operations on >=100,000 elements

**Documentation**: [crates/optim/README.md](crates/optim/README.md)

---

### Draw (`crates/draw/`)

**Purpose**: Tensor data visualization and plotting

**Key Features:**
- SVG export (scalable vector graphics)
- Interactive window display (minifb)
- Single series plotting
- Multiple series plotting (same X, multiple Y)
- Multiple (X, Y) pairs plotting
- Configurable: titles, labels, ranges, ticks, colors

**Dependencies**: plotters (rendering), minifb (display), dejavu (fonts)

**Documentation**: [crates/draw/README.md](crates/draw/README.md)

---

### Formula (`crates/formula/`)

**Purpose**: Mathematical functions for data analysis and ML

**Key Features:**
- Normal distribution (probability density function)
- Extensible design for adding new formulas
- f32 precision with IEEE 754 compliance
- Seamless integration with tensor operations

**Current Functions:**
- `normal_distribution(x, mu, sigma)` - Standard normal distribution PDF

**Mathematical Formula:**
```
f(x) = (1 / (σ * √(2π))) * e^(-0.5 * ((x - μ) / σ)^2)
```

**Documentation**: [crates/formula/README.md](crates/formula/README.md)

---

## Getting Started

### Dependencies

The project uses these key dependencies:
- **rayon**: Parallel iterator library
- **plotters**: Chart rendering
- **minifb**: Window display
- **dejavu**: Font rendering
- **once_cell**: Lazy static initialization
- **num_cpus**: CPU core detection

All dependencies are in `Cargo.lock`.

### Build and Test

```bash
# Build all crates
cargo build

# Run tests
cargo test

# Run specific example
cargo run -p tensor-example
cargo run -p draw-example
cargo run -p formula-draw-example
```

---

## Quick Examples

### Tensor Operations

```rust
use tensor::{tensor, Tensor};

// Create and compute
let x = tensor!([1.0, 2.0, 3.0]);
let y = x + 1.0;           // [2.0, 3.0, 4.0]
let squared = x.pow(2.0);      // [1.0, 4.0, 9.0]

// Indexing
let t = tensor!([[1.0, 2.0], [3.0, 4.0]]);
let val = t[[0, 1]];  // = 2.0

// Element-wise mapping
let doubled = x.map(|_indices, v| v * 2.0);
```

### Mathematical Formulas

```rust
use tensor::Tensor;
use formula;

let x = tensor!([-2.0, -1.0, 0.0, 1.0, 2.0])?;
let pdf = x.map(|_indices, x_val| {
    formula::normal_distribution(x_val, 0.0, 1.0)
});
```

### Plotting

```rust
use tensor::Tensor;
use draw::{plot, PlotConfig, ImageType};

let x = tensor!([0.0, 1.0, 2.0, 3.0]);
let y = x * 2.0;

let config = PlotConfig::new()
    .title("Linear Growth")
    .xlabel("Time")
    .ylabel("Value")
    .export("output/plot.svg", ImageType::Svg);

plot(&config, &x, &y)?;
```

### Formula + Tensor + Plot

```rust
use tensor::Tensor;
use draw::{plot, PlotConfig, ImageType};
use formula;

let x = tensor!([-5.0, -4.9, ..., 5.0])?;
let y = x.map(|_indices, x_val| {
    formula::normal_distribution(x_val, 0.0, 1.0)
});

let config = PlotConfig::new()
    .title("Normal Distribution N(0, 1)")
    .export("output/normal.svg", ImageType::Svg);

plot(&config, &x, &y)?;
```

---

## Performance Characteristics

- **Automatic parallelization**: Tensor operations use thread pool for >=100,000 elements
- **Thread pool**: Uses all available CPU cores
- **Speedup**: Typically 3-5x for CPU-bound operations on large datasets
- **Overhead**: Small operations (<100,000 elements) run serially to avoid thread creation cost

---

## License

MIT License - See individual crate LICENSE files for details

---

## Development

### Adding New Features

1. **Tensor crate**: Add new operations, index types, slicing variants
2. **Optim crate**: Add new parallel patterns, task schedulers
3. **Draw crate**: Add new plot types, export formats, styling options
4. **Formula crate**: Add new distributions, statistical functions, activation functions

### Code Style

- Use clippy for additional linting
- Format code with `cargo fmt`
- Add tests for new functionality
- Update documentation

---

## Resources

- [Rust](https://www.rust-lang.org/)
- [Rayon Documentation](https://docs.rs/rayon/rayon/)
- [Plotters Documentation](https://plotters-rs.github.io/)
