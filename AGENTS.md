# M-Learn Development Guide

This guide helps agentic coding agents work effectively in the m-learn repository.

---

## Build, Lint, and Test Commands

### Build
```bash
# Build all workspace packages
cargo build

# Build specific package
cargo build -p tensor
```

### Lint
```bash
# Run clippy on all packages
cargo clippy

# Run clippy on specific package
cargo clippy -p tensor

# Fix clippy warnings automatically
cargo clippy --fix
```

### Format
```bash
# Format all code
cargo fmt

# Check formatting without making changes
cargo fmt --check
```

### Test
```bash
# Run all tests across workspace
cargo test

# Run tests for specific package
cargo test -p tensor

# Run a single test by name
cargo test -p tensor test_add_tensors
cargo test -p formula test_normal_distribution_standard

# Run tests with output (useful for debugging)
cargo test -- --nocapture

# Run tests with verbose output
cargo test -- -v
```

### Run Examples
```bash
cargo run -p tensor-example
cargo run -p draw-example
cargo run -p formula-draw-example
```

---

## Code Style Guidelines

### Types
- **Default numeric type**: `f32` for all tensor operations and mathematical computations
- **Error types**: Use `Result<T, E>` pattern with custom error enums
- **Error enum pattern**:
  ```rust
  #[derive(Debug, Clone)]
  pub enum TensorError {
      InvalidShape,
      IncompatibleShapes,
      IndexError { dim: usize, max: usize },
  }
  ```
- Implement `std::fmt::Display` and `std::error::Error` for custom error types

### Naming Conventions
- **Structs/Enums**: `PascalCase` (e.g., `Tensor`, `PlotConfig`, `TensorError`)
- **Functions**: `snake_case` (e.g., `normal_distribution`, `parallel_map_op`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `PARALLEL_THRESHOLD`, `FONT_REGISTER`)
- **Private fields**: Same name with `_` prefix in getters (e.g., `data() -> &Vec<f32>`)
- **Macros**: `snake_case!` (e.g., `tensor!()`, `make_tensor()`)

### Imports
- **External crates**: List at top of file, grouped by crate
- **Local modules**: Use `use crate::` path for workspace packages
- **Standard library**: Use full paths where clarity matters (e.g., `std::ops::Add`)

### Error Handling
- **Public functions**: Return `Result<T, ErrorType>` and document error cases
- **Error propagation**: Use `?` operator for clean error propagation
- **Validation**: Validate inputs early and return descriptive errors

### Parallel Execution
- **Threshold**: Use `PARALLEL_THRESHOLD = 100_000` for switching serial ↔ parallel
- **Thread pool**: Import `use rayon::prelude::*` for parallel iterators
- **Pattern**:
  ```rust
  const PARALLEL_THRESHOLD: usize = 100_000;

  if len < PARALLEL_THRESHOLD {
      data.iter().map(|x| process(x)).collect()
  } else {
      data.par_iter().map(|x| process(x)).collect()
  }
  ```

### Testing
- **Test modules**: Use `#[cfg(test)] mod tests { ... }` pattern
- **Helper functions**: Create `make_tensor()` or similar helpers for test setup
- **Test naming**: `test_<function>_<scenario>` (e.g., `test_add_tensors`)
- **Assertions**: Use `assert_eq!()` for equality, `assert!()` for boolean conditions

### Documentation
- **Public APIs**: Document with `///` doc comments
- **Examples**: Include runnable examples in doc comments
- **Parameters**: Document all parameters with types and constraints
- **Returns**: Document return types and possible errors

### Code Organization
- **Module structure**: One feature per file (e.g., `tensor.rs`, `operator.rs`, `display.rs`)
- **Re-exports**: Use `pub use` in lib.rs for clean public API
- **Visibility**: Keep internal logic `pub(crate)`, expose only necessary public API
- **Separation of concerns**: Split computation, display, and utilities into separate modules

### Performance Best Practices
- **Prevent allocations**: Use `Vec::with_capacity()` when size is known
- **Zero-copy**: Prefer references (`&Tensor`, `&[f32]`) over cloning
- **Parallel threshold**: Always check `len < PARALLEL_THRESHOLD` before parallelizing
- **Lazy evaluation**: Use iterators instead of collecting intermediate results

---

## Project-Specific Patterns

### Tensor Operations
- **Element-wise operations**: Implement using `map()` for consistency
- **Binary operations**: Validate shape compatibility before execution
- **Indexing**: Support multi-dimensional indexing via `Index` and `IndexMut` traits

### Formula Integration
- **Pure functions**: Formula functions should be pure (no side effects)
- **f32 precision**: All mathematical operations use f32 for consistency with tensors
- **Tensor compatibility**: Formula functions integrate via `tensor.map(|_indices, val| formula::func(val))`

### Draw Configuration
- **Builder pattern**: Use method chaining for `PlotConfig` (e.g., `.title().xlabel().ylabel()`)
- **Default implementation**: Implement `Default` trait for sensible defaults
- **Optional settings**: Use `Option<T>` for optional configuration fields

---

## Common Pitfalls

- **Do not use `as any` or `@ts-ignore`**: These do not apply to Rust code
- **Avoid unwrap() in production code**: Always handle errors properly
- **Do not ignore clippy warnings**: Fix or explicitly allow with `#[allow(clippy::...)]`
- **Do not use mutable references when immutable suffices**: Prefer immutability
- **Do not parallelize small datasets**: Use `PARALLEL_THRESHOLD` check to avoid overhead
- **Do not mix Result and Option**: Choose one error handling pattern consistently

---

## Crate Dependencies

- **tensor**: Depends on `optim` for thread pool
- **draw**: Depends on `tensor` for data structures
- **optim**: Standalone, no workspace dependencies
- **formula**: Standalone, no workspace dependencies

When adding new dependencies:
1. Check if similar functionality exists in existing crates
2. Prefer minimal dependencies
3. Update `Cargo.lock` by running `cargo build`
4. Document the dependency in the crate's README

---

## Workspace Structure

This is a Cargo workspace with multiple packages. Use `-p <package>` flag to target specific packages for build/test commands.
