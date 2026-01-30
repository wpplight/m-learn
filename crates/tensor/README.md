# Tensor Crate Documentation

Tensor crate provides multidimensional array operations for machine learning and scientific computing.

## Overview

- **Create tensors**: `Tensor::build()`, `tensor!` macro
- **Mathematical operations**: `+`, `-`, `*`, `/`, `pow()`
- **Indexing**: `tensor[1]`, `tensor[[0, 1]]`, `tensor[[0, 1, 2]]`, etc.
- **Slicing**: `tensor.slice(range)`
- **Element-wise mapping**: `tensor.map()` - NEW
- **Utility functions**: `zeros()`, `arange()`, `rand()`, `randn()`, `sum()`, `reshape()`, `display()`
- **Nested arrays**: Supports tensor-of-tensors for N-dimensional indexing

---

## API Reference

### Create Tensors

#### `Tensor::build(data, shape)`

Creates a tensor from data vector and shape.

**Parameters:**
- `data: Vec<f32>` - Flat data array
- `shape: Vec<usize>` - Dimension sizes (e.g., `[2, 3]` for 2x3 matrix)

**Returns:**
- `Result<Tensor, TensorError>` - Tensor or error if data/shape mismatch

**Behavior:**
- Validates that `data.len() == shape.iter().product()`
- Creates new Tensor by copying data into internal storage

**Example:**
```rust
use tensor::Tensor;

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let shape = vec![2, 3];
let tensor = Tensor::build(data, shape)?;
```

#### `tensor!([values])` macro

Convenient macro for creating 1D tensors from literal values.

**Parameters:**
- `values` - Comma-separated f32 values

**Returns:**
- `Result<Tensor, TensorError>` - Tensor or error

**Advantages:**
- No need to create separate data vector
- Shape is inferred from value count (1D)
- Clean, readable code

**Example:**
```rust
use tensor::{tensor, Tensor};

let t = tensor!([1.0, 2.0, 3.0, 4.0, 5.0]);
// Creates tensor with shape=[5]
// t.data() = [1.0, 2.0, 3.0, 4.0, 5.0]
```

---

### Mathematical Operations

#### `tensor + other_tensor`

Element-wise addition of two tensors.

**Parameters:**
- `self: Tensor` - Left operand (any reference)
- `other: Tensor` - Right operand (any reference)

**Returns:**
- `Result<Tensor, TensorError>` - Result tensor or error if shapes don't match

**Requirements:**
- Both tensors must have identical `shape` vectors
- Shapes must be compatible for addition

**Behavior:**
- Element-wise operation: each element at position `(i0, i1, ...)` is added to `(i0, i1, ...) + (j0, j1, ...)`
- Creates new tensor with same shape as inputs
- Uses parallel execution for >= 100,000 elements

**Performance:**
- Small tensors (< 100k elements): Serial to avoid thread pool overhead
- Large tensors (>= 100k elements): Automatic parallelization via global thread pool

**Example:**
```rust
let x = Tensor::build(vec![1.0, 2.0, 3.0], vec![2])?;
let y = Tensor::build(vec![3.0, 4.0, 5.0], vec![2])?;
let result = (x + y)?;
// result.data() = [4.0, 6.0, 8.0]
```

---

#### `tensor - other_tensor`

Element-wise subtraction of two tensors.

**Parameters:**
- `self: Tensor` - Left operand
- `other: Tensor` - Right operand

**Returns:**
- `Result<Tensor, TensorError>` - Result tensor or error if shapes don't match

**Example:**
```rust
let x = Tensor::build(vec![5.0, 6.0], vec![2])?;
let y = Tensor::build(vec![2.0, 1.0], vec![2])?;
let result = (x - y)?;
// result.data() = [3.0, 5.0]
```

---

#### `tensor * other_tensor`

Element-wise multiplication of two tensors.

**Parameters:**
- `self: Tensor` - Left operand
- `other: Tensor` - Right operand

**Returns:**
- `Result<Tensor, TensorError>` - Result tensor or error if shapes don't match

**Example:**
```rust
let x = Tensor::build(vec![1.0, 2.0], vec![2])?;
let y = Tensor::build(vec![3.0, 4.0], vec![2])?;
let result = (x * y)?;
// result.data() = [3.0, 8.0]
```

---

#### `tensor / other_tensor`

Element-wise division of two tensors.

**Parameters:**
- `self: Tensor` - Left operand (dividend)
- `other: Tensor` - Right operand (divisor)

**Returns:**
- `Result<Tensor, TensorError>` - Result tensor or error if shapes don't match
- Division by zero returns `NaN` values

**Example:**
```rust
let x = Tensor::build(vec![4.0, 9.0], vec![2])?;
let y = Tensor::build(vec![2.0, 3.0, 3.0], vec![2])?;
let result = (x / y)?;
// result.data() = [2.0, 3.0, 3.0]
```

---

#### `tensor + scalar`

Adds scalar value to all elements in a tensor.

**Parameters:**
- `self: Tensor` - Tensor operand (any reference)
- `scalar: f32` - Scalar value to add

**Returns:**
- `Tensor` - New tensor with scalar added to all elements

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0], vec![3])?;
let result = t + 5.0;
// result.data() = [6.0, 7.0, 8.0]
```

---

#### `tensor - scalar`

Subtracts scalar value from all elements in a tensor.

**Parameters:**
- `self: Tensor` - Tensor operand
- `scalar: f32` - Scalar value to subtract

**Returns:**
- `Tensor` - New tensor with scalar subtracted from all elements

**Example:**
```rust
let t = Tensor::build(vec![10.0, 15.0, 20.0], vec![3])?;
let result = t - 5.0;
// result.data() = [5.0, 10.0, 15.0]
```

---

#### `tensor * scalar`

Multiplies all elements in a tensor by a scalar value.

**Parameters:**
- `self: Tensor` - Tensor operand
- `scalar: f32` - Scalar value to multiply

**Returns:**
- `Tensor` - New tensor with all elements multiplied by scalar

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0], vec![3])?;
let result = t * 3.0;
// result.data() = [3.0, 6.0, 9.0]
```

---

#### `tensor / scalar`

Divides all elements in a tensor by a scalar value.

**Parameters:**
- `self: Tensor` - Tensor operand (dividend)
- `scalar: f32` - Scalar value to divide

**Returns:**
- `Tensor` - New tensor with all elements divided by scalar

**Common errors:**
- Division by zero: Elements where `scalar == 0` become `NaN`

**Example:**
```rust
let t = Tensor::build(vec![6.0, 9.0, 12.0], vec![3])?;
let result = t / 3.0;
// result.data() = [2.0, 3.0, 4.0]
```

---

#### `tensor.pow(exp)`

Raises all elements in a tensor to a power.

**Parameters:**
- `self: Tensor` - Tensor operand
- `exp: f32` - Exponent to raise each element to

**Returns:**
- `Tensor` - New tensor with all elements raised to power

**Example:**
```rust
let t = Tensor::build(vec![2.0, 3.0], vec![2])?;
let result = t.pow(2.0);
// result.data() = [4.0, 9.0]
```

---

### Indexing

#### `tensor[linear_index]`

Access element by 0-based linear index.

**Parameters:**
- `linear_index: usize` - Linear position in flat data array

**Returns:**
- `&f32` - Reference to element at that index

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0], vec![3])?;
let value = t[1];  // = 2.0
```

**Note:** Equivalent to `t.data()[linear_index]` but more readable

---

#### `tensor[[row, col]]`

Access 2D tensor element by row and column indices.

**Parameters:**
- `[usize; 2]` - `[row_index, col_index]`

**Returns:**
- `&f32` - Reference to element at that position

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]])?;
let value = t[[1, 0]];  // = 3.0 (row 1, col 0)
let value2 = t[[0, 1]];  // = 4.0 (row 0, col 1)
```

**Note:** Row indices are 0-based for first dimension

---

#### `tensor[d0, d1, d2]`

Access 3D tensor element by three dimension indices.

**Parameters:**
- `[usize; 3]` - `[dim0_index, dim1_index, dim2_index]`

**Returns:**
- `&f32` - Reference to element

**Example:**
```rust
let t = Tensor::build(
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    vec![2, 2, 2]
)?;
let value = t[[1, 1, 1]];  // = 8.0
```

---

#### `tensor[d0, d1, d2, d3]`

Access 4D tensor element by four dimension indices.

**Parameters:**
- `[usize; 4]` - Four-dimensional indices

**Returns:**
- `&f32` - Reference to element

**Example:**
```rust
let t = Tensor::build(
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    vec![2, 2, 2, 2]
)?;
let value = t[[1, 1, 1, 1]];  // = 8.0
```

---

#### `tensor[index] = value`

Mutates tensor element by linear index (in-place modification).

**Example:**
```rust
let mut t = Tensor::build(vec![1.0, 2.0, 3.0], vec![3])?;
t[1] = 10.0;
// Now t.data() = [10.0, 2.0, 3.0]
```

**Note:** This is an in-place operation - modifies the original tensor directly

---

### Slicing

#### `tensor.slice(range)`

Extracts a slice along the first dimension.

**Parameters:**
- `range: Range<usize>` - Range of first dimension indices

**Returns:**
- `Result<Tensor, TensorError>` - Sliced tensor or error if out of bounds

**Requirements:**
- `start < end` - Valid range (must be non-empty)
- `start <= shape[0]` - Start must be within first dimension bounds
- `end <= shape[0]` - End must be within first dimension bounds

**Behavior:**
- Creates new tensor with reduced first dimension
- Preserves remaining dimensions
- Data is contiguous (not copied unless necessary)

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])?;
let sliced = t.slice(0..2)?;  // First 2 rows
// sliced.shape() = [2, 2]
// sliced.data() = [1.0, 2.0, 3.0, 4.0]
```

---

### Element-wise Mapping

#### `tensor.map(closure)` - NEW

Applies a custom function to each element with access to its multi-dimensional position indices.

**Parameters:**
- `closure: F` where `F: Fn(&[usize], f32) -> f32 + Sync + Send`

**Closure receives:**
- `indices: &[usize]` - Multi-dimensional position (e.g., `[1, 0]` for 2D tensor, `[0, 1, 1]` for 3D tensor)
- `value: f32` - Current element value at that position

**Returns:**
- `Tensor` - New tensor with transformed values (original tensor is unchanged)

**Behavior:**
- Element count < 100,000: Serial execution
- Element count >= 100,000: Parallel execution (uses global thread pool)
- Always creates new tensor (immutable operation)

**Example 1: Simple element-wise operation**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0, 4.0], vec![4])?;
let doubled = t.map(|_indices, value| value * 2.0);
// doubled.data() = [2.0, 4.0, 6.0, 8.0]
```

**Example 2: Use position index for conditional scaling**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]])?;
let scaled = t.map(|indices, value| {
    let row = indices[0];
    let col = indices[1];
    value * (row as f32 + 1.0)  // Scale by row index
});
```

**Example 3: Complex calculation with formula crate**
```rust
use tensor::Tensor;
use formula;

let x = Tensor::arange(100)?;
let pdf = x.map(|_indices, x_val| {
    formula::normal_distribution(x_val, 0.0, 1.0)
});
```

**Note:** This demonstrates seamless integration between tensor and formula crates

---

### Utility Functions

#### `Tensor::zeros(shape)`

Creates a tensor filled with zeros.

**Parameters:**
- `shape: Vec<usize>` - Dimension sizes

**Returns:**
- `Result<Tensor, TensorError>` - Zero tensor or error

**Example:**
```rust
let zeros = Tensor::zeros(vec![2, 3])?;
// zeros.data() = [0.0, 0.0, 0.0, 0.0, 0.0]
```

#### `Tensor::arange(n)`

Creates a 1D tensor with values 0 to n-1.

**Parameters:**
- `n: usize` - Number of elements

**Returns:**
- `Tensor` - Tensor with [0.0, 1.0, 2.0, ..., n-1]

**Example:**
```rust
let t = Tensor::arange(5)?;
// t.data() = [0.0, 1.0, 2.0, 3.0, 4.0]
```

#### `Tensor::rand(shape)`

Creates tensor with uniform random values [0, 1).

**Parameters:**
- `shape: Vec<usize>` - Dimension sizes

**Returns:**
- `Result<Tensor, TensorError>` - Random tensor or error

**Example:**
```rust
let random = Tensor::rand(vec![2, 3])?;
// Each value is in [0.0, 1.0]
```

#### `Tensor::randn(shape)`

Creates tensor with standard normal distribution (μ=0, σ=1).

**Parameters:**
- `shape: Vec<usize>` - Dimension sizes

**Returns:**
- `Result<Tensor, TensorError>` - Normal random tensor or error

**Distribution properties:**
- Mean (μ): 0 by default
- Standard deviation (σ): 1 by default
- Values follow N(0, 1) distribution
- Peak at x=μ: `1 / (1 * √(2π))` ≈ 0.3989

**Example:**
```rust
let normal = Tensor::randn(vec![2, 3])?;
// Values follow N(0, 1) distribution
```

#### `tensor.sum()`

Returns sum of all elements.

**Returns:**
- `f32` - Sum of all elements

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0], vec![3])?;
let total = t.sum();  // = 6.0
```

#### `tensor.reshape(new_shape)`

Changes tensor shape while preserving data.

**Parameters:**
- `new_shape: Vec<usize>` - New dimension sizes
- `total_elements: usize` - Must equal `new_shape.iter().product()`

**Returns:**
- `Result<Tensor, TensorError>` - Reshaped tensor or error if invalid

**Requirements:**
- Same total element count: `data.len() == new_shape.iter().product()`
- Compatible dimensions: No dimension can be arbitrarily split or combined

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
let reshaped = t.reshape(vec![4])?;
// reshaped.shape() = [4]
// reshaped.data() = [1.0, 2.0, 3.0, 4.0] (same data)
```

**Memory efficiency:**
- Reshape uses `Vec::from()` internally - no data copy for same total size
- Contiguous reshape is zero-cost view transformation

#### `tensor.display()`

Prints tensor information to console for debugging.

**Output format:**
```
Tensor(shape=[d0, d1, ...]):
[[v00, v01, ..., vmn]]
```

**Usage:**
- Debug tensor contents during development
- Verify data after transformations
- Log tensor statistics in training loops

---

### Nested Array Support

Tensor supports tensor-of-tensors (N-dimensional arrays) via multi-dimensional indexing:

**1D tensors**: `Tensor<data=[N]>`
- 2D tensors: `Tensor<data=[rows, cols]>`
- 3D tensors: `Tensor<data=[depth, height, width]>`
- 4D tensors: `Tensor<data=[d0, d1, d2, d3]>`

Create nested tensors using `tensor!` macro with nested arrays:

```rust
let t2d = tensor!([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
]);  // 2x3 matrix
// Access: t2d[[row, col]]
```

---

## Performance Notes

- **Automatic parallelization**: Tensor operations use global thread pool for >= 100,000 elements
- **Parallel threshold**: `PARALLEL_THRESHOLD = 100,000`
- **Thread pool**: Uses all available CPU cores via rayon
- **Zero-copy design**: `map()` creates new tensor without copying original data

**Performance characteristics:**
- Small tensors (< 100k): ~10ms per 10k elements
- Large tensors (>= 100k): ~2ms per 100k elements (5x speedup)
- Parallel scaling: Near-linear with CPU core count

---

## Error Types

### `TensorError`

Possible errors:
- `InvalidShape` - Data length doesn't match shape product
- `IncompatibleShapes` - Tensor shapes don't match for binary operation
- `IndexError { dim, max }` - Index out of bounds for a dimension

Each error includes descriptive message for debugging.

---

## Best Practices

1. **Validate before operations**: Check shape compatibility before binary operations
2. **Use Result<Tensor, TensorError>**: Handle errors gracefully with `?` operator
3. **Prefer immutable operations**: Use `map()` over in-place mutations when possible
4. **Optimize large datasets**: Let automatic parallelization handle performance
5. **Check indices**: Validate indices against data length before access

---

## API Reference

### Create Tensors

#### `Tensor::build(data, shape)`

Creates a tensor from data vector and shape.

**Parameters:**
- `data: Vec<f32>` - Flat data array
- `shape: Vec<usize>` - Dimension sizes (e.g., `[2, 3]` for 2x3 matrix)

**Returns:**
- `Result<Tensor, TensorError>` - Tensor or error if data/shape mismatch

**Usage:**
- Suitable when you have raw data arrays
- Must ensure `data.len() == shape.iter().product()`
- Use for creating tensors from CSV, JSON, or computation results

**Common errors:**
- `InvalidShape`: Data length doesn't match shape product
  - Solution: Check `data.len()` equals expected total

**Example:**
```rust
use tensor::Tensor;

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let shape = vec![2, 3];
let tensor = Tensor::build(data, shape)?;
// 2x3 matrix with values [[1,2,3], [4,5,6]]
```

**Best practices:**
- Validate data length before calling build
- Use `let total: usize = shape.iter().product()` to check

---

#### `tensor!([values])` macro

Convenient macro for creating tensors from literal values.

**Parameters:**
- `values` - Comma-separated f32 values (inferred 1D shape)

**Returns:**
- `Result<Tensor, TensorError>` - Tensor or error

**Usage:**
- Use for quick tensor creation with known values
- Simpler than `Tensor::build()` for constant data
- Shape is inferred from value count

**Advantages:**
- No need to create separate data vector
- Clean, readable code
- Compile-time shape verification

**Example:**
```rust
use tensor::{tensor, Tensor};

let t = tensor!([1.0, 2.0, 3.0, 4.0]);
// Creates 1D tensor: [1.0, 2.0, 3.0, 4.0]

let matrix = tensor!([1.0, 2.0, 3.0, 4.0]);
// Creates 2D tensor: [[1.0, 2.0], [3.0, 4.0]]

// Complex tensor with mixed dimensions
let t2 = tensor!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
// Creates 2x4 tensor: [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
```

**Best practices:**
- Use for test data and small constant tensors
- Prefer over `Tensor::build()` when values are literals
- Good for initialization and configuration

---

#### `tensor + other_tensor`

Convenient macro for creating tensors from literal values.

**Parameters:**
- `values` - Comma-separated f32 values (infers 1D shape)

**Returns:**
- `Result<Tensor, TensorError>` - Tensor or error

**Example:**
```rust
use tensor::{tensor, Tensor};

let t = tensor!([1.0, 2.0, 3.0, 4.0]);
```

---

### Mathematical Operations

#### `tensor + other_tensor`

Element-wise addition of two tensors.

**Parameters:**
- `self: Tensor` - Left operand
- `other: Tensor` - Right operand

**Returns:**
- `Result<Tensor, TensorError>` - Result tensor or error if shapes don't match

**Requirements:**
- Both tensors must have identical `shape` vectors
- Example: `x.shape() == y.shape()` required for valid addition

**Common errors:**
- `IncompatibleShapes`: Returned when shapes don't match
  - Solution: Check tensor shapes before operation, or use `reshape()` to make them compatible

**Usage scenarios:**
- Element-wise arithmetic: Adding two tensors of same shape
- Broadcasting: Not currently supported (both must have identical shapes)
- Matrix operations: Not supported (use linear algebra libraries for matrix math)

**Performance:**
- Automatically uses parallel execution for >= 100,000 elements
- Uses global thread pool via optim crate

**Best practices:**
- Always validate shape compatibility before binary operations
- Use shape queries: `x.shape()` and `y.shape()` to check dimensions match
- Handle `Result<Tensor, TensorError>` properly with `?` operator for error propagation

**Example:**
```rust
use tensor::Tensor;

let x = Tensor::build(vec![1.0, 2.0, 3.0, 4.0], vec![2])?;
let y = Tensor::build(vec![3.0, 4.0, 5.0, 6.0], vec![2])?;

match (x + y) {
    Ok(result) => {
        println!("Result: {:?}", result.data());
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
// Output: [4.0, 6.0, 8.0]
```

---

#### `tensor - other_tensor`

Element-wise subtraction of two tensors.

**Parameters:**
- `self: Tensor` - Left operand (minuend)
- `other: Tensor` - Right operand (subtrahend)

**Returns:**
- `Result<Tensor, TensorError>` - Result tensor or error if shapes don't match

**Requirements:**
- Both tensors must have identical `shape` vectors
- Subtraction is commutative: `(x - y) != (y - x)`

**Usage scenarios:**
- Element-wise subtraction: Subtracting corresponding values
- Difference calculation: Computing how much one tensor differs from another

**Performance:**
- Automatically uses parallel execution for >= 100,000 elements

**Best practices:**
- Use subtraction for finding differences between datasets
- Validate shapes before operation

**Example:**
```rust
use tensor::Tensor;

let baseline = Tensor::build(vec![5.0, 6.0, 7.0], vec![4])?;
let measured = Tensor::build(vec![4.5, 5.8, 6.7], vec![4])?;

let difference = (baseline - measured)?;
// difference.data() = [0.5, 0.2, 0.3]
```

---

#### `tensor * other_tensor`

Element-wise multiplication of two tensors.

**Parameters:**
- `self: Tensor` - Left operand
- `other: Tensor` - Right operand

**Returns:**
- `Result<Tensor, TensorError>` - Result tensor or error if shapes don't match

**Requirements:**
- Both tensors must have identical `shape` vectors
- Multiplication is commutative and associative

**Usage scenarios:**
- Element-wise multiplication: Scaling a tensor by another
- Hadamard product: Not supported (use dedicated linear algebra library)
- Element-wise operations in ML: Used in activation functions, weight application

**Performance:**
- Automatically uses parallel execution for >= 100,000 elements
- O(n) complexity where n is total elements

**Best practices:**
- Validate shapes before multiplication
- Use for matrix operations with external linear algebra crates
- Leverage automatic parallelization for large tensors

**Example:**
```rust
use tensor::Tensor;

let weights = Tensor::build(vec![0.5, 0.1, 0.2], vec![3])?;
let activations = Tensor::build(vec![1.0, 2.0, 3.0], vec![3])?;

let output = (weights * activations)?;
// Element-wise multiplication: [0.5, 0.2, 0.6]
```

---

#### `tensor / other_tensor`

Element-wise division of two tensors.

**Parameters:**
- `self: Tensor` - Left operand (dividend)
- `other: Tensor` - Right operand (divisor)

**Returns:**
- `Result<Tensor, TensorError>` - Result tensor or error if shapes don't match

**Requirements:**
- Both tensors must have identical `shape` vectors
- Divisor elements should not be zero (undefined behavior)
- Division by zero returns `NaN` values

**Common errors:**
- Division by zero: Elements where `other` is zero become `NaN`
  - Solution: Check for zero values before division, or use `tensor.clip()` to limit values

**Performance:**
- Automatically uses parallel execution for >= 100,000 elements

**Usage scenarios:**
- Normalization: Divide each tensor by a scaling factor
- Probability adjustment: Convert log-probabilities using division
- Statistical calculations: Computing ratios between datasets

**Best practices:**
- Add small epsilon to avoid division by zero: `x / (y + 1e-6)`
- Validate divisor is non-zero before division
- Handle `NaN` results appropriately in subsequent calculations

**Example:**
```rust
use tensor::Tensor;

let total = Tensor::build(vec![100.0], vec![1])?;
let count = Tensor::build(vec![10.0, 20.0], vec![1])?;

let proportions = (count / total)?;
// proportions.data() = [0.1, 0.2]
// Each element: 10/100 = 0.1, 20/100 = 0.2
```

---

#### `tensor + scalar`

Adds a scalar value to all elements in the tensor.

**Parameters:**
- `self: Tensor` - Tensor operand
- `scalar: f32` - Scalar value to add

**Returns:**
- `Tensor` - New tensor with scalar added to all elements

**Requirements:**
- Scalar value must be a valid `f32`
- Can be used with both immutable references (`&Tensor`) and owned values (`Tensor`)

**Usage scenarios:**
- Bias addition: Add constant offset to all activations
- Gradient adjustment: Add small learning rate to gradients
- Data augmentation: Add noise or offset to training data
- Thresholding: Add threshold to binary classification outputs

**Performance:**
- Automatically uses parallel execution for >= 100,000 elements
- O(n) complexity

**Best practices:**
- Use broadcasting syntax: `tensor + scalar` is more readable than loops
- Combine scalar additions: `tensor + a + b` more efficient than `tensor + a; tensor + b`
- Prefer scalar operations over element-wise loops when possible

**Example:**
```rust
use tensor::Tensor;

let logits = Tensor::build(vec![0.5, 1.2, -0.8], vec![4])?;

// Add bias (scalar)
let with_bias = logits + 0.1;
// with_bias.data() = [0.6, 1.3, -0.7]

// Learning rate (small scalar multiplier)
let lr = 0.01;
let gradients = Tensor::build(vec![0.1, 0.05, -0.03], vec![4])?;
let updated = logits - (gradients * lr)?;
// Backpropagation: apply learning rate
```

---

#### `tensor - scalar`

Subtracts a scalar value from all elements in the tensor.

**Parameters:**
- `self: Tensor` - Tensor operand
- `scalar: f32` - Scalar value to subtract

**Returns:**
- `Tensor` - New tensor with scalar subtracted from all elements

**Requirements:**
- Scalar value must be a valid `f32`

**Usage scenarios:**
- Gradient subtraction: Subtract learning rate from accumulated gradients
- Mean centering: Subtract dataset mean from each sample
- Bias removal: Subtract attention bias from attention maps
- Normalization: Subtract mean and divide by std for standardization

**Performance:**
- Automatically uses parallel execution for >= 100,000 elements

**Best practices:**
- For gradient subtraction, ensure tensors have same shape
- For standardization, compute mean once and reuse: `x - x_mean`

**Example:**
```rust
use tensor::Tensor;

let predictions = Tensor::build(vec![0.8, 0.6, 0.4], vec![3])?;
let targets = Tensor::build(vec![1.0, 0.5, 0.3], vec![3])?;

// Simple error (loss)
let error = predictions - targets;
// error.data() = [-0.2, 0.1, 0.1]

// For gradient descent (subtract LR * error)
let adjusted = predictions - (error * 0.01);
```

---

#### `tensor * scalar`

Multiplies all elements in the tensor by a scalar value.

**Parameters:**
- `self: Tensor` - Tensor operand
- `scalar: f32` - Scalar value to multiply

**Returns:**
- `Tensor` - New tensor with all elements multiplied by scalar

**Requirements:**
- Scalar value must be a valid `f32`

**Usage scenarios:**
- Learning rate application: Multiply gradients by learning rate
- Weight updates: Scale accumulated gradients by decay factor
- Signal scaling: Multiply activations by alpha (0 < α < 1)
- Normalization: Multiply by standard deviation for standard scores
- Temperature scaling: Divide logits by temperature in softmax

**Performance:**
- Automatically uses parallel execution for >= 100,000 elements
- O(n) complexity

**Best practices:**
- Use small decay factors: `0.99`, `0.999` for momentum
- Combine scalar multiplications: `tensor * a * b` for sequential operations
- For temperature scaling, ensure `scalar > 0` to avoid division by zero
- Clip values after multiplication if needed to maintain stability

**Example:**
```rust
use tensor::Tensor;

let weights = Tensor::build(vec![0.1, 0.5, 0.8], vec![3])?;
let gradients = Tensor::build(vec![0.2, 1.0, -0.1], vec![3])?;
let lr = 0.01;

// Update with decay (gradient * lr * 0.99)
let new_weights = weights - ((gradients * lr) * 0.99);

// Learning rate schedule (grad * 0.01)
let updated = weights - (gradients * lr);
```

---

#### `tensor / scalar`

Divides all elements in the tensor by a scalar value.

**Parameters:**
- `self: Tensor` - Tensor operand (dividend)
- `scalar: f32` - Scalar value to divide

**Returns:**
- `Tensor` - New tensor with all elements divided by scalar

**Requirements:**
- Scalar value must be a valid `f32` - Required for valid division
- Scalar value should not be zero (unless `NaN` result is desired)

**Common errors:**
- Division by zero: All elements where `scalar == 0` become `NaN`
  - Solution: Add small epsilon to scalar: `scalar + 1e-6`

**Usage scenarios:**
- Normalization: Divide by standard deviation: `x / std_dev`
- Learning rate schedule: Divide learning rate over time
- Gradient averaging: Divide accumulated gradients by count
- Probability computation: Convert logits to probabilities using division

**Performance:**
- Automatically uses parallel execution for >= 100,000 elements
- O(n) complexity

**Best practices:**
- Use epsilon for division to avoid NaN: `x / (std + 1e-6)`
- For normalization, compute standard deviation once: `let std = x.std();` then `x / std`
- Avoid in-place division in hot loops

**Example:**
```rust
use tensor::Tensor;

let x = Tensor::build(vec![1.0, 2.0, 4.0, 5.0, 6.0], vec![5])?;
let std = 1.5;

// Standard normalization
let normalized = x / std;
// normalized.data() = [0.667, 1.333, 2.667, 3.333, 4.0]

// Learning rate decay (epoch 100: initial_lr / 100.0)
let lr = 0.01;
let epoch = 100.0;
let decayed_lr = lr / epoch;
// decayed_lr = 0.0001

// Gradient averaging (sum / count)
let grad_sum = gradients.sum()?;
let count = 10.0;
let avg_grad = grad_sum / count;
```

---

#### `tensor.pow(exp)`

Raises each element in the tensor to a power.

**Parameters:**
- `self: Tensor` - Tensor operand
- `exp: f32` - Exponent to raise each element to

**Returns:**
- `Tensor` - New tensor with each element raised to `exp`

**Requirements:**
- `exp: f32` - Must be a valid floating-point number
- Negative exponents: Result contains reciprocals (e.g., `x^-2 = 1/x^2`)

**Usage scenarios:**
- Squared error: `error.pow(2.0)` for MSE loss computation
- L2 regularization: Add squared values with coefficient: `x.pow(2.0) * lambda`
- Polynomial features: Create higher-order terms: `x.pow(3.0)`, `x.pow(4.0)`
- Root operations: `x.pow(0.5)` for square root

**Performance:**
- Automatically uses parallel execution for >= 100,000 elements
- Uses `f32::powi()` for integer exponents for better precision

**Best practices:**
- Use `pow(2.0)` not `pow(2)` for squaring (small error)
- Prefer integer exponents when possible: `powi(3)` vs `pow(3.0)`
- Be aware of floating-point precision with large exponents
- For polynomial features, use Horner's method or vectorized operations

**Example:**
```rust
use tensor::Tensor;

let x = Tensor::build(vec![1.0, 2.0, 3.0], vec![4])?;

// Squared error (MSE)
let target = Tensor::build(vec![1.5, 3.0, 5.0], vec![4])?;
let error = x - target;
let mse = (error.pow(2.0)).sum()?; // Sum of squared errors

// L2 regularization term
let lambda = 0.01;
let reg_term = x.pow(2.0) * lambda; // x^2 * 0.01
let total_loss = mse + reg_term;
```

---

#### `tensor[index] = value`

#### `tensor - other_tensor`

Element-wise subtraction.

**Parameters:**
- `self: Tensor` - Left operand
- `other: Tensor` - Right operand

**Returns:**
- `Result<Tensor, TensorError>` - Result tensor or error if shapes don't match

**Example:**
```rust
let x = Tensor::build(vec![5.0, 6.0], vec![2])?;
let y = Tensor::build(vec![2.0, 1.0], vec![2])?;
let result = (x - y)?;
// result.data() = [3.0, 5.0]
```

#### `tensor * other_tensor`

Element-wise multiplication.

**Parameters:**
- `self: Tensor` - Left operand
- `other: Tensor` - Right operand

**Returns:**
- `Result<Tensor, TensorError>` - Result tensor or error if shapes don't match

**Example:**
```rust
let x = Tensor::build(vec![1.0, 2.0], vec![2])?;
let y = Tensor::build(vec![3.0, 4.0], vec![2])?;
let result = (x * y)?;
// result.data() = [3.0, 8.0]
```

#### `tensor / other_tensor`

Element-wise division.

**Parameters:**
- `self: Tensor` - Left operand
- `other: Tensor` - Right operand

**Returns:**
- `Result<Tensor, TensorError>` - Result tensor or error if shapes don't match

**Example:**
```rust
let x = Tensor::build(vec![4.0, 9.0], vec![2])?;
let y = Tensor::build(vec![2.0, 3.0], vec![2])?;
let result = (x / y)?;
// result.data() = [2.0, 3.0]
```

#### `tensor + scalar`

Adds scalar to all elements.

**Parameters:**
- `self: Tensor` - Tensor operand
- `scalar: f32` - Scalar value

**Returns:**
- `Tensor` - New tensor with scalar added

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0], vec![3])?;
let result = t + 5.0;
// result.data() = [6.0, 7.0, 8.0]
```

#### `tensor - scalar`

Subtracts scalar from all elements.

**Parameters:**
- `self: Tensor` - Tensor operand
- `scalar: f32` - Scalar value

**Returns:**
- `Tensor` - New tensor with scalar subtracted

**Example:**
```rust
let t = Tensor::build(vec![10.0, 15.0, 20.0], vec![3])?;
let result = t - 5.0;
// result.data() = [5.0, 10.0, 15.0]
```

#### `tensor * scalar`

Multiplies all elements by scalar.

**Parameters:**
- `self: Tensor` - Tensor operand
- `scalar: f32` - Scalar value

**Returns:**
- `Tensor` - New tensor with scalar multiplied

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0], vec![3])?;
let result = t * 3.0;
// result.data() = [3.0, 6.0, 9.0]
```

#### `tensor / scalar`

Divides all elements by scalar.

**Parameters:**
- `self: Tensor` - Tensor operand
- `scalar: f32` - Scalar value

**Returns:**
- `Tensor` - New tensor with scalar divided

**Example:**
```rust
let t = Tensor::build(vec![6.0, 9.0, 12.0], vec![3])?;
let result = t / 3.0;
// result.data() = [2.0, 3.0, 4.0]
```

#### `tensor.pow(exp)`

Raises all elements to power.

**Parameters:**
- `self: Tensor` - Tensor operand
- `exp: f32` - Exponent

**Returns:**
- `Tensor` - New tensor with elements raised to power

**Example:**
```rust
let t = Tensor::build(vec![2.0, 3.0], vec![2])?;
let result = t.pow(2.0);
// result.data() = [4.0, 9.0]
```

---

### Indexing

#### `tensor[linear_index]`

Access element by linear index.

**Parameters:**
- `linear_index: usize` - 0-based linear index

**Returns:**
- `&f32` - Reference to element

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0], vec![3])?;
let value = t[1];  // = 2.0
```

#### `tensor[row, col]`

Access 2D tensor element by row and column.

**Parameters:**
- `[usize; 2]` - [row_index, col_index]

**Returns:**
- `&f32` - Reference to element

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
let value = t[[1, 0]];  // = 3.0 (row 1, col 0)
let value2 = t[[0, 1]];  // = 4.0 (row 0, col 1)
```

#### `tensor[d0, d1, d2]`

Access 3D tensor element by three indices.

**Parameters:**
- `[usize; 3]` - [dim0_index, dim1_index, dim2_index]

**Returns:**
- `&f32` - Reference to element

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])?;
let value = t[[1, 1, 0]];  // = 6.0
```

#### `tensor[d0, d1, d2, d3]`

Access 4D tensor element by four indices.

**Parameters:**
- `[usize; 4]` - Four-dimensional indices

**Returns:**
- `&f32` - Reference to element

**Example:**
```rust
let t = Tensor::build(
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    vec![2, 2, 2, 2]
)?;
let value = t[[1, 1, 1, 1]];  // = 8.0
```

#### `tensor[index] = value`

Mutate tensor element by index (mutable).

**Example:**
```rust
let mut t = Tensor::build(vec![1.0, 2.0, 3.0], vec![3])?;
t[1] = 10.0;  // Now t.data() = [1.0, 10.0, 3.0]
```

---

### Slicing

#### `tensor.slice(range)`

Extracts a slice along the first dimension.

**Parameters:**
- `range: Range<usize>` - Range of first dimension indices

**Returns:**
- `Result<Tensor, TensorError>` - Sliced tensor or error if out of bounds

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])?;
let sliced = t.slice(0..2)?;  // First 2 rows
// sliced.shape() = [2, 2]
// sliced.data() = [1.0, 2.0, 3.0, 4.0]
```

---

### Element-wise Mapping

#### `tensor.map(closure)`

Applies a function to each element with access to its position indices.

**Parameters:**
- `closure: F` where `F: Fn(&[usize], f32) -> f32 + Sync + Send`

**Closure receives:**
- `indices: &[usize]` - Multi-dimensional position (e.g., `[1, 0]` for 2D tensor)
- `value: f32` - Current element value at that position

**Returns:**
- `Tensor` - New tensor with transformed values

**Behavior:**
- Element count < 100,000: Serial execution
- Element count >= 100,000: Parallel execution (uses thread pool)

**Example 1: Simple element-wise operation**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0, 4.0], vec![4])?;
let doubled = t.map(|_indices, value| value * 2.0);
// doubled.data() = [2.0, 4.0, 6.0, 8.0]
```

**Example 2: Use position index**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
let scaled = t.map(|indices, value| {
    let row = indices[0];
    let col = indices[1];
    value * (row as f32 + 1.0)  // Scale by row index
});
```

**Example 3: Sum indices and add to value**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])?;
let result = t.map(|indices, value| {
    let idx_sum: usize = indices.iter().sum();
    value + idx_sum as f32  // Add position sum to value
});
```

**Example 4: Combine with formula crate**
```rust
use formula;

let x = Tensor::build(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5])?;
let normal_pdf = x.map(|_indices, x_val| {
    formula::normal_distribution(x_val, 0.0, 1.0)
});
```

**Note:** This is an immutable operation - the original tensor is preserved.

---

### Utility Functions

#### `Tensor::zeros(shape)`

Creates a tensor filled with zeros.

**Parameters:**
- `shape: Vec<usize>` - Dimension sizes

**Returns:**
- `Result<Tensor, TensorError>` - Zero tensor or error

**Example:**
```rust
let zeros = Tensor::zeros(vec![2, 3])?;
// zeros.data() = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

#### `Tensor::arange(n)`

Creates a 1D tensor with values 0 to n-1.

**Parameters:**
- `n: usize` - Number of elements

**Returns:**
- `Tensor` - Tensor with [0.0, 1.0, 2.0, ..., n-1]

**Example:**
```rust
let t = Tensor::arange(5)?;
// t.data() = [0.0, 1.0, 2.0, 3.0, 4.0]
```

#### `Tensor::rand(shape)`

Creates tensor with uniform random values [0, 1).

**Parameters:**
- `shape: Vec<usize>` - Dimension sizes

**Returns:**
- `Result<Tensor, TensorError>` - Random tensor or error

**Example:**
```rust
let random = Tensor::rand(vec![2, 3])?;
// Each value is in [0.0, 1.0]
```

#### `Tensor::randn(shape)`

Creates tensor with standard normal distribution (μ=0, σ=1).

**Parameters:**
- `shape: Vec<usize>` - Dimension sizes

**Returns:**
- `Result<Tensor, TensorError>` - Normal random tensor or error

**Example:**
```rust
let normal = Tensor::randn(vec![2, 3])?;
// Values follow N(0, 1) distribution
```

#### `tensor.sum()`

Returns sum of all elements.

**Returns:**
- `f32` - Sum of all elements

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0], vec![3])?;
let total = t.sum();  // = 6.0
```

#### `tensor.reshape(new_shape)`

Changes tensor shape while preserving data.

**Parameters:**
- `new_shape: Vec<usize>` - New dimension sizes

**Returns:**
- `Result<Tensor, TensorError>` - Reshaped tensor or error if invalid

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
let reshaped = t.reshape(vec![4])?;
// reshaped.shape() = [4]
// reshaped.data() = [1.0, 2.0, 3.0, 4.0] (same data)
```

#### `tensor.display()`

Prints tensor information to console.

**Example:**
```rust
let t = Tensor::build(vec![1.0, 2.0, 3.0], vec![3])?;
t.display();
// Output:
// Tensor(shape=[3]):
// [[1.0, 2.0, 3.0]]
```

---

## Performance Notes

- Tensor operations automatically use parallel execution for >= 100,000 elements
- Mathematical operations use thread pool for large tensors
- `map` function switches between serial and parallel based on element count

---

## Error Types

### `TensorError`

Possible errors:
- `InvalidShape` - Data length doesn't match shape product
- `IncompatibleShapes` - Tensor shapes don't match for binary operation
- `IndexError { dim, max }` - Index out of bounds for a dimension
