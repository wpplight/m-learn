use crate::tensor::Tensor;
use crate::tensor::TensorError;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

fn check_shape_compatible(lhs: &Tensor, rhs: &Tensor) -> Result<(), TensorError> {
    if lhs.shape() == rhs.shape() {
        Ok(())
    } else {
        Err(TensorError::IncompatibleShapes)
    }
}

const PARALLEL_THRESHOLD: usize = 100_000;

fn parallel_map_op<F>(data: &[f32], shape: &[usize], op: F) -> Vec<f32>
where
    F: Fn(&[usize], f32) -> f32 + Sync + Send,
{
    let len = data.len();
    let ndim = shape.len();

    if len < PARALLEL_THRESHOLD {
        let mut result = Vec::with_capacity(len);
        let mut indices = vec![0usize; ndim];
        let mut strides = vec![1usize; ndim];

        for i in (0..ndim).rev() {
            if i == ndim - 1 {
                strides[i] = 1;
            } else {
                strides[i] = shape[i + 1..].iter().product::<usize>();
            }
        }

        for linear_idx in 0..len {
            for i in 0..ndim {
                indices[i] = (linear_idx / strides[i]) % shape[i];
            }
            let value = data[linear_idx];
            result.push(op(&indices, value));
        }
        result
    } else {
        use rayon::prelude::*;
        let strides_vec: Vec<usize> = (0..ndim)
            .rev()
            .map(|i| {
                if i == ndim - 1 {
                    1
                } else {
                    shape[i + 1..].iter().product::<usize>()
                }
            })
            .collect();

        data.par_iter()
            .enumerate()
            .map(|(linear_idx, &value)| {
                let mut indices = vec![0usize; ndim];
                for i in 0..ndim {
                    indices[i] = (linear_idx / strides_vec[i]) % shape[i];
                }
                op(&indices, value)
            })
            .collect()
    }
}

/// 分段并行处理两个张量的元素运算
fn parallel_binary_op<F>(lhs_data: &[f32], rhs_data: &[f32], op: F) -> Vec<f32>
where
    F: Fn(f32, f32) -> f32 + Sync + Send,
{
    let len = lhs_data.len();

    if len < PARALLEL_THRESHOLD {
        // 串行处理小数据量
        lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| op(a, b))
            .collect()
    } else {
        // 大数据量使用线程池并行
        use rayon::prelude::*;
        lhs_data
            .par_iter()
            .zip(rhs_data.par_iter())
            .map(|(&a, &b)| op(a, b))
            .collect()
    }
}

/// 分段并行处理张量和标量的运算
fn parallel_scalar_op<F>(data: &[f32], scalar: f32, op: F) -> Vec<f32>
where
    F: Fn(f32, f32) -> f32 + Sync + Send,
{
    let len = data.len();

    if len < PARALLEL_THRESHOLD {
        // 串行处理小数据量
        data.iter().map(|&a| op(a, scalar)).collect()
    } else {
        // 大数据量使用线程池并行
        use rayon::prelude::*;
        data.par_iter().map(|&a| op(a, scalar)).collect()
    }
}

impl Add for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Add for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn add(self, rhs: Self) -> Self::Output {
        check_shape_compatible(self, rhs)?;
        let data = parallel_binary_op(self.data(), rhs.data(), |a, b| a + b);
        Ok(Tensor::build(data, self.shape().clone())?)
    }
}

impl Sub for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Sub for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn sub(self, rhs: Self) -> Self::Output {
        check_shape_compatible(self, rhs)?;
        let data = parallel_binary_op(self.data(), rhs.data(), |a, b| a - b);
        Ok(Tensor::build(data, self.shape().clone())?)
    }
}

impl Mul for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Mul for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn mul(self, rhs: Self) -> Self::Output {
        check_shape_compatible(self, rhs)?;
        let data = parallel_binary_op(self.data(), rhs.data(), |a, b| a * b);
        Ok(Tensor::build(data, self.shape().clone())?)
    }
}

impl Div for Tensor {
    type Output = Result<Tensor, TensorError>;

    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

impl Div for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn div(self, rhs: Self) -> Self::Output {
        check_shape_compatible(self, rhs)?;
        let data = parallel_binary_op(self.data(), rhs.data(), |a, b| a / b);
        Ok(Tensor::build(data, self.shape().clone())?)
    }
}

impl Add<f32> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Self::Output {
        &self + rhs
    }
}

impl Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Self::Output {
        let data = parallel_scalar_op(self.data(), rhs, |a, b| a + b);
        Tensor::build(data, self.shape().clone()).unwrap()
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f32) -> Self::Output {
        &self - rhs
    }
}

impl Sub<f32> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f32) -> Self::Output {
        let data = parallel_scalar_op(self.data(), rhs, |a, b| a - b);
        Tensor::build(data, self.shape().clone()).unwrap()
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Self::Output {
        &self * rhs
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Self::Output {
        let data = parallel_scalar_op(self.data(), rhs, |a, b| a * b);
        Tensor::build(data, self.shape().clone()).unwrap()
    }
}

impl Div<f32> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: f32) -> Self::Output {
        &self / rhs
    }
}

impl Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: f32) -> Self::Output {
        let data = parallel_scalar_op(self.data(), rhs, |a, b| a / b);
        Tensor::build(data, self.shape().clone()).unwrap()
    }
}

impl Tensor {
    pub fn pow(&self, exp: f32) -> Self {
        let data: Vec<f32> = self.data().iter().map(|a| a.powf(exp)).collect();
        Tensor::build(data, self.shape().clone()).unwrap()
    }

    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(&[usize], f32) -> f32 + Sync + Send,
    {
        let data = parallel_map_op(self.data(), self.shape(), f);
        Tensor::build(data, self.shape().clone()).unwrap()
    }

    fn flatten_index(&self, indices: &[usize]) -> usize {
        let shape = self.shape();
        let ndim = shape.len();

        let mut strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = shape[i + 1..].iter().product::<usize>();
        }

        indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| idx * strides[i])
            .sum()
    }

    pub fn slice(&self, range: std::ops::Range<usize>) -> Result<Tensor, TensorError> {
        let shape = self.shape();
        if shape.is_empty() {
            return Err(TensorError::InvalidShape);
        }

        let first_dim = shape[0];
        if range.start >= first_dim || range.end > first_dim {
            return Err(TensorError::IndexError {
                dim: 0,
                max: first_dim - 1,
            });
        }

        let new_first_dim = range.end - range.start;
        let row_size = shape[1..].iter().product::<usize>();
        let start = range.start * row_size;
        let end = range.end * row_size;
        let sub_data: Vec<f32> = self.data()[start..end].to_vec();
        let mut new_shape = shape.clone();
        new_shape[0] = new_first_dim;

        Tensor::build(sub_data, new_shape)
    }
}

impl Index<usize> for Tensor {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data()[index]
    }
}

impl Index<[usize; 1]> for Tensor {
    type Output = f32;

    fn index(&self, indices: [usize; 1]) -> &Self::Output {
        &self.data()[indices[0]]
    }
}

impl Index<[usize; 2]> for Tensor {
    type Output = f32;

    fn index(&self, indices: [usize; 2]) -> &Self::Output {
        let flat = self.flatten_index(&indices);
        &self.data()[flat]
    }
}

impl Index<[usize; 3]> for Tensor {
    type Output = f32;

    fn index(&self, indices: [usize; 3]) -> &Self::Output {
        let flat = self.flatten_index(&indices);
        &self.data()[flat]
    }
}

impl Index<[usize; 4]> for Tensor {
    type Output = f32;

    fn index(&self, indices: [usize; 4]) -> &Self::Output {
        let flat = self.flatten_index(&indices);
        &self.data()[flat]
    }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.get_data_mut()[index]
    }
}

impl IndexMut<[usize; 1]> for Tensor {
    fn index_mut(&mut self, indices: [usize; 1]) -> &mut Self::Output {
        &mut self.get_data_mut()[indices[0]]
    }
}

impl IndexMut<[usize; 2]> for Tensor {
    fn index_mut(&mut self, indices: [usize; 2]) -> &mut Self::Output {
        let flat = self.flatten_index(&indices);
        &mut self.get_data_mut()[flat]
    }
}

impl IndexMut<[usize; 3]> for Tensor {
    fn index_mut(&mut self, indices: [usize; 3]) -> &mut Self::Output {
        let flat = self.flatten_index(&indices);
        &mut self.get_data_mut()[flat]
    }
}

impl IndexMut<[usize; 4]> for Tensor {
    fn index_mut(&mut self, indices: [usize; 4]) -> &mut Self::Output {
        let flat = self.flatten_index(&indices);
        &mut self.get_data_mut()[flat]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::build(data, shape).unwrap()
    }

    #[test]
    fn test_add_tensors() {
        let x = make_tensor(vec![1.0, 2.0, 4.0, 8.0], vec![4]);
        let y = make_tensor(vec![2.0, 2.0, 2.0, 2.0], vec![4]);
        let result = (x + y).unwrap();
        assert_eq!(result.data(), &vec![3.0, 4.0, 6.0, 10.0]);
    }

    #[test]
    fn test_sub_tensors() {
        let x = make_tensor(vec![1.0, 2.0, 4.0, 8.0], vec![4]);
        let y = make_tensor(vec![2.0, 2.0, 2.0, 2.0], vec![4]);
        let result = (x - y).unwrap();
        assert_eq!(result.data(), &vec![-1.0, 0.0, 2.0, 6.0]);
    }

    #[test]
    fn test_mul_tensors() {
        let x = make_tensor(vec![1.0, 2.0, 4.0, 8.0], vec![4]);
        let y = make_tensor(vec![2.0, 2.0, 2.0, 2.0], vec![4]);
        let result = (x * y).unwrap();
        assert_eq!(result.data(), &vec![2.0, 4.0, 8.0, 16.0]);
    }

    #[test]
    fn test_div_tensors() {
        let x = make_tensor(vec![1.0, 2.0, 4.0, 8.0], vec![4]);
        let y = make_tensor(vec![2.0, 2.0, 2.0, 2.0], vec![4]);
        let result = (x / y).unwrap();
        assert_eq!(result.get_data(), &vec![0.5, 1.0, 2.0, 4.0]);
    }

    #[test]
    fn test_pow_tensors() {
        let x = make_tensor(vec![1.0, 2.0, 4.0, 8.0], vec![4]);
        let result = x.pow(2.0);
        assert_eq!(result.get_data(), &vec![1.0, 4.0, 16.0, 64.0]);
    }

    #[test]
    fn test_add_scalar() {
        let x = make_tensor(vec![1.0, 2.0, 4.0, 8.0], vec![4]);
        let result = x + 2.0;
        assert_eq!(result.get_data(), &vec![3.0, 4.0, 6.0, 10.0]);
    }

    #[test]
    fn test_incompatible_shapes() {
        let x = make_tensor(vec![1.0, 2.0], vec![2]);
        let y = make_tensor(vec![1.0, 2.0, 3.0], vec![3]);
        let result = x + y;
        assert!(result.is_err());
    }

    #[test]
    fn test_1d_index() {
        let t = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        assert_eq!(t[0], 1.0);
        assert_eq!(t[1], 2.0);
        assert_eq!(t[2], 3.0);
        assert_eq!(t[3], 4.0);
    }

    #[test]
    fn test_2d_index() {
        let t = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t[[0, 0]], 1.0);
        assert_eq!(t[[0, 1]], 2.0);
        assert_eq!(t[[1, 0]], 3.0);
        assert_eq!(t[[1, 1]], 4.0);
    }

    #[test]
    fn test_3d_index() {
        let t = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        assert_eq!(t[[0, 0, 0]], 1.0);
        assert_eq!(t[[0, 0, 1]], 2.0);
        assert_eq!(t[[0, 1, 0]], 3.0);
        assert_eq!(t[[0, 1, 1]], 4.0);
        assert_eq!(t[[1, 0, 0]], 5.0);
        assert_eq!(t[[1, 0, 1]], 6.0);
        assert_eq!(t[[1, 1, 0]], 7.0);
        assert_eq!(t[[1, 1, 1]], 8.0);
    }

    #[test]
    fn test_4d_index() {
        let data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let t = make_tensor(data, vec![2, 2, 2, 2]);
        assert_eq!(t[[0, 0, 0, 0]], 1.0);
        assert_eq!(t[[0, 0, 0, 1]], 2.0);
        assert_eq!(t[[0, 0, 1, 0]], 3.0);
        assert_eq!(t[[0, 0, 1, 1]], 4.0);
        assert_eq!(t[[1, 1, 1, 0]], 15.0);
        assert_eq!(t[[1, 1, 1, 1]], 16.0);
    }

    #[test]
    fn test_1d_mutable_index() {
        let mut t = make_tensor(vec![1.0, 2.0, 3.0], vec![3]);
        t[1] = 10.0;
        assert_eq!(t[1], 10.0);
    }

    #[test]
    fn test_2d_mutable_index() {
        let mut t = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        t[[1, 0]] = 10.0;
        assert_eq!(t[[1, 0]], 10.0);
    }

    #[test]
    fn test_slice_1d() {
        let t = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let sliced = t.slice(1..4).unwrap();
        assert_eq!(sliced.shape(), &vec![3]);
        assert_eq!(sliced.get_data(), &vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_slice_2d() {
        let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let t = make_tensor(data, vec![3, 4]);
        let sliced = t.slice(0..2).unwrap();
        assert_eq!(sliced.shape(), &vec![2, 4]);
        assert_eq!(
            sliced.get_data(),
            &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn test_slice_out_of_bounds() {
        let t = make_tensor(vec![1.0, 2.0, 3.0], vec![3]);
        let result = t.slice(0..10);
        assert!(result.is_err());
    }

    #[test]
    fn test_map_1d() {
        let t = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let result = t.map(|indices, value| {
            assert_eq!(indices.len(), 1);
            value * 2.0
        });
        assert_eq!(result.get_data(), &vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(result.shape(), &vec![4]);
    }

    #[test]
    fn test_map_2d() {
        let t = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = t.map(|indices, value| {
            assert_eq!(indices.len(), 2);
            value * 3.0
        });
        assert_eq!(result.get_data(), &vec![3.0, 6.0, 9.0, 12.0]);
        assert_eq!(result.shape(), &vec![2, 2]);
    }

    #[test]
    fn test_map_with_index() {
        let t = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = t.map(|indices, value| {
            let sum: usize = indices.iter().sum();
            value + sum as f32
        });
        assert_eq!(result.get_data(), &vec![1.0, 3.0, 4.0, 6.0]);
        assert_eq!(result.shape(), &vec![2, 2]);
    }

    #[test]
    fn test_map_3d() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let t = make_tensor(data, vec![2, 2, 2]);
        let result = t.map(|indices, value| {
            let flat = indices[0] * 4 + indices[1] * 2 + indices[2];
            assert!(flat < 8);
            value * 2.0
        });
        assert_eq!(
            result.data(),
            &vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        );
    }

    #[test]
    fn test_map_parallel_large() {
        let data: Vec<f32> = (0..200_000).map(|i| i as f32).collect();
        let t = make_tensor(data, vec![200_000]);
        let result = t.map(|_indices, value| value * 2.0);
        assert_eq!(result.data().len(), 200_000);
        assert_eq!(result.data()[0], 0.0);
        assert_eq!(result.data()[199_999], 399_998.0);
    }
}
