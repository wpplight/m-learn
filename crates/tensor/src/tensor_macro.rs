use crate::tensor::Tensor;

#[macro_export]
macro_rules! tensor {
    ([$($x:literal),* $(,)*]) => {{
        let data: Vec<f32> = vec![$($x),*];
        let len = data.len();
        Tensor::build(data, vec![len])
    }};

    ([$([$($x:literal),*]),+ $(,)*]) => {{
        let mut data = Vec::new();
        let mut cols = 0;
        $(
            {
                let row: Vec<f32> = vec![$($x),*];
                if cols == 0 {
                    cols = row.len();
                }
                data.extend(row);
            }
        )*
        let rows = data.len() / cols;
        Tensor::build(data, vec![rows, cols])
    }};
}

impl From<Vec<f32>> for Tensor {
    fn from(data: Vec<f32>) -> Self {
        let len = data.len();
        Tensor::new_internal(data, vec![len])
    }
}

impl From<Vec<Vec<f32>>> for Tensor {
    fn from(data: Vec<Vec<f32>>) -> Self {
        let rows = data.len();
        let cols = data.get(0).map_or(0, |v| v.len());
        let flat: Vec<f32> = data.into_iter().flatten().collect();
        Tensor::new_internal(flat, vec![rows, cols])
    }
}

impl From<Vec<Vec<Vec<f32>>>> for Tensor {
    fn from(data: Vec<Vec<Vec<f32>>>) -> Self {
        let depth = data.len();
        let rows = data.get(0).map_or(0, |v| v.len());
        let cols = data.get(0).and_then(|v| v.get(0)).map_or(0, |v| v.len());
        let flat: Vec<f32> = data.into_iter().flatten().flatten().collect();
        Tensor::new_internal(flat, vec![depth, rows, cols])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_1d() {
        let t = tensor!([1.0, 2.0, 3.0]).unwrap();
        assert_eq!(t.shape(), &vec![3]);
        assert_eq!(t.get_data(), &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_macro_2d() {
        let t = tensor!([[1.0, 2.0], [3.0, 4.0]]).unwrap();
        assert_eq!(t.shape(), &vec![2, 2]);
        assert_eq!(t.get_data(), &vec![1.0, 2.0, 3.0, 4.0]);
    }
}
