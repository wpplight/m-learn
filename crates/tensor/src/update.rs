use crate::tensor::Tensor;
use crate::tensor::TensorError;

impl Tensor {
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), TensorError> {
        let new_size: usize = new_shape.iter().product();
        let data_len = self.data().len();
        let shape: &mut Vec<usize> = self.get_mut_shape();
        if new_size != data_len {
            return Err(TensorError::InvalidShape);
        }
        *shape = new_shape.clone();
        Ok(())
    }
}
