#[derive(Debug, Clone)]
pub enum TensorError {
    InvalidShape,
    IncompatibleShapes,
    IndexError { dim: usize, max: usize },
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::InvalidShape => write!(f, "Invalid shape"),
            TensorError::IncompatibleShapes => write!(f, "Incompatible shapes"),
            TensorError::IndexError { dim, max } => {
                write!(f, "Index {} out of bounds (max: {})", dim, max)
            }
        }
    }
}

impl std::error::Error for TensorError {}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub(crate) fn new_internal(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn build(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        if data.len() != shape.iter().product::<usize>() {
            Err(TensorError::InvalidShape)
        } else {
            Ok(Self::new_internal(data, shape))
        }
    }

    pub fn data(&self) -> &Vec<f32> {
        &self.data
    }

    pub fn get_data(&self) -> &Vec<f32> {
        &self.data
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub(crate) fn get_mut_shape(&mut self) -> &mut Vec<usize> {
        &mut self.shape
    }

    pub fn get_data_mut(&mut self) -> &mut Vec<f32> {
        &mut self.data
    }
}
