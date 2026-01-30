use crate::tensor::Tensor;
use crate::tensor::TensorError;
use rand::Rng;

impl Tensor {
    pub fn new(shape: Vec<usize>, fill: f32) -> Result<Self, TensorError> {
        if shape.is_empty() {
            Err(TensorError::InvalidShape)
        } else {
            let size = shape.iter().product();
            let data = vec![fill; size];
            Ok(Self::new_internal(data, shape))
        }
    }

    pub fn ones(shape: Vec<usize>) -> Result<Self, TensorError> {
        if shape.is_empty() {
            Err(TensorError::InvalidShape)
        } else {
            let size = shape.iter().product();
            let data = vec![1.0; size];
            Ok(Self::new_internal(data, shape))
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        if shape.is_empty() {
            Err(TensorError::InvalidShape)
        } else {
            let size = shape.iter().product();
            let data = vec![0.0; size];
            Ok(Self::new_internal(data, shape))
        }
    }

    pub fn arange(len: usize) -> Result<Self, TensorError> {
        if len == 0 {
            Err(TensorError::InvalidShape)
        } else {
            let data = (0..len as usize).map(|i| i as f32).collect();
            Ok(Self::new_internal(data, vec![len]))
        }
    }

    /// 生成 [0, 1) 范围的均匀分布随机张量
    pub fn rand(shape: Vec<usize>) -> Result<Self, TensorError> {
        let mut rng = rand::rng();
        if shape.is_empty() {
            Err(TensorError::InvalidShape)
        } else {
            let size = shape.iter().product();
            let data = (0..size).map(|_| rng.random::<f32>()).collect();
            Ok(Self::new_internal(data, shape))
        }
    }
    
    //生成在范围内固定步长的向量
    pub fn range(begin: f32, end: f32, step: f32) -> Result<Self, TensorError> {
        if step == 0.0 {
            Err(TensorError::InvalidShape)
        } else {
            let len = ((end - begin) / step).ceil() as usize;
            let data = (0..len).map(|i| begin + i as f32 * step).collect();
            Ok(Self::new_internal(data, vec![len]))
        }
    }

    /// 生成标准正态分布（均值0，标准差1）的随机张量
    /// 使用 Box-Muller 变换生成
    pub fn randn(shape: Vec<usize>) -> Result<Self, TensorError> {
        let mut rng = rand::rng();
        if shape.is_empty() {
            Err(TensorError::InvalidShape)
        } else {
            let size = shape.iter().product();
            let data: Vec<f32> = (0..size)
                .map(|_| {
                    let u1: f32 = rng.random();
                    let u2: f32 = rng.random();
                    let radius = (-2.0 * u1.ln()).sqrt();
                    let theta = 2.0 * std::f32::consts::PI * u2;
                    radius * theta.cos()
                })
                .collect();
            Ok(Self::new_internal(data, shape))
        }
    }
}
