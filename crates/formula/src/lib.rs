use std::f32::consts::PI;

pub fn normal_distribution(x: f32, mu: f32, sigma: f32) -> f32 {
    let sqrt_2pi = (2.0 * PI).sqrt();
    let coefficient = 1.0 / (sigma * sqrt_2pi);
    let exponent = -0.5 * ((x - mu) / sigma).powi(2);
    coefficient * exponent.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_distribution_standard() {
        let result = normal_distribution(0.0, 0.0, 1.0);
        assert_eq!(result, 1.0 / (2.0 * PI).sqrt());
    }

    #[test]
    fn test_normal_distribution_peak() {
        let result = normal_distribution(1.0, 1.0, 1.0);
        let expected = 1.0 / (1.0 * (2.0 * PI).sqrt());
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_normal_distribution_symmetric() {
        let x = 0.5;
        let mu = 0.0;
        let sigma = 1.0;
        let y1 = normal_distribution(x, mu, sigma);
        let y2 = normal_distribution(-x + 2.0 * mu, mu, sigma);
        assert!((y1 - y2).abs() < 1e-6);
    }

    #[test]
    fn test_normal_distribution_different_sigma() {
        let sigma = 0.5;
        let result = normal_distribution(0.0, 0.0, sigma);
        let expected = 1.0 / (sigma * (2.0 * PI).sqrt());
        assert!((result - expected).abs() < 1e-6);
    }
}
