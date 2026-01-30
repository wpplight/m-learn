use minifb::{Key, Window, WindowOptions};
use plotters::prelude::*;
use plotters::style::{register_font, FontStyle};
use std::sync::Once;
use tensor::Tensor;

static FONT_REGISTER: Once = Once::new();

fn ensure_font() {
    FONT_REGISTER.call_once(|| {
        let _ = register_font("sans-serif", FontStyle::Normal, dejavu::sans::regular());
        let _ = register_font("sans-serif", FontStyle::Bold, dejavu::sans::bold());
    });
}

fn rgb_to_argb_u32(rgb: &[u8], out: &mut [u32]) {
    assert!(rgb.len() >= out.len() * 3);
    for (i, pixel) in out.iter_mut().enumerate() {
        let r = rgb[i * 3] as u32;
        let g = rgb[i * 3 + 1] as u32;
        let b = rgb[i * 3 + 2] as u32;
        *pixel = 0xff_00_00_00 | (r << 16) | (g << 8) | b;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageType {
    Svg,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportConfig {
    pub filepath: String,
    pub image_type: ImageType,
}

pub struct PlotConfig {
    pub title: String,
    pub xlabel: String,
    pub ylabel: String,
    pub width: u32,
    pub height: u32,
    pub legends: Vec<String>,
    pub output_path: String,
    pub x_range: Option<(f32, f32)>,
    pub y_range: Option<(f32, f32)>,
    pub x_ticks: Option<usize>,
    pub y_ticks: Option<usize>,
    pub svg: bool,
    pub show_window: bool,
    pub export_config: Option<ExportConfig>,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            title: "Plot".to_string(),
            xlabel: "X".to_string(),
            ylabel: "Y".to_string(),
            width: 800,
            height: 600,
            legends: vec!["Series".to_string()],
            output_path: "plot.png".to_string(),
            x_range: None,
            y_range: None,
            x_ticks: None,
            y_ticks: None,
            svg: false,
            show_window: true,
            export_config: None,
        }
    }
}

impl PlotConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    pub fn xlabel(mut self, xlabel: impl Into<String>) -> Self {
        self.xlabel = xlabel.into();
        self
    }

    pub fn ylabel(mut self, ylabel: impl Into<String>) -> Self {
        self.ylabel = ylabel.into();
        self
    }

    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn legends(mut self, legends: Vec<String>) -> Self {
        self.legends = legends;
        self
    }

    pub fn output_path(mut self, path: impl Into<String>) -> Self {
        self.output_path = path.into();
        self
    }

    pub fn x_range(mut self, min: f32, max: f32) -> Self {
        self.x_range = Some((min, max));
        self
    }

    pub fn y_range(mut self, min: f32, max: f32) -> Self {
        self.y_range = Some((min, max));
        self
    }

    pub fn x_ticks(mut self, count: usize) -> Self {
        self.x_ticks = Some(count);
        self
    }

    pub fn y_ticks(mut self, count: usize) -> Self {
        self.y_ticks = Some(count);
        self
    }

    pub fn svg(mut self, enable: bool) -> Self {
        self.svg = enable;
        self
    }

    pub fn show_window(mut self, show: bool) -> Self {
        self.show_window = show;
        self
    }

    pub fn export(mut self, filepath: impl Into<String>, image_type: ImageType) -> Self {
        self.export_config = Some(ExportConfig {
            filepath: filepath.into(),
            image_type,
        });
        self
    }
}

pub fn plot(config: &PlotConfig, x: &Tensor, y: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let x_data = x.data();
    let y_data = y.data();

    if x_data.len() != y_data.len() {
        return Err("X and Y tensors must have same length".into());
    }

    let x_range = config.x_range.unwrap_or_else(|| {
        (
            x_data.iter().cloned().fold(f32::INFINITY, f32::min),
            x_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        )
    });
    let y_range = config.y_range.unwrap_or_else(|| {
        (
            y_data.iter().cloned().fold(f32::INFINITY, f32::min),
            y_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        )
    });

    if config.show_window {
        ensure_font();
        let (w, h) = (config.width as usize, config.height as usize);
        let mut rgb_buf = vec![0u8; w * h * 3];
        {
            let root = BitMapBackend::with_buffer(&mut rgb_buf, (config.width, config.height))
                .into_drawing_area();
            root.fill(&WHITE)?;

            let mut chart = ChartBuilder::on(&root)
                .margin(10)
                .caption(&config.title, ("sans-serif", 20))
                .x_label_area_size(40)
                .y_label_area_size(60)
                .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

            let mut mesh = chart.configure_mesh();
            mesh.x_desc(&config.xlabel).y_desc(&config.ylabel);
            if let Some(n) = config.x_ticks {
                mesh.x_labels(n);
            }
            if let Some(n) = config.y_ticks {
                mesh.y_labels(n);
            }
            mesh.draw()?;

            let label = config
                .legends
                .first()
                .cloned()
                .unwrap_or_else(|| "Series".to_string());
            chart
                .draw_series(LineSeries::new(
                    x_data.iter().zip(y_data.iter()).map(|(&x, &y)| (x, y)),
                    &BLUE,
                ))?
                .label(label)
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

            chart
                .configure_series_labels()
                .background_style(&WHITE)
                .border_style(&BLACK)
                .draw()?;

            root.present()?;
        }

        let mut argb_buf = vec![0u32; w * h];
        rgb_to_argb_u32(&rgb_buf, &mut argb_buf);

        let mut window = Window::new(&config.title, w, h, WindowOptions::default())
            .map_err(|e| format!("minifb window: {}", e))?;
        while window.is_open() && !window.is_key_down(Key::Escape) {
            window
                .update_with_buffer(&argb_buf, w, h)
                .map_err(|e| format!("minifb update: {}", e))?;
        }
    }

    if let Some(ref export_config) = config.export_config {
        ensure_font();
        let root = match export_config.image_type {
            ImageType::Svg => {
                SVGBackend::new(&export_config.filepath, (config.width, config.height))
                    .into_drawing_area()
            }
        };

        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .caption(&config.title, ("sans-serif", 20))
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        let mut mesh = chart.configure_mesh();
        mesh.x_desc(&config.xlabel).y_desc(&config.ylabel);
        if let Some(n) = config.x_ticks {
            mesh.x_labels(n);
        }
        if let Some(n) = config.y_ticks {
            mesh.y_labels(n);
        }
        mesh.draw()?;

        let label = config
            .legends
            .first()
            .cloned()
            .unwrap_or_else(|| "Series".to_string());
        chart
            .draw_series(LineSeries::new(
                x_data.iter().zip(y_data.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))?
            .label(label)
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

        chart
            .configure_series_labels()
            .background_style(&WHITE)
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
    }

    Ok(())
}

pub fn plot_series(
    config: &PlotConfig,
    x: &Tensor,
    y_series: Vec<&Tensor>,
) -> Result<(), Box<dyn std::error::Error>> {
    let x_data = x.data();
    let y_data_vec: Vec<&Vec<f32>> = y_series.iter().map(|y| y.data()).collect();

    for y_data in &y_data_vec {
        if x_data.len() != y_data.len() {
            return Err("X and all Y tensors must have same length".into());
        }
    }

    let y_range = config.y_range.unwrap_or_else(|| {
        (
            y_data_vec
                .iter()
                .flat_map(|y| y.iter())
                .cloned()
                .fold(f32::INFINITY, f32::min),
            y_data_vec
                .iter()
                .flat_map(|y| y.iter())
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max),
        )
    });
    let x_range = config.x_range.unwrap_or_else(|| {
        (
            x_data.iter().cloned().fold(f32::INFINITY, f32::min),
            x_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        )
    });

    let colors = [BLUE, RED, GREEN, CYAN, MAGENTA, YELLOW];

    if config.show_window {
        ensure_font();
        let (w, h) = (config.width as usize, config.height as usize);
        let mut rgb_buf = vec![0u8; w * h * 3];
        {
            let root = BitMapBackend::with_buffer(&mut rgb_buf, (config.width, config.height))
                .into_drawing_area();
            root.fill(&WHITE)?;

            let mut chart = ChartBuilder::on(&root)
                .margin(10)
                .caption(&config.title, ("sans-serif", 20))
                .x_label_area_size(40)
                .y_label_area_size(60)
                .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

            let mut mesh = chart.configure_mesh();
            mesh.x_desc(&config.xlabel).y_desc(&config.ylabel);
            if let Some(n) = config.x_ticks {
                mesh.x_labels(n);
            }
            if let Some(n) = config.y_ticks {
                mesh.y_labels(n);
            }
            mesh.draw()?;

            for (idx, y_data) in y_data_vec.iter().enumerate() {
                let color = colors[idx % colors.len()];
                let label = config
                    .legends
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("Series {}", idx + 1));
                chart
                    .draw_series(LineSeries::new(
                        x_data.iter().zip(y_data.iter()).map(|(&x, &y)| (x, y)),
                        color,
                    ))?
                    .label(label)
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
            }

            chart
                .configure_series_labels()
                .background_style(&WHITE)
                .border_style(&BLACK)
                .draw()?;

            root.present()?;
        }

        let mut argb_buf = vec![0u32; w * h];
        rgb_to_argb_u32(&rgb_buf, &mut argb_buf);

        let mut window = Window::new(&config.title, w, h, WindowOptions::default())
            .map_err(|e| format!("minifb window: {}", e))?;
        while window.is_open() && !window.is_key_down(Key::Escape) {
            window
                .update_with_buffer(&argb_buf, w, h)
                .map_err(|e| format!("minifb update: {}", e))?;
        }
    }

    if let Some(ref export_config) = config.export_config {
        ensure_font();
        let root = match export_config.image_type {
            ImageType::Svg => {
                SVGBackend::new(&export_config.filepath, (config.width, config.height))
                    .into_drawing_area()
            }
        };

        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .caption(&config.title, ("sans-serif", 20))
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        let mut mesh = chart.configure_mesh();
        mesh.x_desc(&config.xlabel).y_desc(&config.ylabel);
        if let Some(n) = config.x_ticks {
            mesh.x_labels(n);
        }
        if let Some(n) = config.y_ticks {
            mesh.y_labels(n);
        }
        mesh.draw()?;

        for (idx, y_data) in y_data_vec.iter().enumerate() {
            let color = colors[idx % colors.len()];
            let label = config
                .legends
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("Series {}", idx + 1));
            chart
                .draw_series(LineSeries::new(
                    x_data.iter().zip(y_data.iter()).map(|(&x, &y)| (x, y)),
                    color,
                ))?
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
        }

        chart
            .configure_series_labels()
            .background_style(&WHITE)
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
    }

    Ok(())
}

pub fn plot_pairs(
    config: &PlotConfig,
    x_series: Vec<&Tensor>,
    y_series: Vec<&Tensor>,
) -> Result<(), Box<dyn std::error::Error>> {
    if x_series.len() != y_series.len() {
        return Err("X series and Y series must have the same number of elements".into());
    }

    if x_series.is_empty() {
        return Err("At least one (x, y) pair is required".into());
    }

    let x_data_vec: Vec<&Vec<f32>> = x_series.iter().map(|x| x.data()).collect();
    let y_data_vec: Vec<&Vec<f32>> = y_series.iter().map(|y| y.data()).collect();

    for (idx, (x_data, y_data)) in x_data_vec.iter().zip(y_data_vec.iter()).enumerate() {
        if x_data.len() != y_data.len() {
            return Err(format!(
                "X and Y at index {} must have same length (X: {}, Y: {})",
                idx,
                x_data.len(),
                y_data.len()
            )
            .into());
        }
    }

    let x_range = config.x_range.unwrap_or_else(|| {
        (
            x_data_vec
                .iter()
                .flat_map(|x| x.iter())
                .cloned()
                .fold(f32::INFINITY, f32::min),
            x_data_vec
                .iter()
                .flat_map(|x| x.iter())
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max),
        )
    });
    let y_range = config.y_range.unwrap_or_else(|| {
        (
            y_data_vec
                .iter()
                .flat_map(|y| y.iter())
                .cloned()
                .fold(f32::INFINITY, f32::min),
            y_data_vec
                .iter()
                .flat_map(|y| y.iter())
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max),
        )
    });

    let colors = [BLUE, RED, GREEN, CYAN, MAGENTA, YELLOW];

    if config.show_window {
        ensure_font();
        let (w, h) = (config.width as usize, config.height as usize);
        let mut rgb_buf = vec![0u8; w * h * 3];
        {
            let root = BitMapBackend::with_buffer(&mut rgb_buf, (config.width, config.height))
                .into_drawing_area();
            root.fill(&WHITE)?;

            let mut chart = ChartBuilder::on(&root)
                .margin(10)
                .caption(&config.title, ("sans-serif", 20))
                .x_label_area_size(40)
                .y_label_area_size(60)
                .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

            let mut mesh = chart.configure_mesh();
            mesh.x_desc(&config.xlabel).y_desc(&config.ylabel);
            if let Some(n) = config.x_ticks {
                mesh.x_labels(n);
            }
            if let Some(n) = config.y_ticks {
                mesh.y_labels(n);
            }
            mesh.draw()?;

            for (idx, (x_data, y_data)) in x_data_vec.iter().zip(y_data_vec.iter()).enumerate() {
                let color = colors[idx % colors.len()];
                let label = config
                    .legends
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("Pair {}", idx + 1));
                chart
                    .draw_series(LineSeries::new(
                        x_data.iter().zip(y_data.iter()).map(|(&x, &y)| (x, y)),
                        color,
                    ))?
                    .label(label)
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
            }

            chart
                .configure_series_labels()
                .background_style(&WHITE)
                .border_style(&BLACK)
                .draw()?;

            root.present()?;
        }

        let mut argb_buf = vec![0u32; w * h];
        rgb_to_argb_u32(&rgb_buf, &mut argb_buf);

        let mut window = Window::new(&config.title, w, h, WindowOptions::default())
            .map_err(|e| format!("minifb window: {}", e))?;
        while window.is_open() && !window.is_key_down(Key::Escape) {
            window
                .update_with_buffer(&argb_buf, w, h)
                .map_err(|e| format!("minifb update: {}", e))?;
        }
    }

    if let Some(ref export_config) = config.export_config {
        ensure_font();
        let root = match export_config.image_type {
            ImageType::Svg => {
                SVGBackend::new(&export_config.filepath, (config.width, config.height))
                    .into_drawing_area()
            }
        };

        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .caption(&config.title, ("sans-serif", 20))
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        let mut mesh = chart.configure_mesh();
        mesh.x_desc(&config.xlabel).y_desc(&config.ylabel);
        if let Some(n) = config.x_ticks {
            mesh.x_labels(n);
        }
        if let Some(n) = config.y_ticks {
            mesh.y_labels(n);
        }
        mesh.draw()?;

        for (idx, (x_data, y_data)) in x_data_vec.iter().zip(y_data_vec.iter()).enumerate() {
            let color = colors[idx % colors.len()];
            let label = config
                .legends
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("Pair {}", idx + 1));
            chart
                .draw_series(LineSeries::new(
                    x_data.iter().zip(y_data.iter()).map(|(&x, &y)| (x, y)),
                    color,
                ))?
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
        }

        chart
            .configure_series_labels()
            .background_style(&WHITE)
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
    }

    Ok(())
}

pub fn export(
    config: &PlotConfig,
    x: &Tensor,
    y: &Tensor,
    filepath: &str,
    imagetype: ImageType,
) -> Result<(), Box<dyn std::error::Error>> {
    let x_data = x.data();
    let y_data = y.data();

    if x_data.len() != y_data.len() {
        return Err("X and Y tensors must have same length".into());
    }

    let root = match imagetype {
        ImageType::Svg => {
            SVGBackend::new(filepath, (config.width, config.height)).into_drawing_area()
        }
    };
    root.fill(&WHITE)?;

    let x_range = config.x_range.unwrap_or_else(|| {
        (
            x_data.iter().cloned().fold(f32::INFINITY, f32::min),
            x_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        )
    });
    let y_range = config.y_range.unwrap_or_else(|| {
        (
            y_data.iter().cloned().fold(f32::INFINITY, f32::min),
            y_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        )
    });

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(&config.title, ("sans-serif", 20))
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

    let mut mesh = chart.configure_mesh();
    mesh.x_desc(&config.xlabel).y_desc(&config.ylabel);
    if let Some(n) = config.x_ticks {
        mesh.x_labels(n);
    }
    if let Some(n) = config.y_ticks {
        mesh.y_labels(n);
    }
    mesh.draw()?;

    let label = config
        .legends
        .first()
        .cloned()
        .unwrap_or_else(|| "Series".to_string());
    chart
        .draw_series(LineSeries::new(
            x_data.iter().zip(y_data.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?
        .label(label)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE)
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

pub fn export_series(
    config: &PlotConfig,
    x: &Tensor,
    y_series: Vec<&Tensor>,
    filepath: &str,
    imagetype: ImageType,
) -> Result<(), Box<dyn std::error::Error>> {
    let x_data = x.data();
    let y_data_vec: Vec<&Vec<f32>> = y_series.iter().map(|y| y.data()).collect();

    for y_data in &y_data_vec {
        if x_data.len() != y_data.len() {
            return Err("X and all Y tensors must have same length".into());
        }
    }

    let root = match imagetype {
        ImageType::Svg => {
            SVGBackend::new(filepath, (config.width, config.height)).into_drawing_area()
        }
    };
    root.fill(&WHITE)?;

    let y_range = config.y_range.unwrap_or_else(|| {
        (
            y_data_vec
                .iter()
                .flat_map(|y| y.iter())
                .cloned()
                .fold(f32::INFINITY, f32::min),
            y_data_vec
                .iter()
                .flat_map(|y| y.iter())
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max),
        )
    });
    let x_range = config.x_range.unwrap_or_else(|| {
        (
            x_data.iter().cloned().fold(f32::INFINITY, f32::min),
            x_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        )
    });

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(&config.title, ("sans-serif", 20))
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

    let mut mesh = chart.configure_mesh();
    mesh.x_desc(&config.xlabel).y_desc(&config.ylabel);
    if let Some(n) = config.x_ticks {
        mesh.x_labels(n);
    }
    if let Some(n) = config.y_ticks {
        mesh.y_labels(n);
    }
    mesh.draw()?;

    let colors = [BLUE, RED, GREEN, CYAN, MAGENTA, YELLOW];
    for (idx, y_data) in y_data_vec.iter().enumerate() {
        let color = colors[idx % colors.len()];
        let label = config
            .legends
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("Series {}", idx + 1));
        chart
            .draw_series(LineSeries::new(
                x_data.iter().zip(y_data.iter()).map(|(&x, &y)| (x, y)),
                color,
            ))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE)
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

pub fn export_series_array(
    config: &PlotConfig,
    x: &[f32],
    y_series: Vec<&[f32]>,
    filepath: &str,
    imagetype: ImageType,
) -> Result<(), Box<dyn std::error::Error>> {
    for y in &y_series {
        if x.len() != y.len() {
            return Err("X and all Y arrays must have same length".into());
        }
    }

    let root = match imagetype {
        ImageType::Svg => {
            SVGBackend::new(filepath, (config.width, config.height)).into_drawing_area()
        }
    };
    root.fill(&WHITE)?;

    let y_range = config.y_range.unwrap_or_else(|| {
        (
            y_series
                .iter()
                .flat_map(|y| y.iter())
                .cloned()
                .fold(f32::INFINITY, f32::min),
            y_series
                .iter()
                .flat_map(|y| y.iter())
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max),
        )
    });
    let x_range = config.x_range.unwrap_or_else(|| {
        (
            x.iter().cloned().fold(f32::INFINITY, f32::min),
            x.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        )
    });

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(&config.title, ("sans-serif", 20))
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

    let mut mesh = chart.configure_mesh();
    mesh.x_desc(&config.xlabel).y_desc(&config.ylabel);
    if let Some(n) = config.x_ticks {
        mesh.x_labels(n);
    }
    if let Some(n) = config.y_ticks {
        mesh.y_labels(n);
    }
    mesh.draw()?;

    let colors = [BLUE, RED, GREEN, CYAN, MAGENTA, YELLOW];
    for (idx, y_data) in y_series.iter().enumerate() {
        let color = colors[idx % colors.len()];
        let label = config
            .legends
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("Series {}", idx + 1));
        chart
            .draw_series(LineSeries::new(
                x.iter().zip(y_data.iter()).map(|(&x, &y)| (x, y)),
                color,
            ))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE)
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

pub fn plot_array(
    config: &PlotConfig,
    x: &[f32],
    y: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    if x.len() != y.len() {
        return Err("X and Y arrays must have same length".into());
    }

    if let Some(ref export_config) = config.export_config {
        ensure_font();
        let root = match export_config.image_type {
            ImageType::Svg => {
                SVGBackend::new(&export_config.filepath, (config.width, config.height))
                    .into_drawing_area()
            }
        };

        root.fill(&WHITE)?;

        let x_range = config.x_range.unwrap_or_else(|| {
            (
                x.iter().cloned().fold(f32::INFINITY, f32::min),
                x.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            )
        });
        let y_range = config.y_range.unwrap_or_else(|| {
            (
                y.iter().cloned().fold(f32::INFINITY, f32::min),
                y.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            )
        });

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .caption(&config.title, ("sans-serif", 20))
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        let mut mesh = chart.configure_mesh();
        mesh.x_desc(&config.xlabel).y_desc(&config.ylabel);
        if let Some(n) = config.x_ticks {
            mesh.x_labels(n);
        }
        if let Some(n) = config.y_ticks {
            mesh.y_labels(n);
        }
        mesh.draw()?;

        let label = config
            .legends
            .first()
            .cloned()
            .unwrap_or_else(|| "Series".to_string());
        chart
            .draw_series(LineSeries::new(
                x.iter().zip(y.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))?
            .label(label)
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

        chart
            .configure_series_labels()
            .background_style(&WHITE)
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
    }

    Ok(())
}

pub fn plot_series_array(
    config: &PlotConfig,
    x: &[f32],
    y_series: Vec<&[f32]>,
) -> Result<(), Box<dyn std::error::Error>> {
    for y in &y_series {
        if x.len() != y.len() {
            return Err("X and all Y arrays must have same length".into());
        }
    }

    if let Some(ref export_config) = config.export_config {
        ensure_font();
        let root = match export_config.image_type {
            ImageType::Svg => {
                SVGBackend::new(&export_config.filepath, (config.width, config.height))
                    .into_drawing_area()
            }
        };

        root.fill(&WHITE)?;

        let y_range = config.y_range.unwrap_or_else(|| {
            (
                y_series
                    .iter()
                    .flat_map(|y| y.iter())
                    .cloned()
                    .fold(f32::INFINITY, f32::min),
                y_series
                    .iter()
                    .flat_map(|y| y.iter())
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max),
            )
        });
        let x_range = config.x_range.unwrap_or_else(|| {
            (
                x.iter().cloned().fold(f32::INFINITY, f32::min),
                x.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            )
        });

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .caption(&config.title, ("sans-serif", 20))
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        let mut mesh = chart.configure_mesh();
        mesh.x_desc(&config.xlabel).y_desc(&config.ylabel);
        if let Some(n) = config.x_ticks {
            mesh.x_labels(n);
        }
        if let Some(n) = config.y_ticks {
            mesh.y_labels(n);
        }
        mesh.draw()?;

        let colors = [BLUE, RED, GREEN, CYAN, MAGENTA, YELLOW];
        for (idx, y_data) in y_series.iter().enumerate() {
            let color = colors[idx % colors.len()];
            let label = config
                .legends
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("Series {}", idx + 1));
            chart
                .draw_series(LineSeries::new(
                    x.iter().zip(y_data.iter()).map(|(&x, &y)| (x, y)),
                    color,
                ))?
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
        }

        chart
            .configure_series_labels()
            .background_style(&WHITE)
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
    }

    Ok(())
}
