# Draw Crate Documentation

Draw crate provides plotting capabilities for tensors with support for SVG export and window display.

## Overview

- **Plot tensors**: Line charts for tensor data
- **Single series**: `plot()`, `export()`
- **Multiple series**: `plot_series()`, `export_series()`
- **Multiple (x, y) pairs**: `plot_pairs()`, `export_series_array()`
- **Configurable**: Titles, labels, ranges, ticks, colors
- **Export formats**: SVG (scalable vector graphics)
- **Window display**: Interactive preview (minifb)

---

## API Reference

### Configuration

#### `PlotConfig`

Configuration structure for plot appearance and behavior.

**Builder methods:**
- `new()` - Creates default configuration
- `title(String)` - Set plot title
- `xlabel(String)` - Set x-axis label
- `ylabel(String)` - Set y-axis label
- `size(width, height)` - Set canvas dimensions (default: 800x600)
- `legends(Vec<String>)` - Set series legend labels
- `output_path(String)` - Set output file path (default: "plot.png")
- `x_range(min, max)` - Set x-axis range
- `y_range(min, max)` - Set y-axis range
- `x_ticks(count)` - Set number of x-axis ticks
- `y_ticks(count)` - Set number of y-axis ticks
- `show_window(bool)` - Enable/disable window display (default: true)
- `export(filepath, ImageType)` - Configure export to file

**Example:**
```rust
use draw::{PlotConfig, ImageType};

let config = PlotConfig::new()
    .title("My Data")
    .xlabel("X Axis")
    .ylabel("Y Axis")
    .size(1000, 600)
    .x_range(0.0, 10.0)
    .y_range(0.0, 100.0)
    .legends(vec!["Series A".to_string(), "Series B".to_string()])
    .show_window(false)
    .export("output/my_plot.svg", ImageType::Svg);
```

---

### Basic Plotting

#### `plot(config, x, y)`

Plot a single (x, y) series.

**Parameters:**
- `config: &PlotConfig` - Plot configuration
- `x: &Tensor` - X-axis values (1D tensor)
- `y: &Tensor` - Y-axis values (1D tensor)

**Returns:**
- `Result<(), Box<dyn std::error::Error>>`

**Behavior:**
- Shows in window if `config.show_window == true`
- Exports to file if `config.export_config` is set

**Example:**
```rust
use draw::{plot, PlotConfig, ImageType};
use tensor::{tensor, Tensor};

let x = tensor!([0.0, 1.0, 2.0, 3.0, 4.0]);
let y = tensor!([1.0, 2.0, 4.0, 8.0, 16.0]);

let config = PlotConfig::new()
    .title("Linear Function")
    .xlabel("X")
    .ylabel("Y")
    .export("output/linear.svg", ImageType::Svg)
    .show_window(false);

plot(&config, &x, &y)?;
// Creates output/linear.svg
```

---

### Multiple Series

#### `plot_series(config, x, y_series)`

Plot multiple Y series sharing the same X values.

**Parameters:**
- `config: &PlotConfig` - Plot configuration
- `x: &Tensor` - X-axis values (1D tensor)
- `y_series: Vec<&Tensor>` - Multiple Y tensors (all same length as x)

**Returns:**
- `Result<(), Box<dyn std::error::Error>>`

**Features:**
- Each series uses different color (blue, red, green, cyan, magenta, yellow)
- Automatically shows legend with configured labels
- Color cycle repeats for more than 6 series

**Example:**
```rust
use draw::{plot_series, PlotConfig, ImageType};
use tensor::{tensor, Tensor};

let x = tensor!([0.0, 1.0, 2.0, 3.0, 4.0]);
let y1 = tensor!([1.0, 2.0, 3.0, 4.0, 5.0]);
let y2 = tensor!([2.0, 4.0, 6.0, 8.0, 10.0]);
let y3 = tensor!([0.5, 1.0, 1.5, 2.0, 2.5]);

let config = PlotConfig::new()
    .title("Multiple Functions")
    .legends(vec![
        "f(x) = x".to_string(),
        "f(x) = 2x".to_string(),
        "f(x) = 0.5x".to_string(),
    ])
    .export("output/multi_series.svg", ImageType::Svg)
    .show_window(false);

plot_series(&config, &x, vec![&y1, &y2, &y3])?;
```

---

### Multiple (X, Y) Pairs

#### `plot_pairs(config, x_series, y_series)`

Plot multiple independent (x, y) pairs, each with its own x values.

**Parameters:**
- `config: &PlotConfig` - Plot configuration
- `x_series: Vec<&Tensor>` - Multiple X tensors
- `y_series: Vec<&Tensor>` - Multiple Y tensors (same length as x_series)

**Requirements:**
- `x_series.len() == y_series.len()` - Same number of x and y tensors
- `x_series.len() > 0` - At least one pair required
- Each pair's x and y must have same length

**Returns:**
- `Result<(), Box<dyn std::error::Error>>`

**Features:**
- Each pair uses different color (cycling: blue, red, green, cyan, magenta, yellow)
- Global X/Y range computed from all series
- Supports up to 6 pairs before color cycling

**Example:**
```rust
use draw::{plot_pairs, PlotConfig, ImageType};
use tensor::{tensor, Tensor};

let x1 = tensor!([0.0, 1.0, 2.0]);
let y1 = tensor!([1.0, 2.0, 4.0]);
// Pair 1: y = 2x

let x2 = tensor!([0.0, 0.5, 1.0]);
let y2 = tensor!([1.0, 3.0, 9.0]);
// Pair 2: y = 3x^2

let config = PlotConfig::new()
    .title("Multiple (X,Y) Pairs")
    .legends(vec![
        "y = 2x".to_string(),
        "y = 3x^2".to_string(),
    ])
    .export("output/pairs.svg", ImageType::Svg)
    .show_window(false);

plot_pairs(&config, vec![&x1, &x2], vec![&y1, &y2])?;
```

---

### Export Functions

#### `export(config, x, y, filepath, image_type)`

Export a single series to file without window display.

**Parameters:**
- `config: &PlotConfig` - Plot configuration
- `x: &Tensor` - X-axis values
- `y: &Tensor` - Y-axis values
- `filepath: &str` - Output file path
- `image_type: ImageType` - Export format (currently only SVG supported)

**Returns:**
- `Result<(), Box<dyn std::error::Error>>`

**Example:**
```rust
use draw::{export, ImageType};
use tensor::{tensor, Tensor};

let x = tensor!([1.0, 2.0, 3.0, 4.0]);
let y = tensor!([2.0, 4.0, 6.0, 8.0]);

export(&PlotConfig::new().title("Data"), &x, &y, "output/plot.svg", ImageType::Svg)?;
// Creates output/plot.svg without showing window
```

#### `export_series(config, x, y_series, filepath, image_type)`

Export multiple series sharing same X values to file.

**Parameters:**
- `config: &PlotConfig` - Plot configuration
- `x: &Tensor` - X-axis values
- `y_series: Vec<&Tensor>` - Multiple Y tensors
- `filepath: &str` - Output file path
- `image_type: ImageType` - Export format (SVG)

**Returns:**
- `Result<(), Box<dyn std::error::Error>>`

**Example:**
```rust
use draw::{export_series, ImageType};
use tensor::{tensor, Tensor};

let x = tensor!([0.0, 1.0, 2.0, 3.0, 4.0]);
let y1 = tensor!([1.0, 2.0, 3.0, 4.0]);
let y2 = tensor!([0.5, 1.0, 1.5, 2.0]);

export_series(&PlotConfig::new(), &x, vec![&y1, &y2], "output/multi.svg", ImageType::Svg)?;
```

#### `export_series_array(config, x, y_series, filepath, image_type)`

Same as `export_series` but accepts array slices instead of Tensor references.

**Parameters:**
- `config: &PlotConfig` - Plot configuration
- `x: &[f32]` - X-axis values as slice
- `y_series: Vec<&[f32]>` - Multiple Y slices
- `filepath: &str` - Output file path
- `image_type: ImageType` - Export format (SVG)

**Returns:**
- `Result<(), Box<dyn std::error::Error>>`

---

#### `plot_array(config, x, y)`

Plot from slices (export-only, no window).

**Parameters:**
- `config: &PlotConfig` - Plot configuration
- `x: &[f32]` - X-axis slice
- `y: &[f32]` - Y-axis slice

**Returns:**
- `Result<(), Box<dyn std::error::Error>>`

---

#### `plot_series_array(config, x, y_series)`

Plot multiple series from slices (export-only).

**Parameters:**
- `config: &PlotConfig` - Plot configuration
- `x: &[f32]` - X-axis slice
- `y_series: Vec<&[f32]>` - Multiple Y slices

**Returns:**
- `Result<(), Box<dyn std::error::Error>>`

---

## Image Types

#### `ImageType`

Supported export formats.

**Variants:**
- `Svg` - Scalable Vector Graphics format

**Example:**
```rust
use draw::ImageType;

let config = PlotConfig::new()
    .export("output/plot.svg", ImageType::Svg);
```

---

## Plot Config

#### `ExportConfig`

Configuration for exporting plots.

**Fields:**
- `filepath: String` - Output file path
- `image_type: ImageType` - Export format

**Example:**
```rust
use draw::{ImageType, ExportConfig, PlotConfig};

let export_config = ExportConfig {
    filepath: "output/plot.svg".to_string(),
    image_type: ImageType::Svg,
};

let config = PlotConfig::new()
    .export_config(export_config);
```

---

## Color Palette

When plotting multiple series or pairs, colors cycle in order:

1. Blue
2. Red
3. Green
4. Cyan
5. Magenta
6. Yellow

Then repeats from 1.

---

## Font Support

The draw crate uses DejaVu Sans font family for text rendering:
- Normal weight for regular text
- Bold weight for titles and labels

---

## Window Display

When `show_window(true)`:
- Opens interactive window using minifb
- Shows real-time plot
- Close window with ESC key or close button
- Default: enabled

---

## File Output

Default output paths:
- **Working directory**: `./output/` (relative to project root)
- **Default filename**: `plot.png` (for display), `.svg` (for export)

Generated files are SVG (text-based vector graphics), suitable for:
- Web display
- Document embedding
- Vector editing tools
- Printing at any resolution

---

## Dependencies

- **plotters**: Chart rendering engine
- **minifb**: Window display library
- **dejavu**: Font rendering
- **tensor**: Data source

---

## Tips

1. **Disable window for server environments**: `.show_window(false)`
2. **Use vector export for publications**: SVG format is resolution-independent
3. **Set appropriate ranges**: `x_range()` and `y_range()` improve plot readability
4. **Customize legends**: Use descriptive names for series in multi-series plots
5. **Adjust ticks**: `x_ticks()` and `y_ticks()` control grid density
