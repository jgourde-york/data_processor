# Data Processor

A geospatial data processing pipeline that converts LiDAR point clouds (LAS/LAZ) and GeoTIFF rasters into machine-learning-ready patches with optional ground truth labels, train/val/test splits, and interactive shapefile editing tools.

## Directory Structure

```
data_processor/
├── configs/
│   ├── process_data.yml        # Main processing pipeline config
│   ├── create_shapefiles.yml   # AOI & test plot editor config
│   └── env.yml                 # Conda environment specification
├── modules/
│   ├── chm_generator.py        # CHM/DSM/DTM raster generation
│   ├── raster_io.py            # LAS & GeoTIFF I/O
│   ├── patch_generator.py      # Patch extraction, rotation, overlap
│   ├── normalizer.py           # Point cloud height normalization
│   ├── split_generator.py      # Train/val/test split assignment
│   ├── intensity_generator.py  # LiDAR intensity raster generation
│   ├── density_generator.py    # Point density raster generation
│   ├── aoi_generator.py        # Area of Interest shapefile generation
│   └── test_plot_generator.py  # Interactive test plot placement
├── process_data.py             # Main CLI entry point
├── create_shapefiles.py        # Interactive shapefile editors
├── visualize_splits.py         # Split visualization utility
└── logs/                       # Auto-created processing logs
```

## Quick Start

```bash
# Install environment
conda env create -f configs/env.yml
conda activate data_processor

# Process LAS files using default config
python data_processor/process_data.py

# Process a single LAS file with labels
python data_processor/process_data.py --las_file data/raw/site1.las --labels data/labels/site1.shp

# Generate patches from existing GeoTIFF rasters
python data_processor/process_data.py --from-rasters data/rasters/site/0.25m/chm/ --labels-dir data/labels/

# Create AOI and test region shapefiles interactively
python data_processor/create_shapefiles.py

# Visualize train/val/test splits
python data_processor/visualize_splits.py --splits-csv data/splits/split_0.15m_128px.csv \
    --raster data/rasters/site/0.15m/chm/plot.tif --output-dir viz/
```

## CLI Reference

### `process_data.py` — Main Processing Pipeline

Orchestrates the full pipeline: loading data, generating rasters, extracting patches, and writing splits.

```
python data_processor/process_data.py [OPTIONS]
```

| Flag | Description |
|---|---|
| `--config PATH` | Path to YAML config file (default: `configs/process_data.yml`) |
| `--las_file PATH` | Process a single LAS/LAZ file (bypasses config input modes) |
| `--labels PATH` | Labels shapefile for single-file mode (use with `--las_file`) |
| `--chm-only` | Stop after raster generation — skip rotation and patching |
| `--save-rotated` | Save rotated rasters in addition to the originals |
| `--normalize-points` | Height-normalize point cloud Z values by subtracting ground elevation |
| `--from-rasters PATH` | Generate patches from existing GeoTIFF(s) — single file or directory |
| `--labels-dir PATH` | Directory of label shapefiles, matched to rasters by stem name |
| `--band INT` | Band index (1-based) to extract from multi-band rasters |
| `--upsample-to FLOAT` | Resample rasters to a target resolution (meters) before patching |
| `--site-name STR` | Override the auto-detected site name for output paths |
| `--aoi PATH` | AOI shapefile — only patches fully inside are kept |
| `--test-regions PATH` | Test region shapefile for geographic train/test splitting |

### `create_shapefiles.py` — Interactive Shapefile Editors

Two interactive matplotlib-based editors for defining Areas of Interest and test plot regions.

**Config-driven batch mode** (processes all pairs from config):
```bash
python data_processor/create_shapefiles.py
python data_processor/create_shapefiles.py --config configs/create_shapefiles.yml
```

**Subcommand: `aoi`** — Generate AOI mask from labels with interactive gap editing:
```bash
python data_processor/create_shapefiles.py aoi --raster chm.tif --labels labels.shp \
    [--buffer METERS] [--max-gap-area M2] [--save output.shp]
```
- Left-click gaps to toggle fill/unfill
- Yellow = unfilled gaps, green = filled gaps
- Confirm & Save button writes the result

**Subcommand: `test-plots`** — Place rectangular test regions interactively:
```bash
python data_processor/create_shapefiles.py test-plots --raster chm.tif \
    [--labels labels.shp] [--plot-width M] [--plot-height M] \
    [--grid-size M] [--save output.shp]
```
- Left-click to place, right-click to remove
- Adjustable width/height/grid via text inputs
- Optional snap-to-grid and grid overlay

### `visualize_splits.py` — Split Visualization

Generates overview images of train/val/test patch distributions.

```bash
python data_processor/visualize_splits.py --splits-csv SPLITS.csv --raster RASTER.tif \
    --output-dir OUT/ [--labels LABELS.shp] [--aoi AOI.shp] \
    [--test-regions TEST.shp] [--base-dir BASE]
```

**Outputs:**
- `test_region_N.png` — stitched mosaic per test region
- `test_mosaic.png` — all test patches stitched together
- `train_val_mosaic.png` — train + validation patches stitched
- `full_raster_splits.png` — full raster with colored patch outlines by split

## Input Modes

The pipeline supports five config-driven input modes (set in `configs/process_data.yml` under `paths:`). Higher-numbered modes take priority when multiple are defined.

| Mode | Config Key | Description |
|---|---|---|
| 1 | `input_dir` | Single directory — process all LAS/LAZ files found |
| 2 | `input_dirs` | Multiple directories — batch process across sites |
| 3 | `input_files` | Specific files — process an explicit list of LAS/LAZ files |
| 4 | `file_label_pairs` | File-label pairs — each LAS matched with its label shapefile |
| 5 | `dir_label_pairs` | Directory-label pairs — directories of data + labels, matched by stem |

Mode 5 also supports GeoTIFF rasters (auto-detected by `.tif`/`.tiff` extension), with optional upsampling via the `upsample` config section.

## Processing Pipelines

### 1. Full LAS-to-Patches Pipeline

```
LAS/LAZ file
  → Load point cloud (x, y, z, classification, intensity, return_num, num_returns)
  → Generate rasters (CHM, DSM, DTM, intensity, density — as enabled)
  → Save full-extent GeoTIFFs
  → Compute optimal rotation (PCA-based, optional)
  → Extract patches (sliding window with adaptive overlap)
  → Filter by coverage threshold and AOI
  → Clip labels to each patch
  → Save per-layer patch GeoTIFFs + label shapefiles
  → Assign train/val/test splits
  → Write splits CSV
```

### 2. GeoTIFF-to-Patches Pipeline

```
GeoTIFF raster(s)
  → Load from disk (single file or directory of layers)
  → Upsample to target resolution (optional)
  → Compute optimal rotation (optional)
  → Extract patches
  → Save patches + splits CSV
```

### 3. Point Cloud Normalization

```
LAS/LAZ file
  → Interpolate DTM ground surface
  → Subtract ground elevation from every point's Z
  → Filter to [min_height, max_height]
  → Save normalized LAS/LAZ
```

### 4. AOI Generation

```
Label shapefile
  → Buffered union of all geometries
  → Detect interior gaps (concavities)
  → Auto-fill gaps below area threshold
  → Interactive editor for remaining gaps
  → Save AOI shapefile
```

### 5. Test Plot Placement

```
Raster + optional labels
  → Interactive matplotlib editor
  → Click to place/remove rectangular test regions
  → Adjustable dimensions, optional grid snap
  → Save test regions shapefile
```

## Core Modules

### CHMGenerator
Generates Canopy Height Models, Digital Surface Models, and Digital Terrain Models from point clouds.

- **DTM**: Ground points (LAS class 2) or 10th-percentile fallback, interpolated via `griddata` (linear/nearest/cubic), with gap filling
- **DSM**: Maximum elevation per pixel via bincount aggregation
- **CHM**: DSM minus DTM, clipped to `[min_height, max_height]`, optional Gaussian smoothing with percentile-based rescaling

### RasterIO
Handles all file I/O for LAS point clouds and GeoTIFF rasters.

- Loads LAS/LAZ into an Nx7 array (x, y, z, classification, intensity, return_num, num_returns)
- Auto-detects CRS from LAS header with configurable EPSG fallback
- Saves per-layer GeoTIFFs preserving CRS and affine transform
- Resamples rasters via `scipy.ndimage.zoom` (nearest, bilinear, cubic)

### PatchGenerator
Extracts georeferenced patches from rasters using a sliding window approach.

- **Rotation optimization**: PCA-based alignment to cardinal directions to minimize wasted area
- **Adaptive overlap**: Tests overlaps from `min_overlap` to 50%, selects the value maximizing valid patches
- **Support map**: Coarse downsampled coverage map for fast valid-area estimation
- **Label clipping**: Clips and transforms label geometries to patch pixel coordinates
- **AOI filtering**: Discards patches outside the AOI polygon

### PointCloudNormalizer
Height-normalizes point clouds by subtracting interpolated ground elevation from each point's Z coordinate.

### SplitGenerator
Assigns patches to train/val/test splits.

- **Geographic split**: Patches inside test region polygons go to the test set; remainder split into train/val
- **Ratio-based split**: Random assignment using configurable ratios
- Seeded RNG for reproducibility
- Writes a splits CSV with columns: `patch_file, label_file, split, site, plot`

### IntensityGenerator
Generates a normalized [0, 1] intensity raster from first-return LiDAR intensity values, averaged per pixel.

### DensityGenerator
Generates a point density raster in points/m², computed via bincount aggregation normalized by pixel area.

### AOIGenerator
Creates Area of Interest polygons from label geometries.

- Buffered union with configurable buffer distance
- Interior gap detection and auto-fill below an area threshold
- Interactive fill/unfill via the `create_shapefiles.py` editor

### TestPlotGenerator
Creates rectangular test region polygons with interactive placement.

- Configurable width, height, and grid size
- Optional snap-to-grid alignment
- Saves as a GeoDataFrame with CRS

## Configuration Reference

### `process_data.yml`

```yaml
paths:
  input_dir: "data/raw"              # Mode 1: single directory
  # input_dirs: [...]                # Mode 2: multiple directories
  # input_files: [...]               # Mode 3: specific files
  # file_label_pairs: [...]          # Mode 4: file-label pairs
  # dir_label_pairs: [...]           # Mode 5: directory-label pairs
  rasters_dir: "data/rasters"        # Output base directory
  normalized_dir: "data/normalized"  # Height-normalized output
  labels_shapefile: null             # Global labels (Modes 1-3)
  aoi: null                          # AOI shapefile path
  test_regions: null                 # Test regions shapefile path

chm:
  resolution: 0.15        # meters/pixel
  min_height: 0.5         # CHM floor (m)
  max_height: 50.0        # CHM ceiling (m)
  dtm_method: "linear"    # linear | nearest | cubic
  use_ground_class: true  # Use LAS class 2 for ground
  smoothing:
    enabled: true
    sigma: 1.25           # Gaussian sigma (pixels)
    rescale: "p95"        # p95 | none
    rescale_percentile: 95

layers:
  chm: true
  dsm: false
  dtm: false
  intensity: false
  density: false

patches:
  size: 128              # pixels (square)
  min_overlap: 0.5       # 0.0–0.5
  min_coverage: 0.7      # 0.0–1.0
  buffer_distance: 2.0   # meters

rotation:
  enabled: true

upsample:
  enabled: false
  target_resolution: 0.25  # meters
  method: bilinear         # nearest | bilinear | cubic
  band: null               # 1-based band index
  site_name: null          # override site name

output:
  compression: "lzw"       # lzw | deflate | none
  dtype: "float32"
  nodata: 0.0
  save_annotations: true
  naming_pattern: "{site}_{plot}_{index:04d}"

dataset:
  auto_split: true
  split_ratios:
    train: 0.70
    val: 0.15
    test: 0.15
  seed: 42
```

### `create_shapefiles.yml`

```yaml
mode: both                # aoi | test-plots | both

file_label_pairs:
  - raster: "data/rasters/Site/0.15m/chm/Plot.tif"
    labels: "data/labels/Plot.shp"

aoi_output_dir: "data/shapefiles/aoi"
test_regions_output_dir: "data/shapefiles/test_regions"

aoi:
  buffer_distance: 5.0    # meters
  max_gap_area: 500.0     # m² (0 = disable auto-fill)

test_plots:
  plot_width: 50.0        # meters
  plot_height: 50.0
  grid_size: 50.0
```

## Data Formats

### Input
| Format | Extensions | Description |
|---|---|---|
| LiDAR point clouds | `.las`, `.laz` | v1.0–1.4+, with classification, intensity, and return info |
| Rasters | `.tif`, `.tiff` | GeoTIFF, single or multi-band, any resolution/CRS |
| Labels | `.shp` | Shapefile with Polygon or PolygonZ geometries |

### Output

**Rasters and patches** are saved as single-band GeoTIFFs (LZW compressed, float32) organized by layer:
```
data/rasters/{site}/{resolution}/{layer}/{plot}.tif                    # Full raster
data/rasters/{site}/{resolution}/{layer}/{patch_size}/{patch}.tif      # Patch
data/rasters/{site}/{resolution}/labels/{patch_size}/{patch}.shp       # Patch labels
```

**Splits CSV** — one row per patch:
```
data/splits/split_{resolution}_{patch_size}.csv
```
Columns: `patch_file`, `label_file`, `split`, `site`, `plot`

**Normalized point clouds**: LAS/LAZ files saved to `data/normalized/`

**Shapefiles**: AOI masks and test region polygons saved as `.shp`

## Key Features

- **5 input modes** for flexible batch processing of LAS/LAZ and GeoTIFF data
- **Multi-layer raster generation** — CHM, DSM, DTM, intensity, and density from point clouds
- **PCA-based rotation optimization** to align raster data to cardinal directions and minimize wasted area
- **Adaptive patch overlap** — automatically tests overlaps to maximize valid patch yield
- **Geographic test splitting** — patches in defined test regions go to the test set
- **AOI filtering** — only patches fully inside the AOI are retained
- **Interactive shapefile editors** — matplotlib-based tools for AOI gap editing and test plot placement
- **Multi-band raster support** with band selection and resolution upsampling
- **Point cloud height normalization** by ground surface subtraction
- **Gaussian CHM smoothing** with percentile-based rescaling for sparse point clouds
- **Per-layer GeoTIFF output** preserving CRS and georeferencing
- **Reproducible splits** via seeded RNG
- **Dual logging** — INFO to console, DEBUG to timestamped log files

## Dependencies

Defined in `configs/env.yml`. Key packages:

- **Python** 3.10
- **numpy**, **scipy**, **pandas**, **scikit-learn** — numerical computing
- **rasterio**, **geopandas**, **fiona**, **shapely**, **pyproj** — geospatial I/O and geometry
- **laspy[lazrs]** — LAS/LAZ point cloud reading/writing
- **matplotlib** — visualization and interactive editors
- **pyyaml** — configuration loading
- **tqdm** — progress bars
