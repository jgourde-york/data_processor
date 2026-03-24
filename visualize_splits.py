"""
Visualize train/val/test splits for patch datasets.

Creates three visualizations:
1. Stitched test plots (one per test region)
2. Stitched train+val mosaic
3. Full raster overview with colored patch outlines by split
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger('data_processor')


def load_splits(splits_csv: Path) -> dict:
    """Load splits CSV and group by split type."""
    groups = defaultdict(list)
    with open(splits_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            groups[row['split']].append(row)
    return groups


def get_patch_bounds(tif_path: Path) -> tuple:
    """Get geographic bounds (left, bottom, right, top) from a GeoTIFF."""
    with rasterio.open(tif_path) as src:
        return src.bounds


def get_patch_data(tif_path: Path) -> tuple:
    """Read patch raster data and metadata."""
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        return data, src.bounds, src.crs


def stitch_patches(patch_rows: list, base_dir: Path) -> tuple:
    """Stitch patches into a single mosaic. Returns (array, bounds, crs)."""
    if not patch_rows:
        return None, None, None

    # Collect all bounds
    bounds_list = []
    for row in patch_rows:
        tif_path = base_dir / row['patch_file']
        if tif_path.exists():
            bounds_list.append((row, get_patch_bounds(tif_path)))

    if not bounds_list:
        return None, None, None

    # Compute mosaic extent
    all_lefts = [b.left for _, b in bounds_list]
    all_bottoms = [b.bottom for _, b in bounds_list]
    all_rights = [b.right for _, b in bounds_list]
    all_tops = [b.top for _, b in bounds_list]

    mosaic_left = min(all_lefts)
    mosaic_bottom = min(all_bottoms)
    mosaic_right = max(all_rights)
    mosaic_top = max(all_tops)

    # Determine resolution from first patch
    first_path = base_dir / bounds_list[0][0]['patch_file']
    with rasterio.open(first_path) as src:
        res_x = src.res[0]
        res_y = src.res[1]
        crs = src.crs

    # Create mosaic array
    mosaic_width = int(np.ceil((mosaic_right - mosaic_left) / res_x))
    mosaic_height = int(np.ceil((mosaic_top - mosaic_bottom) / res_y))
    mosaic = np.full((mosaic_height, mosaic_width), np.nan, dtype=np.float32)
    count = np.zeros((mosaic_height, mosaic_width), dtype=np.float32)

    # Place each patch
    for row, bounds in bounds_list:
        tif_path = base_dir / row['patch_file']
        with rasterio.open(tif_path) as src:
            data = src.read(1).astype(np.float32)

        col_off = int(round((bounds.left - mosaic_left) / res_x))
        row_off = int(round((mosaic_top - bounds.top) / res_y))

        h, w = data.shape
        # Clip to mosaic bounds
        h_clip = min(h, mosaic_height - row_off)
        w_clip = min(w, mosaic_width - col_off)
        if h_clip <= 0 or w_clip <= 0:
            continue

        patch_slice = data[:h_clip, :w_clip]
        mosaic_slice = mosaic[row_off:row_off+h_clip, col_off:col_off+w_clip]
        count_slice = count[row_off:row_off+h_clip, col_off:col_off+w_clip]

        # Average overlapping regions
        valid = ~np.isnan(patch_slice) & (patch_slice != 0)
        existing_valid = ~np.isnan(mosaic_slice)

        # Where mosaic is nan, just place data
        new_pixels = valid & ~existing_valid
        mosaic_slice[new_pixels] = 0
        mosaic_slice[valid] = np.where(
            existing_valid[valid],
            (mosaic_slice[valid] * count_slice[valid] + patch_slice[valid]) / (count_slice[valid] + 1),
            patch_slice[valid]
        )
        count_slice[valid] += 1

    mosaic_bounds = (mosaic_left, mosaic_bottom, mosaic_right, mosaic_top)
    return mosaic, mosaic_bounds, crs


def plot_stitched_mosaic(mosaic, bounds, title, output_path, labels_gdf=None, crs=None):
    """Plot a stitched mosaic with optional label overlay."""
    if mosaic is None:
        logger.warning(f"No patches to stitch for: {title}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    vmax = np.nanpercentile(mosaic[~np.isnan(mosaic)], 98) if np.any(~np.isnan(mosaic)) else 1
    ax.imshow(mosaic, extent=extent, cmap='viridis', vmin=0, vmax=vmax, origin='upper')

    if labels_gdf is not None and crs is not None:
        if labels_gdf.crs is None:
            labels_proj = labels_gdf.set_crs(crs)
        else:
            labels_proj = labels_gdf.to_crs(crs)
        # Clip to mosaic bounds
        mosaic_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
        labels_clip = labels_proj[labels_proj.intersects(mosaic_box)]
        if len(labels_clip) > 0:
            labels_clip.boundary.plot(ax=ax, color='red', linewidth=0.5, alpha=0.7)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(useOffset=False, style='plain')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_full_raster_with_outlines(raster_path, splits, base_dir, output_path,
                                   test_regions_path=None, aoi_path=None, labels_path=None):
    """Plot full raster with colored outlines around patches by split."""
    with rasterio.open(raster_path) as src:
        raster = src.read(1).astype(np.float32)
        raster_bounds = src.bounds
        raster_crs = src.crs
        extent = [raster_bounds.left, raster_bounds.right,
                  raster_bounds.bottom, raster_bounds.top]

    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    vmax = np.nanpercentile(raster[raster > 0], 98) if np.any(raster > 0) else 1
    ax.imshow(raster, extent=extent, cmap='Greens', vmin=0, vmax=vmax, origin='upper', alpha=0.8)

    # Colors for each split
    split_colors = {
        'train': '#2196F3',   # blue
        'val': '#FF9800',     # orange
        'test': '#F44336',    # red
    }
    split_labels = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}

    # Draw patch outlines
    for split_name, rows in splits.items():
        color = split_colors.get(split_name, '#9E9E9E')
        rects = []
        for row in rows:
            tif_path = base_dir / row['patch_file']
            if not tif_path.exists():
                continue
            b = get_patch_bounds(tif_path)
            rect = mpatches.Rectangle(
                (b.left, b.bottom), b.right - b.left, b.top - b.bottom,
                linewidth=0.8, edgecolor=color, facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)

    # Draw test regions if provided
    if test_regions_path and Path(test_regions_path).exists():
        test_gdf = gpd.read_file(test_regions_path)
        if test_gdf.crs is None:
            test_gdf = test_gdf.set_crs(raster_crs)
        else:
            test_gdf = test_gdf.to_crs(raster_crs)
        for _, region in test_gdf.iterrows():
            b = region.geometry.bounds
            rect = mpatches.Rectangle(
                (b[0], b[1]), b[2] - b[0], b[3] - b[1],
                linewidth=2.5, edgecolor='#D32F2F', facecolor='none',
                linestyle='--', alpha=1.0
            )
            ax.add_patch(rect)

    # Draw AOI boundary if provided
    if aoi_path and Path(aoi_path).exists():
        aoi_gdf = gpd.read_file(aoi_path)
        if aoi_gdf.crs is None:
            aoi_gdf = aoi_gdf.set_crs(raster_crs)
        else:
            aoi_gdf = aoi_gdf.to_crs(raster_crs)
        aoi_gdf.boundary.plot(ax=ax, color='white', linewidth=1.5, linestyle='-', alpha=0.9)

    # Draw labels if provided
    if labels_path and Path(labels_path).exists():
        labels_gdf = gpd.read_file(labels_path)
        if labels_gdf.crs is None:
            labels_gdf = labels_gdf.set_crs(raster_crs)
        else:
            labels_gdf = labels_gdf.to_crs(raster_crs)
        raster_box = box(raster_bounds.left, raster_bounds.bottom,
                         raster_bounds.right, raster_bounds.top)
        labels_clip = labels_gdf[labels_gdf.intersects(raster_box)]
        if len(labels_clip) > 0:
            labels_clip.boundary.plot(ax=ax, color='yellow', linewidth=0.3, alpha=0.5)

    # Legend
    legend_patches = []
    for split_name in ['train', 'val', 'test']:
        if split_name in splits:
            color = split_colors[split_name]
            label = f"{split_labels[split_name]} ({len(splits[split_name])} patches)"
            legend_patches.append(mpatches.Patch(edgecolor=color, facecolor='none',
                                                  linewidth=2, label=label))
    if test_regions_path and Path(test_regions_path).exists():
        legend_patches.append(mpatches.Patch(edgecolor='#D32F2F', facecolor='none',
                                              linewidth=2, linestyle='--',
                                              label='Test regions'))
    if aoi_path and Path(aoi_path).exists():
        legend_patches.append(mpatches.Patch(edgecolor='white', facecolor='none',
                                              linewidth=2, label='AOI boundary'))
    if labels_path and Path(labels_path).exists():
        legend_patches.append(mpatches.Patch(edgecolor='yellow', facecolor='none',
                                              linewidth=2, label='Labels'))

    ax.legend(handles=legend_patches, loc='upper right', fontsize=10,
              framealpha=0.9, edgecolor='gray')

    ax.set_title('Full Raster with Train/Val/Test Split Outlines', fontsize=14)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(useOffset=False, style='plain')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize patch splits')
    parser.add_argument('--splits-csv', required=True, help='Path to splits.csv')
    parser.add_argument('--raster', required=True, help='Path to full raster GeoTIFF')
    parser.add_argument('--output-dir', required=True, help='Directory for output images')
    parser.add_argument('--labels', default=None, help='Path to labels shapefile')
    parser.add_argument('--aoi', default=None, help='Path to AOI shapefile')
    parser.add_argument('--test-regions', default=None, help='Path to test regions shapefile')
    parser.add_argument('--base-dir', default='.', help='Base directory for resolving relative paths in CSV')
    args = parser.parse_args()

    # Set up logging: console (INFO) + file (DEBUG)
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger('data_processor')
    root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    log_file = log_dir / f"visualize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels if provided
    labels_gdf = None
    if args.labels:
        labels_gdf = gpd.read_file(args.labels)

    logger.info("Loading splits CSV...")
    splits = load_splits(Path(args.splits_csv))
    for split_name, rows in splits.items():
        logger.info(f"  {split_name}: {len(rows)} patches")

    # 1. Stitch test patches
    if 'test' in splits:
        logger.info("Stitching test patches...")

        # Group test patches by test region if test-regions provided
        if args.test_regions and Path(args.test_regions).exists():
            test_gdf = gpd.read_file(args.test_regions)
            # Get CRS from first patch
            first_test = base_dir / splits['test'][0]['patch_file']
            with rasterio.open(first_test) as src:
                patch_crs = src.crs
            if test_gdf.crs is None:
                test_gdf = test_gdf.set_crs(patch_crs)
            else:
                test_gdf = test_gdf.to_crs(patch_crs)

            for idx, region in test_gdf.iterrows():
                region_box = region.geometry
                region_patches = []
                for row in splits['test']:
                    tif_path = base_dir / row['patch_file']
                    if not tif_path.exists():
                        continue
                    b = get_patch_bounds(tif_path)
                    patch_center = box(b.left, b.bottom, b.right, b.top).centroid
                    if region_box.contains(patch_center):
                        region_patches.append(row)

                if region_patches:
                    logger.info(f"  Test region {idx+1}: {len(region_patches)} patches")
                    mosaic, mosaic_bounds, crs = stitch_patches(region_patches, base_dir)
                    plot_stitched_mosaic(
                        mosaic, mosaic_bounds,
                        f'Test Region {idx+1} ({len(region_patches)} patches)',
                        output_dir / f'test_region_{idx+1}.png',
                        labels_gdf=labels_gdf, crs=crs
                    )
        else:
            # Stitch all test patches together
            mosaic, mosaic_bounds, crs = stitch_patches(splits['test'], base_dir)
            plot_stitched_mosaic(
                mosaic, mosaic_bounds,
                f'Test Set ({len(splits["test"])} patches)',
                output_dir / 'test_mosaic.png',
                labels_gdf=labels_gdf, crs=crs
            )

    # 2. Stitch train+val patches
    train_val = splits.get('train', []) + splits.get('val', [])
    if train_val:
        logger.info(f"Stitching train+val patches ({len(train_val)} total)...")
        mosaic, mosaic_bounds, crs = stitch_patches(train_val, base_dir)
        plot_stitched_mosaic(
            mosaic, mosaic_bounds,
            f'Train + Validation ({len(splits.get("train", []))} train, {len(splits.get("val", []))} val)',
            output_dir / 'train_val_mosaic.png',
            labels_gdf=labels_gdf, crs=crs
        )

    # 3. Full raster with outlines
    logger.info("Creating full raster overview...")
    plot_full_raster_with_outlines(
        args.raster, splits, base_dir, output_dir / 'full_raster_splits.png',
        test_regions_path=args.test_regions,
        aoi_path=args.aoi,
        labels_path=args.labels
    )

    logger.info("Visualization complete")


if __name__ == '__main__':
    main()
