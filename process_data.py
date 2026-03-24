#!/usr/bin/env python3
"""
Data processing orchestrator: LAS/LAZ point clouds and GeoTIFF rasters into
patches for machine learning. Delegates raster generation, patch extraction,
and layer creation to specialized modules.

Usage:
    python process_data.py --config configs/process_data.yml
    python process_data.py --las_file data/raw/site1.las
    python process_data.py --chm-only
    python process_data.py --normalize-points --las_file data/raw/site1.las
    python process_data.py --from-rasters data/rasters/site/0.25m/chm/plot.tif --labels labels.shp
    python process_data.py --from-rasters data/rasters/dir/ --labels-dir labels/ --band 4 --upsample-to 0.25 --site-name MySite
"""

import argparse
import logging
import yaml
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional, List, Dict, Tuple

import geopandas as gpd
import numpy as np

from modules.chm_generator import CHMGenerator
from modules.intensity_generator import IntensityGenerator
from modules.density_generator import DensityGenerator
from modules.patch_generator import PatchGenerator
from modules.raster_io import RasterIO
from modules.normalizer import PointCloudNormalizer
from modules.split_generator import SplitGenerator

logger = logging.getLogger('data_processor')


class DataProcessor:
    """Orchestrates data loading, raster generation, and patch extraction.

    Accepts LAS/LAZ point clouds or GeoTIFF rasters and delegates to
    specialized modules for layer generation and patch extraction.
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.dataset_config = self.config.get('dataset', {})
        self.config.setdefault('upsample', {
            'enabled': False, 'target_resolution': None,
            'method': 'bilinear', 'band': None, 'site_name': None,
        })

        self.chm_generator = CHMGenerator(self.config)
        self.intensity_generator = IntensityGenerator(self.config)
        self.density_generator = DensityGenerator(self.config)
        self.patch_generator = PatchGenerator(self.config)
        self.raster_io = RasterIO(self.config)
        self.normalizer = PointCloudNormalizer(self.config)
        self.split_generator = SplitGenerator(self.dataset_config)

    def _load_config(self, config_path: str) -> dict:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self._validate_config(config)
        return config

    def _validate_config(self, config: dict):
        required_sections = ['paths', 'chm', 'layers', 'patches', 'output']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        if not any(config['layers'].values()):
            raise ValueError("At least one layer must be enabled in configuration")

        # Handle legacy output_dir config
        if 'output_dir' in config['paths'] and 'rasters_dir' not in config['paths']:
            config['paths']['rasters_dir'] = config['paths']['output_dir']

        logger.debug("Configuration validated")

    # -------------------------------------------------------------------------
    # Raster generation from point clouds
    # -------------------------------------------------------------------------

    def generate_rasters(self, points: np.ndarray, bounds: Dict) -> Dict[str, np.ndarray]:
        """Generate all enabled raster layers from point cloud."""
        resolution = self.config['chm']['resolution']

        x_coords = np.arange(bounds['min_x'], bounds['max_x'] + resolution, resolution)
        y_coords = np.arange(bounds['min_y'], bounds['max_y'] + resolution, resolution)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)

        rasters = {}

        if self.config['layers']['dtm'] or self.config['layers']['chm']:
            dtm = self.chm_generator.create_dtm(points, grid_x, grid_y)
            if self.config['layers']['dtm']:
                rasters['dtm'] = dtm

        if self.config['layers']['dsm'] or self.config['layers']['chm']:
            dsm = self.chm_generator.create_dsm(points, grid_x, grid_y)
            if self.config['layers']['dsm']:
                rasters['dsm'] = dsm

        if self.config['layers']['chm']:
            rasters['chm'] = self.chm_generator.create_chm(dsm, dtm)

        if self.config['layers']['intensity']:
            rasters['intensity'] = self.intensity_generator.create_intensity(points, grid_x, grid_y)

        if self.config['layers']['density']:
            rasters['density'] = self.density_generator.create_density(points, grid_x, grid_y)

        # Flip: rasters are generated with row 0 at min_y (bottom), but GeoTIFF
        # convention with negative y-resolution expects row 0 at max_y (top)
        rasters = {key: np.flipud(raster) for key, raster in rasters.items()}

        return rasters

    # -------------------------------------------------------------------------
    # Input discovery
    # -------------------------------------------------------------------------

    def discover_las_files_from_directory(self, input_dir: str) -> List[Path]:
        """Discover all LAS/LAZ files in a directory."""
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        las_files = list(input_dir.glob("*.las")) + list(input_dir.glob("*.laz"))
        las_files += list(input_dir.glob("*.LAS")) + list(input_dir.glob("*.LAZ"))

        # Deduplicate (Windows filesystem is case-insensitive)
        las_files = list({f.resolve(): f for f in las_files}.values())
        return las_files

    def resolve_input_mode(self) -> Tuple[List[Tuple[Path, Optional[Path]]], str]:
        """Resolve input mode from config and return (file_label_pairs, mode_name)."""
        config_paths = self.config['paths']

        # Mode 4: File-label pairs (highest priority)
        if 'file_label_pairs' in config_paths and config_paths['file_label_pairs']:
            logger.info("Using Mode 4: File-label pairs")
            pairs = []
            for pair in config_paths['file_label_pairs']:
                las_file = Path(pair['las_file'])
                if not las_file.exists():
                    raise FileNotFoundError(f"LAS file not found: {las_file}")
                labels = Path(pair['labels']) if pair.get('labels') else None
                if labels and not labels.exists():
                    raise FileNotFoundError(f"Label file not found: {labels}")
                pairs.append((las_file, labels))
            return pairs, "file_label_pairs"

        # Mode 5: Directory-label pairs
        if 'dir_label_pairs' in config_paths and config_paths['dir_label_pairs']:
            logger.info("Using Mode 5: Directory-label pairs")
            pairs = []
            for pair in config_paths['dir_label_pairs']:
                # Support both 'data_dir' (preferred) and 'las_dir' (legacy)
                data_dir = Path(pair.get('data_dir') or pair['las_dir'])
                if not data_dir.exists():
                    raise FileNotFoundError(f"Data directory not found: {data_dir}")
                labels_dir = Path(pair['labels_dir']) if pair.get('labels_dir') else None
                if labels_dir and not labels_dir.exists():
                    raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

                # Discover LAS files first, fall back to TIF rasters
                data_files = self.discover_las_files_from_directory(data_dir)
                if not data_files:
                    tif_files = list(data_dir.glob('*.tif')) + list(data_dir.glob('*.tiff'))
                    data_files = list({f.resolve(): f for f in tif_files}.values())
                    data_files.sort(key=lambda f: f.name)
                    if data_files:
                        logger.info(f"Found {len(data_files)} raster files in {data_dir}")

                for data_file in data_files:
                    label_file = None
                    if labels_dir:
                        label_candidates = list(labels_dir.glob(f"{data_file.stem}.*shp"))
                        if label_candidates:
                            label_file = label_candidates[0]
                    pairs.append((data_file, label_file))
            return pairs, "dir_label_pairs"

        # Mode 3: Specific files list
        if 'input_files' in config_paths and config_paths['input_files']:
            logger.info("Using Mode 3: Specific files list")
            pairs = []
            global_labels = Path(config_paths['labels_shapefile']) if config_paths.get('labels_shapefile') else None
            for las_path in config_paths['input_files']:
                las_file = Path(las_path)
                if not las_file.exists():
                    raise FileNotFoundError(f"LAS file not found: {las_file}")
                pairs.append((las_file, global_labels))
            return pairs, "input_files"

        # Mode 2: Multiple directories
        if 'input_dirs' in config_paths and config_paths['input_dirs']:
            logger.info("Using Mode 2: Multiple directories")
            pairs = []
            global_labels = Path(config_paths['labels_shapefile']) if config_paths.get('labels_shapefile') else None
            for input_dir in config_paths['input_dirs']:
                for las_file in self.discover_las_files_from_directory(input_dir):
                    pairs.append((las_file, global_labels))
            return pairs, "input_dirs"

        # Mode 1: Single directory (default)
        logger.info("Using Mode 1: Single directory")
        input_dir = config_paths.get('input_dir', 'data/raw')
        las_files = self.discover_las_files_from_directory(input_dir)
        global_labels = Path(config_paths['labels_shapefile']) if config_paths.get('labels_shapefile') else None
        return [(f, global_labels) for f in las_files], "input_dir"

    def _extract_site_plot_names(self, las_path: Path) -> Tuple[str, str]:
        """Extract (site_name, plot_name) from file path.

        E.g., data/raw/NorthBay/B2_N.las -> ('NorthBay', 'B2_N')
              data/raw/ITC_TRUTH_veg218.laz -> ('ITC_TRUTH_veg218', 'ITC_TRUTH_veg218')
        """
        plot_name = las_path.stem
        if las_path.parent.name in ['raw', 'data']:
            site_name = plot_name
        else:
            site_name = las_path.parent.name
        return site_name, plot_name

    # -------------------------------------------------------------------------
    # Rotation orchestration
    # -------------------------------------------------------------------------

    def _apply_rotation(self, rasters: Dict[str, np.ndarray], metadata: Dict,
                        labels_gdf: Optional[gpd.GeoDataFrame],
                        resolution: Tuple[float, float],
                        save_rotated: bool = False,
                        rasters_base: Optional[Path] = None,
                        plot_name: Optional[str] = None,
                        ) -> Tuple[Dict[str, np.ndarray], Dict, Optional[gpd.GeoDataFrame], float]:
        """Compute and apply rotation to rasters/labels, updating metadata bounds.

        Returns (rotated_rasters, updated_metadata, rotated_labels, rotation_angle).
        """
        if not self.config['rotation']['enabled']:
            return rasters, metadata, labels_gdf, 0.0

        reference_raster = list(rasters.values())[0]
        rotation_angle = self.patch_generator.compute_optimal_rotation(reference_raster)
        logger.info(f"Optimal rotation angle: {rotation_angle:.2f}")

        if rotation_angle == 0:
            return rasters, metadata, labels_gdf, 0.0

        original_shape = reference_raster.shape
        origin = (metadata['bounds']['min_x'], metadata['bounds']['max_y'])

        rotated_rasters, crop_bounds = self.patch_generator.rotate_rasters(rasters, rotation_angle)
        rotated_shape = list(rotated_rasters.values())[0].shape

        if labels_gdf is not None and len(labels_gdf) > 0:
            labels_gdf = self.patch_generator.rotate_labels(
                labels_gdf, original_shape, rotation_angle,
                crop_bounds, resolution, origin
            )
            logger.debug(f"Rotated {len(labels_gdf)} labels")

        # Update metadata for rotated raster bounds.
        # scipy.ndimage.rotate rotates around the array center, so we compute
        # the world-coordinate center, then offset by the crop bounds.
        rotated_metadata = metadata.copy()
        rotated_metadata['rotated'] = True
        rotated_metadata['rotation_angle'] = rotation_angle

        orig_center_col = original_shape[1] / 2
        orig_center_row = original_shape[0] / 2
        center_x_world = origin[0] + orig_center_col * resolution[0]
        center_y_world = origin[1] - orig_center_row * resolution[1]

        angle_rad = np.radians(rotation_angle)
        h, w = original_shape
        padded_h = int(np.ceil(abs(h * np.cos(angle_rad)) + abs(w * np.sin(angle_rad))))
        padded_w = int(np.ceil(abs(w * np.cos(angle_rad)) + abs(h * np.sin(angle_rad))))

        row_min, _, col_min, _ = crop_bounds
        offset_col = col_min - padded_w / 2
        offset_row = row_min - padded_h / 2
        new_min_x = center_x_world + offset_col * resolution[0]
        new_max_y = center_y_world - offset_row * resolution[1]

        rotated_metadata['bounds'] = {
            'min_x': new_min_x,
            'max_y': new_max_y,
            'max_x': new_min_x + rotated_shape[1] * resolution[0],
            'min_y': new_max_y - rotated_shape[0] * resolution[1],
        }

        logger.debug(f"Rotation center (world): ({center_x_world:.2f}, {center_y_world:.2f})")
        logger.debug(f"New origin: ({new_min_x:.2f}, {new_max_y:.2f})")

        if save_rotated and rasters_base and plot_name:
            logger.info("Saving rotated rasters")
            self.raster_io.save_rasters(rotated_rasters, rotated_metadata, rasters_base,
                                        plot_name, suffix="_rotated",
                                        rotation_angle=rotation_angle)

        return rotated_rasters, rotated_metadata, labels_gdf, rotation_angle

    # -------------------------------------------------------------------------
    # Processing pipelines
    # -------------------------------------------------------------------------

    def process_single_file(self, las_path: Path, rasters_dir: Path,
                           site_name: str, plot_name: str,
                           labels_gdf: Optional[gpd.GeoDataFrame] = None,
                           chm_only: bool = False,
                           save_rotated: bool = False,
                           aoi_gdf: Optional[gpd.GeoDataFrame] = None) -> List[str]:
        """Process a single LAS/LAZ file into per-layer rasters and patches."""
        logger.info(f"Processing: {las_path.name}")
        logger.debug(f"Site: {site_name}, Plot: {plot_name}")

        resolution_str = f"{self.config['chm']['resolution']}m"
        rasters_base = rasters_dir / site_name / resolution_str

        total_steps = 2 if chm_only else 4

        # Step 1: Load LAS
        logger.info(f"[1/{total_steps}] Loading LAS/LAZ file")
        points, metadata = self.raster_io.load_las_file(str(las_path))
        logger.info(f"Loaded {metadata['num_points']:,} points")
        logger.debug(f"Bounds: X=[{metadata['bounds']['min_x']:.2f}, {metadata['bounds']['max_x']:.2f}], "
                     f"Y=[{metadata['bounds']['min_y']:.2f}, {metadata['bounds']['max_y']:.2f}]")

        # Step 2: Generate rasters
        logger.info(f"[2/{total_steps}] Generating raster layers")
        enabled_layers = [k for k, v in self.config['layers'].items() if v]
        logger.debug(f"Layers: {', '.join(enabled_layers)}")

        rasters = self.generate_rasters(points, metadata['bounds'])

        for layer_name, raster in rasters.items():
            non_zero = np.sum(raster != 0)
            coverage = (non_zero / raster.size) * 100
            logger.debug(f"{layer_name}: {raster.shape} ({coverage:.1f}% coverage)")

        logger.info("Saving rasters")
        self.raster_io.save_rasters(rasters, metadata, rasters_base, plot_name)

        if chm_only:
            logger.info("Raster generation complete (--chm-only)")
            return []

        # Step 3: Rotation (optional)
        res = self.config['chm']['resolution']
        logger.info(f"[3/{total_steps}] {'Computing optimal rotation' if self.config['rotation']['enabled'] else 'Rotation optimization disabled'}")
        rotated_rasters, metadata, labels_gdf, rotation_angle = self._apply_rotation(
            rasters, metadata, labels_gdf,
            resolution=(res, res),
            save_rotated=save_rotated,
            rasters_base=rasters_base,
            plot_name=plot_name,
        )

        # Step 4: Generate patches
        logger.info(f"[4/{total_steps}] Generating patches")
        patch_names = self.patch_generator.generate_patches(
            rotated_rasters, metadata, rasters_base,
            site_name, plot_name, labels_gdf, rotation_angle,
            aoi_gdf=aoi_gdf,
        )

        logger.info(f"Generated {len(patch_names)} patches")
        return patch_names

    def process_raster_file(self, raster_path: Path, rasters_dir: Path,
                            labels_gdf: Optional[gpd.GeoDataFrame] = None,
                            save_rotated: bool = False,
                            aoi_gdf: Optional[gpd.GeoDataFrame] = None,
                            upsample_to: Optional[float] = None,
                            band: Optional[int] = None,
                            site_name_override: Optional[str] = None,
                            layer_name_override: Optional[str] = None) -> List[str]:
        """Generate patches from pre-existing GeoTIFF raster(s).

        Skips LAS loading and raster generation; loads rasters from disk
        and runs rotation + patch extraction.
        """
        # Step 1: Load rasters from disk
        logger.info(f"Loading rasters from: {raster_path}")
        rasters, metadata, site_name, plot_name = self.raster_io.load_rasters_from_disk(
            raster_path, band=band, layer_name_override=layer_name_override)
        if site_name_override:
            site_name = site_name_override

        # Determine resolution from the raster itself
        first_raster = list(rasters.values())[0]
        bounds = metadata['bounds']
        detected_res_x = round((bounds['max_x'] - bounds['min_x']) / first_raster.shape[1], 4)
        detected_res_y = round((bounds['max_y'] - bounds['min_y']) / first_raster.shape[0], 4)

        logger.info(f"Site: {site_name}, Plot: {plot_name}")
        logger.info(f"Detected resolution: {detected_res_x:.4f}m x {detected_res_y:.4f}m")

        # Upsample rasters if requested
        if upsample_to is not None and upsample_to < detected_res_x:
            scale = detected_res_x / upsample_to
            old_shape = first_raster.shape
            logger.info(f"Upsampling from {detected_res_x}m to {upsample_to}m (scale={scale:.1f}x)")
            for layer_name in rasters:
                rasters[layer_name] = self.raster_io.resample_raster(
                    rasters[layer_name], detected_res_x, upsample_to)
            detected_res_x = round(upsample_to, 4)
            detected_res_y = round(upsample_to, 4)
            first_raster = list(rasters.values())[0]
            logger.info(f"Resampled: {old_shape} -> {first_raster.shape}")

        resolution_str = f"{detected_res_x}m"
        rasters_base = rasters_dir / site_name / resolution_str

        # Save the full raster (before rotation/patching)
        self.raster_io.save_rasters(rasters, metadata, rasters_base, plot_name,
                                    resolution_override=detected_res_x)

        for layer_name, raster in rasters.items():
            non_zero = np.sum(raster != 0)
            coverage = (non_zero / raster.size) * 100
            logger.debug(f"{layer_name}: {raster.shape} ({coverage:.1f}% coverage)")

        # Step 2: Rotation (optional)
        logger.info(f"[1/2] {'Computing optimal rotation' if self.config['rotation']['enabled'] else 'Rotation optimization disabled'}")
        rotated_rasters, metadata, labels_gdf, rotation_angle = self._apply_rotation(
            rasters, metadata, labels_gdf,
            resolution=(detected_res_x, detected_res_y),
            save_rotated=save_rotated,
            rasters_base=rasters_base,
            plot_name=plot_name,
        )

        # Step 3: Generate patches
        logger.info("[2/2] Generating patches")
        patch_names = self.patch_generator.generate_patches(
            rotated_rasters, metadata, rasters_base,
            site_name, plot_name, labels_gdf, rotation_angle,
            aoi_gdf=aoi_gdf,
            resolution_override=(detected_res_x, detected_res_y),
        )

        logger.info(f"Generated {len(patch_names)} patches")
        return patch_names

    # -------------------------------------------------------------------------
    # Batch processing
    # -------------------------------------------------------------------------

    def process_all(self, chm_only: bool = False, save_rotated: bool = False,
                    aoi_gdf: Optional[gpd.GeoDataFrame] = None,
                    test_regions_gdf: Optional[gpd.GeoDataFrame] = None) -> dict:
        """Process all files based on configuration input mode."""
        paths_config = self.config['paths']
        rasters_dir = Path(paths_config.get('rasters_dir', 'data/rasters'))
        resolution_str = f"{self.config['chm']['resolution']}m"
        patch_size_str = f"{self.config['patches']['size']}px"

        # Load AOI and test regions from config if not provided via args
        if aoi_gdf is None and paths_config.get('aoi'):
            aoi_path = Path(paths_config['aoi'])
            if aoi_path.exists():
                aoi_gdf = gpd.read_file(aoi_path)
                logger.info(f"Loaded AOI with {len(aoi_gdf)} polygon(s) from config")
            else:
                logger.warning(f"AOI file not found: {aoi_path}")

        if test_regions_gdf is None and paths_config.get('test_regions'):
            tr_path = Path(paths_config['test_regions'])
            if tr_path.exists():
                test_regions_gdf = gpd.read_file(tr_path)
                logger.info(f"Loaded {len(test_regions_gdf)} test region(s) from config")
            else:
                logger.warning(f"Test regions file not found: {tr_path}")

        file_label_pairs, mode_name = self.resolve_input_mode()
        logger.info(f"Found {len(file_label_pairs)} file(s) to process")

        upsample_config = self.config.get('upsample', {})

        all_patch_info = []
        raster_patch_infos = []
        failed_files = []

        for data_file, label_file in file_label_pairs:
            try:
                is_raster = data_file.suffix.lower() in ('.tif', '.tiff')

                labels_gdf = None
                has_labels = label_file is not None
                if has_labels:
                    logger.debug(f"Loading labels: {label_file.name}")
                    labels_gdf = gpd.read_file(label_file)
                    if labels_gdf.crs is None:
                        logger.debug("Labels have no CRS, will inherit from data")
                    logger.info(f"Loaded {len(labels_gdf)} label geometries")

                if is_raster:
                    upsample_to = upsample_config.get('target_resolution') if upsample_config.get('enabled') else None
                    band = upsample_config.get('band')
                    site_name_override = upsample_config.get('site_name')

                    patch_infos = self.process_raster_file(
                        data_file, rasters_dir,
                        labels_gdf=labels_gdf,
                        save_rotated=save_rotated,
                        aoi_gdf=aoi_gdf,
                        upsample_to=upsample_to,
                        band=band,
                        site_name_override=site_name_override,
                        layer_name_override='chm',
                    )
                    raster_patch_infos.extend(patch_infos)

                else:
                    site_name, plot_name = self._extract_site_plot_names(data_file)

                    patch_names = self.process_single_file(
                        data_file, rasters_dir,
                        site_name, plot_name, labels_gdf,
                        chm_only=chm_only, save_rotated=save_rotated,
                        aoi_gdf=aoi_gdf,
                    )

                    for patch_info in patch_names:
                        all_patch_info.append({
                            'patch_name': patch_info['filename'],
                            'site': site_name,
                            'plot': plot_name,
                            'has_labels': patch_info.get('has_labels', has_labels),
                            'in_aoi': patch_info.get('in_aoi', True),
                            'bounds': patch_info.get('bounds'),
                        })

            except Exception as e:
                logger.error(f"Error processing {data_file.name}: {e}", exc_info=True)
                failed_files.append(data_file.name)

        # Generate splits CSVs
        if self.dataset_config.get('auto_split', False) and all_patch_info:
            self.split_generator.generate_las_splits(
                all_patch_info, rasters_dir, resolution_str, patch_size_str,
                test_regions_gdf=test_regions_gdf)

        if raster_patch_infos:
            first_raster = next(f for f, _ in file_label_pairs if f.suffix.lower() in ('.tif', '.tiff'))
            self.split_generator.generate_raster_splits(
                raster_patch_infos, first_raster, rasters_dir,
                self.config['patches']['size'],
                test_regions_gdf=test_regions_gdf,
                site_name_override=upsample_config.get('site_name'),
            )

        logger.info("Processing complete")
        logger.info(f"Input mode: {mode_name}")
        logger.info(f"Processed: {len(file_label_pairs) - len(failed_files)}/{len(file_label_pairs)} files")
        logger.info(f"Total patches: {len(all_patch_info)}")
        if failed_files:
            logger.error(f"Failed files: {', '.join(failed_files)}")
        logger.info(f"Output directory: {rasters_dir}")

        return {
            'input_mode': mode_name,
            'total_files': len(file_label_pairs),
            'processed_files': len(file_label_pairs) - len(failed_files),
            'failed_files': failed_files,
            'total_patches': len(all_patch_info),
            'rasters_dir': str(rasters_dir),
        }


# =============================================================================
# CLI helpers
# =============================================================================

def _load_aoi_and_test_regions(args, paths_config):
    """Load AOI and test region shapefiles from CLI args or config.

    CLI args take precedence over config values.
    Returns (aoi_gdf, test_regions_gdf) — either may be None.
    """
    aoi_gdf = None
    aoi_source = args.aoi or paths_config.get('aoi')
    if aoi_source:
        aoi_path = Path(aoi_source)
        if not aoi_path.exists():
            logger.error(f"AOI file not found: {aoi_path}")
            sys.exit(1)
        aoi_gdf = gpd.read_file(aoi_path)
        logger.info(f"Loaded AOI with {len(aoi_gdf)} polygon(s)")

    test_regions_gdf = None
    tr_source = args.test_regions or paths_config.get('test_regions')
    if tr_source:
        tr_path = Path(tr_source)
        if not tr_path.exists():
            logger.error(f"Test regions file not found: {tr_path}")
            sys.exit(1)
        test_regions_gdf = gpd.read_file(tr_path)
        logger.info(f"Loaded {len(test_regions_gdf)} test region(s)")

    return aoi_gdf, test_regions_gdf


def main():
    parser = argparse.ArgumentParser(
        description="Process LAS/LAZ or GeoTIFF data into patches for machine learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_data.py                                                  # Use default config
  python process_data.py --config configs/my_config.yml                   # Custom config
  python process_data.py --las_file data/raw/site1.las --labels site1.shp # Single file with labels
  python process_data.py --chm-only                                       # Rasters only, no patches
  python process_data.py --save-rotated                                   # Also save rotated rasters
  python process_data.py --normalize-points --las_file data/raw/site1.las # Height-normalize points
  python process_data.py --from-rasters data/rasters/site/0.25m/chm/plot.tif --labels labels.shp  # Patches from existing raster
  python process_data.py --from-rasters raster_dir/ --labels-dir shp_dir/ --band 4 --upsample-to 0.25  # Upsample + patch

Configuration modes (specified in configs/process_data.yml):
  Mode 1: Single directory      - input_dir: "data/raw"
  Mode 2: Multiple directories  - input_dirs: ["dir1", "dir2"]
  Mode 3: Specific files list   - input_files: ["file1.las", "file2.las"]
  Mode 4: File-label pairs      - file_label_pairs: [{las_file: ..., labels: ...}]
  Mode 5: Directory-label pairs - dir_label_pairs: [{las_dir: ..., labels_dir: ...}]
        """
    )

    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file (default: configs/process_data.yml)')
    parser.add_argument('--las_file', type=str,
                       help='Process single LAS/LAZ file (bypasses config input modes)')
    parser.add_argument('--labels', type=str,
                       help='Labels shapefile for single file mode (use with --las_file)')
    parser.add_argument('--chm-only', action='store_true', default=False,
                       help='Stop after raster generation (skip rotation and patch extraction)')
    parser.add_argument('--save-rotated', action='store_true', default=False,
                       help='Also save rotated rasters (in addition to unrotated)')
    parser.add_argument('--normalize-points', action='store_true', default=False,
                       help='Normalize point cloud Z values by subtracting DTM ground elevation')
    parser.add_argument('--from-rasters', type=str, default=None,
                       help='Generate patches from existing GeoTIFF raster(s). '
                            'Path to a single .tif file or a directory of .tif files.')
    parser.add_argument('--labels-dir', type=str, default=None,
                       help='Directory of label shapefiles. Matched to rasters by stem name '
                            '(e.g., T1.tif matches T1.shp). Use with --from-rasters directory mode.')
    parser.add_argument('--band', type=int, default=None,
                       help='Band index (1-based) to extract from multi-band rasters. '
                            'E.g., --band 4 to use the 4th band (CHM in Finnish Taiga).')
    parser.add_argument('--upsample-to', type=float, default=None,
                       help='Target resolution in meters. Resamples rasters to this resolution '
                            'before patching. E.g., --upsample-to 0.25 to upsample 0.5m data.')
    parser.add_argument('--site-name', type=str, default=None,
                       help='Override auto-detected site name for output paths and splits CSV.')
    parser.add_argument('--aoi', type=str, default=None,
                       help='AOI shapefile to filter patches. Only patches fully inside the AOI are kept.')
    parser.add_argument('--test-regions', type=str, default=None,
                       help='Shapefile with test region polygons. Patches inside these regions '
                            'are assigned to the test split; remaining patches are split into train/val.')

    args = parser.parse_args()

    # Set up logging: console (INFO) + file (DEBUG)
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger('data_processor')
    root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    log_file = log_dir / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.stream.reconfigure(write_through=True)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    if args.config:
        config_path = args.config
    else:
        script_dir = Path(__file__).parent
        config_path = script_dir / 'configs' / 'process_data.yml'
        if not config_path.exists():
            logger.error(f"Default config file not found: {config_path}")
            logger.error("Please specify a config file with --config or create configs/process_data.yml")
            sys.exit(1)
        logger.info(f"Using default config: {config_path}")

    try:
        processor = DataProcessor(str(config_path))
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    try:
        # --from-rasters: generate patches from existing GeoTIFF rasters
        if args.from_rasters:
            raster_path = Path(args.from_rasters)
            if not raster_path.exists():
                logger.error(f"Raster path not found: {raster_path}")
                sys.exit(1)

            paths_config = processor.config['paths']
            rasters_dir = Path(paths_config.get('rasters_dir', 'data/rasters'))

            aoi_gdf, test_regions_gdf = _load_aoi_and_test_regions(
                args, processor.config['paths'])

            if raster_path.is_dir():
                tif_files = sorted(raster_path.glob('*.tif')) + sorted(raster_path.glob('*.tiff'))
                seen = set()
                unique_tifs = []
                for f in tif_files:
                    if f.name not in seen:
                        seen.add(f.name)
                        unique_tifs.append(f)
                tif_files = unique_tifs

                if not tif_files:
                    logger.error(f"No TIF files found in: {raster_path}")
                    sys.exit(1)
                logger.info(f"Found {len(tif_files)} raster files in {raster_path}")

                labels_dir = Path(args.labels_dir) if args.labels_dir else None
                all_patch_infos = []

                for tif_file in tif_files:
                    labels_gdf = None
                    if labels_dir:
                        shp_file = labels_dir / f"{tif_file.stem}.shp"
                        if shp_file.exists():
                            labels_gdf = gpd.read_file(shp_file)
                            logger.info(f"Loaded {len(labels_gdf)} labels for {tif_file.stem}")
                        else:
                            logger.warning(f"No label file found for {tif_file.stem}")
                    elif args.labels:
                        label_path = Path(args.labels)
                        if label_path.exists():
                            labels_gdf = gpd.read_file(label_path)

                    patch_infos = processor.process_raster_file(
                        tif_file, rasters_dir,
                        labels_gdf=labels_gdf,
                        save_rotated=args.save_rotated,
                        aoi_gdf=aoi_gdf,
                        upsample_to=args.upsample_to,
                        band=args.band,
                        site_name_override=args.site_name,
                        layer_name_override='chm',
                    )
                    all_patch_infos.extend(patch_infos)

                if all_patch_infos:
                    processor.split_generator.generate_raster_splits(
                        all_patch_infos, raster_path, rasters_dir,
                        processor.config['patches']['size'],
                        test_regions_gdf=test_regions_gdf,
                        site_name_override=args.site_name,
                    )
            else:
                labels_gdf = None
                if args.labels:
                    label_path = Path(args.labels)
                    if not label_path.exists():
                        logger.error(f"Label file not found: {label_path}")
                        sys.exit(1)
                    labels_gdf = gpd.read_file(label_path)
                    logger.info(f"Loaded {len(labels_gdf)} label geometries")

                patch_infos = processor.process_raster_file(
                    raster_path, rasters_dir,
                    labels_gdf=labels_gdf,
                    save_rotated=args.save_rotated,
                    aoi_gdf=aoi_gdf,
                    upsample_to=args.upsample_to,
                    band=args.band,
                    site_name_override=args.site_name,
                )

                if patch_infos:
                    processor.split_generator.generate_raster_splits(
                        patch_infos, raster_path, rasters_dir,
                        processor.config['patches']['size'],
                        test_regions_gdf=test_regions_gdf,
                        site_name_override=args.site_name,
                    )

            sys.exit(0)

        # --normalize-points: height-normalize point cloud(s)
        if args.normalize_points:
            paths_config = processor.config['paths']
            normalized_dir = Path(paths_config.get('normalized_dir', 'data/normalized'))

            if args.las_file:
                las_path = Path(args.las_file)
                if not las_path.exists():
                    logger.error(f"LAS file not found: {las_path}")
                    sys.exit(1)
                processor.normalizer.normalize(las_path, normalized_dir)
            else:
                file_label_pairs, _ = processor.resolve_input_mode()
                logger.info(f"Normalizing {len(file_label_pairs)} file(s)")
                for las_file, _ in file_label_pairs:
                    try:
                        processor.normalizer.normalize(las_file, normalized_dir)
                    except Exception as e:
                        logger.error(f"Error normalizing {las_file.name}: {e}", exc_info=True)

        # Run raster pipeline unless --normalize-points was used alone
        run_raster_pipeline = not args.normalize_points or args.chm_only
        if run_raster_pipeline:
            if args.las_file:
                las_path = Path(args.las_file)
                if not las_path.exists():
                    logger.error(f"LAS file not found: {las_path}")
                    sys.exit(1)

                paths_config = processor.config['paths']
                rasters_dir = Path(paths_config.get('rasters_dir', 'data/rasters'))

                labels_gdf = None
                if args.labels:
                    label_path = Path(args.labels)
                    if not label_path.exists():
                        logger.error(f"Label file not found: {label_path}")
                        sys.exit(1)
                    labels_gdf = gpd.read_file(label_path)
                    logger.info(f"Loaded {len(labels_gdf)} label geometries")

                site_name, plot_name = processor._extract_site_plot_names(las_path)

                aoi_gdf, test_regions_gdf = _load_aoi_and_test_regions(
                    args, processor.config['paths'])

                patch_names = processor.process_single_file(
                    las_path, rasters_dir,
                    site_name, plot_name, labels_gdf,
                    chm_only=args.chm_only,
                    save_rotated=args.save_rotated,
                    aoi_gdf=aoi_gdf,
                )

                # Generate splits CSV for single-file mode
                if (not args.chm_only and patch_names
                        and processor.dataset_config.get('auto_split', False)):
                    resolution_str = f"{processor.config['chm']['resolution']}m"
                    patch_size_str = f"{processor.config['patches']['size']}px"
                    has_labels = labels_gdf is not None
                    patch_info = [{
                        'patch_name': p['filename'],
                        'site': site_name,
                        'plot': plot_name,
                        'has_labels': p.get('has_labels', has_labels),
                        'in_aoi': p.get('in_aoi', True),
                        'bounds': p.get('bounds'),
                    } for p in patch_names]
                    processor.split_generator.generate_las_splits(
                        patch_info, rasters_dir, resolution_str, patch_size_str,
                        test_regions_gdf=test_regions_gdf)
            else:
                aoi_gdf, test_regions_gdf = _load_aoi_and_test_regions(
                    args, processor.config['paths'])

                processor.process_all(chm_only=args.chm_only,
                                     save_rotated=args.save_rotated,
                                     aoi_gdf=aoi_gdf,
                                     test_regions_gdf=test_regions_gdf)

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
