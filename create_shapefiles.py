"""
Interactive shapefile creation for AOI masks and test plot regions.

Usage:
  Config-driven (processes all file_label_pairs from config):
    python create_shapefiles.py
    python create_shapefiles.py --config configs/create_shapefiles.yml

  Single-file via subcommands:
    python create_shapefiles.py aoi --raster chm.tif --labels labels.shp --save aoi.shp
    python create_shapefiles.py test-plots --raster chm.tif --labels labels.shp --save test.shp
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, CheckButtons
import rasterio
from matplotlib.patches import Patch
from shapely.geometry import Point

import sys
sys.path.insert(0, str(Path(__file__).parent))
from modules.aoi_generator import AOIGenerator
from modules.test_plot_generator import TestPlotGenerator

DEFAULT_CONFIG = Path(__file__).parent / 'configs' / 'create_shapefiles.yml'

# Shared rendering constants
RASTER_CMAP = 'Greens'
RASTER_VMIN = 0
RASTER_VMAX = 30
RASTER_ALPHA = 0.7


def _load_config(config_path=None):
    """Load config from file, falling back to defaults."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_raster(raster_path):
    """Load raster data, extent, and CRS."""
    with rasterio.open(raster_path) as src:
        raster = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        crs = src.crs
    return raster, extent, crs


def _draw_raster(ax, raster, extent, labels_gdf=None):
    """Draw base raster and optional label overlay."""
    ax.imshow(raster, extent=extent, origin='upper',
              cmap=RASTER_CMAP, vmin=RASTER_VMIN, vmax=RASTER_VMAX, alpha=RASTER_ALPHA)
    if labels_gdf is not None:
        labels_gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.3)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')


def _create_editor_layout(figsize=(14, 14)):
    """Create the shared figure layout with map and side panel."""
    fig, (ax, ax_panel) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={'width_ratios': [4, 1]})
    ax_panel.set_axis_off()
    return fig, ax, ax_panel


def _add_confirm_reset_buttons(fig):
    """Add Confirm & Save and Reset buttons, return (btn_confirm, btn_reset)."""
    ax_confirm = fig.add_axes([0.82, 0.25, 0.14, 0.05])
    ax_reset = fig.add_axes([0.82, 0.18, 0.14, 0.05])
    btn_confirm = Button(ax_confirm, 'Confirm & Save', color='lightgreen', hovercolor='green')
    btn_reset = Button(ax_reset, 'Reset', color='lightyellow', hovercolor='orange')
    return btn_confirm, btn_reset


def _run_editor(fig, on_click, btn_confirm, on_confirm, btn_reset, on_reset,
                redraw, save_path, state):
    """Wire up callbacks, show the editor, and handle close-without-confirm."""
    btn_confirm.on_clicked(on_confirm)
    btn_reset.on_clicked(on_reset)
    fig.canvas.mpl_connect('button_press_event', on_click)
    redraw()
    plt.show()

    if not state['confirmed'] and save_path:
        print("Window closed without confirming — not saved.")


# =============================================================================
# AOI Editor
# =============================================================================

def aoi_editor(raster_path, labels_path, config=None,
               buffer_distance=None, max_gap_area=None, save_path=None):
    """Interactive AOI editor — click gaps to toggle fill/unfill."""
    labels_gdf = gpd.read_file(labels_path)
    raster, extent, raster_crs = _load_raster(raster_path)

    gen = AOIGenerator(config or {})
    buf = buffer_distance if buffer_distance is not None else gen.buffer_distance
    gap = max_gap_area if max_gap_area is not None else gen.max_gap_area

    aoi_gdf = gen.generate(labels_gdf, buffer_distance=buf, max_gap_area=gap, crs=raster_crs)
    raw_aoi = gen.generate(labels_gdf, buffer_distance=buf, max_gap_area=0, crs=raster_crs)
    all_gaps = gen.get_gaps(raw_aoi)

    fig, ax, ax_panel = _create_editor_layout()
    btn_confirm, btn_reset = _add_confirm_reset_buttons(fig)

    state = {'aoi_gdf': aoi_gdf, 'unfilled': [], 'filled': [], 'confirmed': False}

    # Classify gaps as already filled (by auto-fill) or still open
    current_aoi = aoi_gdf.geometry.iloc[0]
    for gap_geom in all_gaps:
        if current_aoi.contains(gap_geom.centroid):
            state['filled'].append(gap_geom)
        else:
            state['unfilled'].append(gap_geom)

    def _draw_gaps(gap_list, facecolor, edgecolor, alpha, text_color):
        for g in gap_list:
            gpd.GeoDataFrame(geometry=[g]).plot(
                ax=ax, facecolor=facecolor, edgecolor=edgecolor,
                alpha=alpha, linewidth=1.5)
            ax.text(g.centroid.x, g.centroid.y, f'{g.area:.0f}m²',
                    ha='center', va='center', fontsize=8,
                    fontweight='bold', color=text_color)

    def redraw():
        ax.clear()
        aoi_geom = state['aoi_gdf'].geometry.iloc[0]

        _draw_raster(ax, raster, extent, labels_gdf)

        aoi_plot = gpd.GeoDataFrame(geometry=[aoi_geom])
        aoi_plot.plot(ax=ax, facecolor='red', edgecolor='red', alpha=0.15, linewidth=1.5)
        aoi_plot.boundary.plot(ax=ax, edgecolor='red', linewidth=1.5)

        _draw_gaps(state['unfilled'], 'yellow', 'orange', 0.5, 'darkorange')
        _draw_gaps(state['filled'], 'lime', 'green', 0.3, 'darkgreen')

        ax.set_title(
            f'Left-click: toggle gap fill/unfill\n'
            f'Yellow = unfilled ({len(state["unfilled"])}) | '
            f'Green = filled ({len(state["filled"])}) | '
            f'AOI: {aoi_geom.area:.0f} m²',
            fontsize=11)

        ax_panel.clear()
        ax_panel.set_axis_off()
        ax_panel.legend(handles=[
            Patch(facecolor='red', alpha=0.15, edgecolor='red', label='AOI'),
            Patch(facecolor='yellow', alpha=0.5, edgecolor='orange', label='Unfilled\n(click to fill)'),
            Patch(facecolor='lime', alpha=0.3, edgecolor='green', label='Filled\n(click to unfill)'),
        ], loc='upper center', fontsize=10, frameon=True, fancybox=True)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        click_pt = Point(event.xdata, event.ydata)

        for gap_geom in state['unfilled']:
            if gap_geom.contains(click_pt):
                state['aoi_gdf'] = gen.fill_gaps(state['aoi_gdf'], [gap_geom])
                state['unfilled'].remove(gap_geom)
                state['filled'].append(gap_geom)
                print(f"Filled gap: {gap_geom.area:.0f} m²")
                redraw()
                return

        for gap_geom in state['filled']:
            if gap_geom.contains(click_pt):
                state['aoi_gdf'] = gen.unfill_gaps(state['aoi_gdf'], [gap_geom])
                state['filled'].remove(gap_geom)
                state['unfilled'].append(gap_geom)
                print(f"Unfilled gap: {gap_geom.area:.0f} m²")
                redraw()
                return

    def on_confirm(event):
        if save_path:
            gen.save(state['aoi_gdf'], save_path)
            print(f"Saved AOI to: {save_path}")
        else:
            print("Confirmed. Use --save <path> to write to file.")
        state['confirmed'] = True
        plt.close(fig)

    def on_reset(event):
        state['aoi_gdf'] = aoi_gdf
        state['unfilled'].clear()
        state['filled'].clear()
        current = aoi_gdf.geometry.iloc[0]
        for g in all_gaps:
            if current.contains(g.centroid):
                state['filled'].append(g)
            else:
                state['unfilled'].append(g)
        print("Reset to initial state")
        redraw()

    _run_editor(fig, on_click, btn_confirm, on_confirm,
                btn_reset, on_reset, redraw, save_path, state)
    return state['aoi_gdf']


# =============================================================================
# Test Plot Editor
# =============================================================================

def test_plot_editor(raster_path, labels_path=None, config=None,
                     plot_width=None, plot_height=None, grid_size=None,
                     save_path=None):
    """Interactive test plot placement — click to place/remove rectangles."""
    raster, extent, raster_crs = _load_raster(raster_path)
    labels_gdf = gpd.read_file(labels_path) if labels_path else None

    gen = TestPlotGenerator(config or {})
    pw = plot_width if plot_width is not None else gen.plot_width
    ph = plot_height if plot_height is not None else gen.plot_height
    gs = grid_size if grid_size is not None else gen.grid_size

    fig, ax, ax_panel = _create_editor_layout()
    btn_confirm, btn_reset = _add_confirm_reset_buttons(fig)

    # Text input widgets
    for label_text, y_label in [('Width (m)', 0.72),
                                ('Height (m)', 0.65),
                                ('Grid (m)', 0.56)]:
        lbl_ax = fig.add_axes([0.80, y_label, 0.08, 0.02])
        lbl_ax.set_axis_off()
        lbl_ax.text(0.5, 0.5, label_text, ha='center', va='center', fontsize=10)

    ax_width = fig.add_axes([0.80, 0.69, 0.14, 0.03])
    ax_height = fig.add_axes([0.80, 0.62, 0.14, 0.03])
    ax_grid = fig.add_axes([0.80, 0.53, 0.14, 0.03])
    ax_snap = fig.add_axes([0.80, 0.46, 0.14, 0.05])
    ax_show_grid = fig.add_axes([0.80, 0.40, 0.14, 0.05])

    tb_width = TextBox(ax_width, '', initial=str(pw))
    tb_height = TextBox(ax_height, '', initial=str(ph))
    tb_grid = TextBox(ax_grid, '', initial=str(gs))
    chk_snap = CheckButtons(ax_snap, ['Snap to grid'], [False])
    chk_grid_vis = CheckButtons(ax_show_grid, ['Show grid'], [True])

    state = {
        'plots': [],
        'width': pw, 'height': ph, 'grid_size': gs,
        'snap': False, 'show_grid': True, 'confirmed': False,
    }

    origin_x, origin_y = extent[0], extent[2]

    def redraw():
        ax.clear()
        _draw_raster(ax, raster, extent, labels_gdf)

        if state['show_grid'] and state['grid_size'] > 0:
            g = state['grid_size']
            for x in np.arange(origin_x, extent[1], g):
                ax.axvline(x, color='gray', alpha=0.3, linewidth=0.5)
            for y in np.arange(origin_y, extent[3], g):
                ax.axhline(y, color='gray', alpha=0.3, linewidth=0.5)

        for i, p in enumerate(state['plots']):
            geom = p['geometry']
            gpd.GeoDataFrame(geometry=[geom]).plot(
                ax=ax, facecolor='red', edgecolor='darkred', alpha=0.3, linewidth=2)
            ax.text(geom.centroid.x, geom.centroid.y, f'{i+1}',
                    ha='center', va='center', fontsize=10,
                    fontweight='bold', color='darkred')

        snap_str = ' [SNAP]' if state['snap'] else ''
        ax.set_title(
            f'Left-click: place plot | Right-click: remove plot\n'
            f'Plot: {state["width"]}m x {state["height"]}m{snap_str} | '
            f'{len(state["plots"])} placed',
            fontsize=11)

        ax_panel.clear()
        ax_panel.set_axis_off()
        legend = [
            Patch(facecolor='green', alpha=0.5, label='Raster'),
            Patch(facecolor='red', alpha=0.3, edgecolor='darkred', label='Test plot'),
        ]
        if labels_gdf is not None:
            legend.insert(1, Patch(facecolor='none', edgecolor='blue', label='Labels'))
        ax_panel.legend(handles=legend, loc='upper center', fontsize=10,
                        frameon=True, fancybox=True)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        x, y = event.xdata, event.ydata

        if event.button == 3:
            click_pt = Point(x, y)
            for p in reversed(state['plots']):
                if p['geometry'].contains(click_pt):
                    state['plots'].remove(p)
                    print(f"Removed plot at ({x:.1f}, {y:.1f})")
                    redraw()
                    return
            return

        if event.button != 1:
            return

        if state['snap'] and state['grid_size'] > 0:
            x, y = gen.snap_to_grid(x, y, state['grid_size'], origin_x, origin_y)

        w, h = state['width'], state['height']
        geom = gen.create_plot(x, y, w, h)
        state['plots'].append({'geometry': geom, 'width': w, 'height': h})
        print(f"Placed plot {len(state['plots'])} at ({x:.1f}, {y:.1f}) — {w}m x {h}m")
        redraw()

    def _update_float(key, val, trigger_redraw=False):
        try:
            state[key] = float(val)
            if trigger_redraw:
                redraw()
        except ValueError:
            pass

    tb_width.on_submit(lambda v: _update_float('width', v))
    tb_height.on_submit(lambda v: _update_float('height', v))
    tb_grid.on_submit(lambda v: _update_float('grid_size', v, trigger_redraw=True))

    def on_snap(label):
        state['snap'] = not state['snap']
        print(f"Snap to grid: {'ON' if state['snap'] else 'OFF'}")
        redraw()

    def on_grid_vis(label):
        state['show_grid'] = not state['show_grid']
        redraw()

    chk_snap.on_clicked(on_snap)
    chk_grid_vis.on_clicked(on_grid_vis)

    def on_confirm(event):
        if save_path and state['plots']:
            plots_gdf = gen.plots_to_geodataframe(state['plots'], crs=raster_crs)
            gen.save(plots_gdf, save_path)
            print(f"Saved {len(state['plots'])} test plot(s) to: {save_path}")
        elif not state['plots']:
            print("No plots placed — nothing to save.")
        else:
            print("Confirmed. Use --save <path> to write to file.")
        state['confirmed'] = True
        plt.close(fig)

    def on_reset(event):
        state['plots'].clear()
        print("Cleared all plots")
        redraw()

    _run_editor(fig, on_click, btn_confirm, on_confirm,
                btn_reset, on_reset, redraw, save_path, state)
    return state['plots']


# =============================================================================
# Config-driven batch mode
# =============================================================================

def run_from_config(config):
    """Process all file_label_pairs from config, running the appropriate editors."""
    mode = config.get('mode', 'both')
    pairs = config.get('file_label_pairs', [])
    aoi_dir = Path(config.get('aoi_output_dir', 'data/shapefiles/aoi'))
    test_dir = Path(config.get('test_regions_output_dir', 'data/shapefiles/test_regions'))

    if not pairs:
        print("No file_label_pairs defined in config. Nothing to do.")
        return

    run_aoi = mode in ('aoi', 'both')
    run_test = mode in ('test-plots', 'both')

    for pair in pairs:
        raster_path = pair['raster']
        labels_path = pair.get('labels')
        plot_name = Path(raster_path).stem

        print(f"\n{'='*60}")
        print(f"Processing: {plot_name}")
        print(f"{'='*60}")

        if run_aoi:
            if not labels_path:
                print(f"Skipping AOI for {plot_name} — no labels provided")
            else:
                print(f"\n--- AOI Editor: {plot_name} ---")
                aoi_editor(
                    raster_path=raster_path,
                    labels_path=labels_path,
                    config=config,
                    save_path=str(aoi_dir / f"{plot_name}_aoi.shp"),
                )

        if run_test:
            print(f"\n--- Test Plot Editor: {plot_name} ---")
            test_plot_editor(
                raster_path=raster_path,
                labels_path=labels_path,
                config=config,
                save_path=str(test_dir / f"{plot_name}_test_regions.shp"),
            )


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Interactive shapefile creation for AOI masks and test plot regions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage modes:

  Config-driven (processes all file_label_pairs from config):
    python create_shapefiles.py
    python create_shapefiles.py --config configs/create_shapefiles.yml

  Single-file (override raster/labels via CLI):
    python create_shapefiles.py aoi --raster chm.tif --labels labels.shp --save aoi.shp
    python create_shapefiles.py test-plots --raster chm.tif --labels labels.shp --save test.shp
        """
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config (default: configs/create_shapefiles.yml)')
    sub = parser.add_subparsers(dest='command', required=False)

    aoi_p = sub.add_parser('aoi', help='Generate AOI from labels with interactive gap editing')
    aoi_p.add_argument('--raster', type=str, required=True, help='Path to GeoTIFF raster')
    aoi_p.add_argument('--labels', type=str, required=True, help='Path to labels shapefile')
    aoi_p.add_argument('--buffer', type=float, default=None,
                       help='Buffer distance in meters (overrides config)')
    aoi_p.add_argument('--max-gap-area', type=float, default=None,
                       help='Auto-fill gaps smaller than this area in m² (overrides config)')
    aoi_p.add_argument('--save', type=str, default=None, help='Save AOI to shapefile')

    tp_p = sub.add_parser('test-plots', help='Place test plot regions interactively')
    tp_p.add_argument('--raster', type=str, required=True, help='Path to GeoTIFF raster')
    tp_p.add_argument('--labels', type=str, default=None,
                      help='Path to labels shapefile (optional overlay)')
    tp_p.add_argument('--plot-width', type=float, default=None,
                      help='Initial plot width in meters (overrides config)')
    tp_p.add_argument('--plot-height', type=float, default=None,
                      help='Initial plot height in meters (overrides config)')
    tp_p.add_argument('--grid-size', type=float, default=None,
                      help='Initial grid size in meters (overrides config)')
    tp_p.add_argument('--save', type=str, default=None, help='Save test plots to shapefile')

    args = parser.parse_args()
    config = _load_config(args.config)

    if args.command is None:
        run_from_config(config)
    elif args.command == 'aoi':
        aoi_editor(
            raster_path=args.raster, labels_path=args.labels,
            config=config, buffer_distance=args.buffer,
            max_gap_area=args.max_gap_area, save_path=args.save,
        )
    elif args.command == 'test-plots':
        test_plot_editor(
            raster_path=args.raster, labels_path=args.labels,
            config=config, plot_width=args.plot_width,
            plot_height=args.plot_height, grid_size=args.grid_size,
            save_path=args.save,
        )


if __name__ == '__main__':
    main()
