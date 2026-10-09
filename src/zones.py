"""Open-field zones on the floor plane (cm), ANY-maze style: a square grid plus zones built from it.

Centre: a central rectangle (percent of the arena side, or a distance from the walls), with grid squares the
user clicked toggled in or out of it. Wall zone: the arena minus the centre. Corners: a square in each corner
whose side is a fraction of the arena side (DLCAnalyzer: 20%). Zones are stored as boolean rasters so any
shape made of rectangles and toggled squares is handled the same way.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

RES_CM = 0.25  # raster cell size


@dataclass
class ZoneSpec:
    width_cm: float = 40.0
    height_cm: float = 40.0
    grid_cm: float = 10.0
    center_mode: str = "percent"   # "percent" of the arena side, or "cm" from the walls
    center_value: float = 50.0
    toggled_cells: list = field(default_factory=list)  # [col, row] grid squares flipped in/out of the centre
    corners: bool = True
    corner_fraction: float = 0.2

    def __post_init__(self):
        nx = int(round(self.width_cm / RES_CM))
        ny = int(round(self.height_cm / RES_CM))
        xs = (np.arange(nx) + 0.5) * RES_CM  # raster cell centres
        ys = (np.arange(ny) + 0.5) * RES_CM
        X, Y = np.meshgrid(xs, ys)

        if self.center_mode == "percent":
            mx = self.width_cm * (1 - self.center_value / 100) / 2
            my = self.height_cm * (1 - self.center_value / 100) / 2
        else:
            mx = my = self.center_value
        center = (X >= mx) & (X <= self.width_cm - mx) & (Y >= my) & (Y <= self.height_cm - my)
        for col, row in self.toggled_cells:
            cell = self._cell_mask(X, Y, col, row)
            center = np.where(cell, ~center, center)

        self.masks = {"centre": center, "wall": ~center}
        if self.corners:
            cw, ch = self.width_cm * self.corner_fraction, self.height_cm * self.corner_fraction
            near_x = (X <= cw) | (X >= self.width_cm - cw)
            near_y = (Y <= ch) | (Y >= self.height_cm - ch)
            self.masks["corners"] = near_x & near_y

    def _cell_mask(self, X, Y, col, row):
        g = self.grid_cm
        return (X >= col * g) & (X < (col + 1) * g) & (Y >= row * g) & (Y < (row + 1) * g)

    @property
    def grid_shape(self):
        """(columns, rows) of the grid; the last column/row may be partial."""
        return int(np.ceil(self.width_cm / self.grid_cm - 1e-9)), int(np.ceil(self.height_cm / self.grid_cm - 1e-9))

    def _raster_index(self, xy_cm):
        xy = np.asarray(xy_cm, float).reshape(-1, 2)
        ix = np.clip((xy[:, 0] / RES_CM).astype(int), 0, self.masks["centre"].shape[1] - 1)
        iy = np.clip((xy[:, 1] / RES_CM).astype(int), 0, self.masks["centre"].shape[0] - 1)
        return ix, iy

    def in_zone(self, name, xy_cm):
        """Per point: is it in the zone? Points outside the floor (e.g. rearing on a wall) are clipped onto it;
        NaN positions return False."""
        xy = np.asarray(xy_cm, float).reshape(-1, 2)
        valid = ~np.isnan(xy).any(axis=1)
        ix, iy = self._raster_index(np.nan_to_num(xy))
        return self.masks[name][iy, ix] & valid

    def grid_cell(self, xy_cm):
        """Grid square index per point (col + row * n_cols), -1 for NaN; points are clipped onto the floor."""
        xy = np.asarray(xy_cm, float).reshape(-1, 2)
        valid = ~np.isnan(xy).any(axis=1)
        ncol, nrow = self.grid_shape
        col = np.clip((np.nan_to_num(xy[:, 0]) // self.grid_cm).astype(int), 0, ncol - 1)
        row = np.clip((np.nan_to_num(xy[:, 1]) // self.grid_cm).astype(int), 0, nrow - 1)
        return np.where(valid, col + row * ncol, -1)

    def outlines_cm(self, name):
        """Zone outline polygons in cm (for drawing), traced from the raster."""
        m = self.masks[name].astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [(c.reshape(-1, 2) * RES_CM).tolist() for c in cnts]

    def grid_lines_cm(self):
        """Grid line segments [[x1, y1], [x2, y2]] in cm."""
        lines = [[[x, 0.0], [x, self.height_cm]] for x in np.arange(self.grid_cm, self.width_cm - 1e-6, self.grid_cm)]
        lines += [[[0.0, y], [self.width_cm, y]] for y in np.arange(self.grid_cm, self.height_cm - 1e-6, self.grid_cm)]
        return [[list(map(float, a)), list(map(float, b))] for a, b in lines]

    def drawing(self):
        """Zone polygons (cm) to draw: name -> list of polygons."""
        out = {"centre": self.outlines_cm("centre")}
        if "corners" in self.masks:
            out["corners"] = self.outlines_cm("corners")
        return out

    @property
    def extent(self):
        """x0, x1, y0 (bottom), y1 (top) for imshow."""
        return 0.0, self.width_cm, self.height_cm, 0.0
