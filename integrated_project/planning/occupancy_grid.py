# download scipy in terminal by running" python -m pip install scipy "
import numpy as np
from scipy.ndimage import binary_dilation


class OccupancyGrid:
    #default: the 5 x 4 m room plus a 0.25 m margin so wall hits land inside the grid
    def __init__(self, x_min=-0.25, y_min=-0.25, width=5.5, height=4.5, res=0.05,
                 l_occ=0.85, l_free=0.4, l_min=-4.0, l_max=4.0):
        self.x_min, self.y_min, self.res = x_min, y_min, res
        self.nx, self.ny = int(round(width / res)), int(round(height / res))
        self.log_odds = np.zeros((self.ny, self.nx))      
        self.l_occ, self.l_free = l_occ, l_free
        self.l_min, self.l_max = l_min, l_max

    #coordinate conversion
    def to_cell(self, x, y):
        """World (x, y) [m] -> (row, col) indices."""
        col = np.floor((np.asarray(x) - self.x_min) / self.res).astype(int)
        row = np.floor((np.asarray(y) - self.y_min) / self.res).astype(int)
        return row, col

    def to_world(self, row, col):
        """(row, col) -> world (x, y) of the cell centre."""
        return (self.x_min + (np.asarray(col) + 0.5) * self.res,
                self.y_min + (np.asarray(row) + 0.5) * self.res)

    def in_bounds(self, row, col):
        return (row >= 0) & (row < self.ny) & (col >= 0) & (col < self.nx)

    #map update
    def update(self, lidar, ranges, pose):
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        origin = np.array([x, y]) + R @ lidar.offset  # sensor position in the world

        hit = np.isfinite(ranges)
        r = np.where(hit, ranges, lidar.range_max)  # misses = free out to max the range
        ang = lidar.angles + th
        ends = origin + np.column_stack([r * np.cos(ang), r * np.sin(ang)])

        free_cells = set()
        for (ex, ey), length in zip(ends, r):
            n = max(int(length / (self.res / 2)), 1)  # sample every half cell
            t = np.linspace(0, 1, n, endpoint=False)
            rows, cols = self.to_cell(origin[0] + t * (ex - origin[0]), origin[1] + t * (ey - origin[1]))
            ok = self.in_bounds(rows, cols)
            free_cells.update(zip(rows[ok], cols[ok]))

        hr, hc = self.to_cell(ends[hit, 0], ends[hit, 1])
        ok = self.in_bounds(hr, hc)
        hit_cells = set(zip(hr[ok], hc[ok]))
        free_cells -= hit_cells  # never clear a cell that was hit

        if free_cells:
            fr, fc = np.array(list(free_cells)).T
            self.log_odds[fr, fc] -= self.l_free
        if hit_cells:
            hr, hc = np.array(list(hit_cells)).T
            self.log_odds[hr, hc] += self.l_occ
        np.clip(self.log_odds, self.l_min, self.l_max, out=self.log_odds)

    
    def probability(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def occupied(self, threshold=0.65):
        return self.probability() > threshold

    def inflated(self, radius=0.25, threshold=0.65):
        k = int(np.ceil(radius / self.res))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        disk = (xx ** 2 + yy ** 2) <= k ** 2
        return binary_dilation(self.occupied(threshold), structure=disk)