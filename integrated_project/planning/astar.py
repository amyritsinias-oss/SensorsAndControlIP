
import heapq

import numpy as np

# 8connected moves
MOVES = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2))]


def astar(blocked, start, goal):
    #A* on a boolean grid (True = blocked). start, goal = (row, col). Returns a list of cells or None.
    ny, nx = blocked.shape

    def h(cell):     # straight-line distance, never overestimates
        return np.hypot(cell[0] - goal[0], cell[1] - goal[1])

    open_set = [(h(start), 0.0, start)]
    g = {start: 0.0}
    came_from = {}
    while open_set:
        _, g_cur, cur = heapq.heappop(open_set)
        if cur == goal:       # walk back to the start
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]
        if g_cur > g[cur]:   # stale queue entry
            continue
        for dr, dc, cost in MOVES:
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < ny and 0 <= nb[1] < nx) or blocked[nb]:
                continue
            if dr and dc and (blocked[cur[0] + dr, cur[1]] or blocked[cur[0], cur[1] + dc]):
                continue    #don't cut diagonally past a blocked corner
            g_new = g_cur + cost
            if g_new < g.get(nb, np.inf):
                g[nb] = g_new
                came_from[nb] = cur
                heapq.heappush(open_set, (g_new + h(nb), g_new, nb))
    return None


def nearest_free(blocked, cell):
    #Closest unblocked cell, in case the start or goal sits inside the inflated zone.
    if not blocked[cell]:
        return cell
    free = np.argwhere(~blocked)
    d = np.hypot(free[:, 0] - cell[0], free[:, 1] - cell[1])
    return tuple(free[np.argmin(d)])


def line_is_free(blocked, a, b):
    #True if the straight line between cells a and b crosses no blocked cell.
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
    rows = np.round(np.linspace(a[0], b[0], n)).astype(int)
    cols = np.round(np.linspace(a[1], b[1], n)).astype(int)
    return not blocked[rows, cols].any()


def prune(blocked, cells):
    #Keep only the cells needed so that each straight segment stays clear, line of sight only
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not line_is_free(blocked, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def plan(grid, start_xy, goal_xy, inflate=0.30):
    """Plan from start_xy to goal_xy on the occupancy grid.

    Returns (waypoints, cells): waypoints is an (N, 2) array in metres from start to goal,
    cells is the raw A* path (for plotting). Returns (None, None) if no path exists.
    Unknown cells count as free, the robot replans when it sees more.
    """
    blocked = grid.inflated(inflate)
    start = nearest_free(blocked, tuple(int(v) for v in grid.to_cell(*start_xy)))
    goal = nearest_free(blocked, tuple(int(v) for v in grid.to_cell(*goal_xy)))
    cells = astar(blocked, start, goal)
    if cells is None:
        return None, None
    keep = prune(blocked, cells)
    xs, ys = grid.to_world(*np.array(keep).T)
    waypoints = np.column_stack([xs, ys])
    waypoints[0], waypoints[-1] = start_xy, goal_xy  # exact start / goal, not cell centres
    return waypoints, cells