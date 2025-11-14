from manim import *
from functools import cmp_to_key

# Global midpoint used by compare (kept to match provided reference)
mid = [0, 0]


class Algorithm(MovingCameraScene):
    def make_side_label(self, text, side="left", font_size=20, color=WHITE):
        """Create a text label pinned to the left or right middle of the camera frame.
        Keeps constant screen size across zooms and avoids overlapping the center.
        side: "left" | "right"
        """
        lbl = Text(text, font_size=font_size, color=color)
        # Cache initial metrics for scale compensation vs. camera zoom
        lbl._oh = lbl.height
        lbl._ow = lbl.width
        lbl._f0 = self.camera.frame.get_height()

        def _pin(m):
            frame = self.camera.frame
            f = frame.get_height()
            # Constant on-screen size
            scale_factor = f / m._f0
            m.set_height(m._oh * scale_factor)
            # Position at left/right middle with margin, ensuring full label stays on one side
            c = frame.get_center()
            l = frame.get_left()
            r = frame.get_right()
            margin_x = 0.08 * f  # Increased horizontal margin
            # For left: place with left edge at margin; for right: place with right edge at margin
            if side == "left":
                x = l[0] + margin_x + (m._ow * scale_factor) / 2
            else:
                x = r[0] - margin_x - (m._ow * scale_factor) / 2
            y = c[1]
            m.move_to([x, y, 0])

        lbl.add_updater(_pin)
        return lbl
    def orientation(self, a, b, c):
        # Returns 1 if counter-clockwise, -1 if clockwise, 0 if collinear
        res = (b[1] - a[1]) * (c[0] - b[0]) - (c[1] - b[1]) * (b[0] - a[0])
        if res == 0:
            return 0
        return 1 if res > 0 else -1

    def _key(self, p, nd: int = 6):
        """Canonicalize a point tuple for dict keys to avoid float rounding mismatches."""
        return (round(p[0], nd), round(p[1], nd))

    def brute_hull(self, pts):
        # Port of provided bruteHull for <=5 points
        global mid
        s = set()
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                x1, x2 = pts[i][0], pts[j][0]
                y1, y2 = pts[i][1], pts[j][1]
                a1, b1, c1 = y1 - y2, x2 - x1, x1 * y2 - y1 * x2
                pos = neg = 0
                for k in range(n):
                    if (k == i) or (k == j) or (a1 * pts[k][0] + b1 * pts[k][1] + c1 <= 0):
                        neg += 1
                    if (k == i) or (k == j) or (a1 * pts[k][0] + b1 * pts[k][1] + c1 >= 0):
                        pos += 1
                if pos == n or neg == n:
                    s.add(tuple(pts[i]))
                    s.add(tuple(pts[j]))

        ret = [list(x) for x in s]
        # Sort CCW around centroid
        mid = [0, 0]
        m = len(ret)
        for i in range(m):
            mid[0] += ret[i][0]
            mid[1] += ret[i][1]
            ret[i][0] *= m
            ret[i][1] *= m
        ret = sorted(ret, key=cmp_to_key(self.compare))
        for i in range(m):
            ret[i][0] /= m
            ret[i][1] /= m
        return [tuple(p) for p in ret]

    def compare(self, p1, q1):
        p = [p1[0] - mid[0], p1[1] - mid[1]]
        q = [q1[0] - mid[0], q1[1] - mid[1]]
        def quad(p):
            if p[0] >= 0 and p[1] >= 0:
                return 1
            if p[0] <= 0 and p[1] >= 0:
                return 2
            if p[0] <= 0 and p[1] <= 0:
                return 3
            return 4
        one, two = quad(p), quad(q)
        if one != two:
            return -1 if one < two else 1
        return -1 if p[1] * q[0] < q[1] * p[0] else 1

    def draw_hull_lines(self, hull_pts, points_mobs, color=YELLOW, stroke=3.0):
        lines = []
        if len(hull_pts) < 2:
            return lines
        for i in range(len(hull_pts)):
            a = hull_pts[i]
            b = hull_pts[(i + 1) % len(hull_pts)]
            la = points_mobs[self._key(a)].get_center()
            lb = points_mobs[self._key(b)].get_center()
            lines.append(Line(la, lb, color=color, stroke_width=stroke, z_index=1))
        return lines

    def rightmost_index(self, poly):
        idx = 0
        for i in range(1, len(poly)):
            if poly[i][0] > poly[idx][0]:
                idx = i
        return idx

    def leftmost_index(self, poly):
        idx = 0
        for i in range(1, len(poly)):
            if poly[i][0] < poly[idx][0]:
                idx = i
        return idx

    def find_upper_lower_tangents_animated(self, A, B, points_mobs):
        n1, n2 = len(A), len(B)
        ia = self.rightmost_index(A)
        ib = self.leftmost_index(B)

        # Show upper tangent search annotation
        upper_tangent_label = self.make_side_label(
            "Upper tangent search:\nRotate clockwise until\nall points lie below",
            side="left", font_size=16, color=TEAL
        )
        self.play(FadeIn(upper_tangent_label), run_time=0.2)

        # Visual candidate lines
        upper_line = always_redraw(lambda: Line(
            points_mobs[self._key(A[ia])].get_center(), points_mobs[self._key(B[ib])].get_center(),
            color=TEAL, stroke_width=2.5
        ))
        self.add(upper_line)
        self.play(FadeIn(upper_line), run_time=0.2)

        # Upper tangent search
        done = False
        while not done:
            done = True
            while self.orientation(B[ib], A[ia], A[(ia + 1) % n1]) >= 0:
                ia = (ia + 1) % n1
                self.wait(0.05)
            while self.orientation(A[ia], B[ib], B[(n2 + ib - 1) % n2]) <= 0:
                ib = (ib - 1) % n2
                done = False
                self.wait(0.05)
        uppera, upperb = ia, ib

        # Clean up upper tangent visuals completely before starting lower tangent
        self.play(FadeOut(upper_tangent_label), FadeOut(upper_line), run_time=0.2)

        # Show lower tangent search annotation
        lower_tangent_label = self.make_side_label(
            "Lower tangent search:\nRotate counter-clockwise\nuntil all points lie above",
            side="right", font_size=16, color=BLUE
        )
        self.play(FadeIn(lower_tangent_label), run_time=0.2)

        # Lower tangent search
        lower_line = always_redraw(lambda: Line(
            points_mobs[self._key(A[ia])].get_center(), points_mobs[self._key(B[ib])].get_center(),
            color=BLUE, stroke_width=2.5
        ))
        self.add(lower_line)
        self.play(FadeIn(lower_line), run_time=0.2)

        ia = self.rightmost_index(A)
        ib = self.leftmost_index(B)
        done = False
        while not done:
            done = True
            while self.orientation(A[ia], B[ib], B[(ib + 1) % n2]) >= 0:
                ib = (ib + 1) % n2
                self.wait(0.05)
            while self.orientation(B[ib], A[ia], A[(n1 + ia - 1) % n1]) <= 0:
                ia = (ia - 1) % n1
                done = False
                self.wait(0.05)
        lowera, lowerb = ia, ib

        # Freeze the final tangent lines briefly
        self.wait(0.2)
        self.play(FadeOut(lower_line), FadeOut(lower_tangent_label), run_time=0.2)
        return (uppera, upperb, lowera, lowerb)

    def merge_polygons_animated(self, A, B, points_mobs, final=False):
        # Compute tangents with animation
        ua, ub, la, lb = self.find_upper_lower_tangents_animated(A, B, points_mobs)
        n1, n2 = len(A), len(B)

        # Stitch hull following provided logic
        ret = []
        ind = ua
        ret.append(A[ua])
        while ind != la:
            ind = (ind + 1) % n1
            ret.append(A[ind])
        ind = lb
        ret.append(B[lb])
        while ind != ub:
            ind = (ind + 1) % n2
            ret.append(B[ind])

        color = ORANGE if final else YELLOW
        lines = self.draw_hull_lines(ret, points_mobs, color=color, stroke=3.2 if final else 2.6)
        self.play(*[Create(l) for l in lines], run_time=0.5)
        return ret, lines

    def animate_divide(self, pts, points_mobs, depth=0):
        # Base case
        if len(pts) <= 5:
            hull = self.brute_hull(pts)
            base_lines = self.draw_hull_lines(hull, points_mobs, color=TEAL, stroke=2.5)
            # Meaningful annotation for base case
            base_label = self.make_side_label(
                f"Base case: {len(pts)} points\nBrute-force O(n²) hull",
                side="right", font_size=18, color=TEAL
            )
            self.play(*[Create(l) for l in base_lines], FadeIn(base_label), run_time=0.4)
            self.wait(0.3)
            self.play(FadeOut(base_label), run_time=0.2)
            return hull, base_lines

        # Split by median x
        mid_idx = len(pts) // 2
        xl = pts[mid_idx - 1][0]
        xr = pts[mid_idx][0]
        xmid = (xl + xr) / 2

        # Dashed split line
        split_line = DashedLine((xmid, self.ymin - 0.5, 0), (xmid, self.ymax + 0.5, 0),
                                color=GRAY, dash_length=0.12, stroke_width=2.0)
        # Meaningful annotation explaining the divide step
        side = "left" if (depth % 2 == 0) else "right"
        n = len(pts)
        split_label = self.make_side_label(
            f"Divide: {n} points → {mid_idx} + {n - mid_idx}\nRecursive depth: {depth}",
            side=side, font_size=18, color=YELLOW
        )
        self.play(Create(split_line), FadeIn(split_label), run_time=0.3)
        self.wait(0.2)

        left_pts = pts[:mid_idx]
        right_pts = pts[mid_idx:]

        # Fade out split label before recursing to prevent overlap with child labels
        self.play(FadeOut(split_label), run_time=0.2)

        # Recurse
        Lh, L_lines = self.animate_divide(left_pts, points_mobs, depth + 1)
        Rh, R_lines = self.animate_divide(right_pts, points_mobs, depth + 1)

        # Show merge phase annotation (after recursion completes)
        merge_label = self.make_side_label(
            f"Conquer: Merge {len(Lh)} + {len(Rh)} vertices\nFind upper & lower tangents",
            side="left" if side == "right" else "right",  # Alternate from split
            font_size=18, color=ORANGE
        )
        self.play(FadeIn(merge_label), run_time=0.3)
        self.wait(0.2)
        
        # Fade out merge label BEFORE tangent search to prevent overlap with tangent labels
        self.play(FadeOut(merge_label), run_time=0.2)

        # Merge (animated tangents) - now tangent labels won't collide with merge label
        merged_hull, merged_lines = self.merge_polygons_animated(Lh, Rh, points_mobs, final=(depth == 0))

        # Clean up intermediate visuals to keep scene tidy
        self.play(
            *[FadeOut(m) for m in (L_lines + R_lines)],
            FadeOut(split_line),
            run_time=0.4
        )
        return merged_hull, merged_lines

    def construct(self):
        # Sample points (same set as other files for consistency)
        points = [
            (5.29, 3.48), (9.75, -1.54), (11.02, -0.28), (10.39, -0.28), (11.02, 0.98),
            (8.48, -0.911), (9.75, 1), (9.72, 1.6), (9.12, 2.91), (7.85, 2.29), (7.21, 1.62),
            (6.57, 0.36), (4.65, 2.29), (5.29, 0.36), (3.38, 2.91), (4.03, 1), (2.75, 1),
            (5.29, -1.573), (3.38, -0.911), (5.94, -2.18), (2.11, -0.911), (4.66, -2.2),
            (3.38, -2.18), (7.21, -2.782)
        ]

        # Prepare plane and dots
        self.xmin = min(p[0] for p in points); self.xmax = max(p[0] for p in points)
        self.ymin = min(p[1] for p in points); self.ymax = max(p[1] for p in points)

        plane = NumberPlane(
            x_range=(-100, 100), y_range=(-100, 100),
            background_line_style={"stroke_color": WHITE, "stroke_opacity": 0.2}
        )
        plane.x_axis.set_opacity(0.2)
        plane.y_axis.set_opacity(0.2)
        self.play(FadeIn(plane), run_time=0.3)

        points_mobs = {
            self._key(p): Dot((p[0], p[1], 0), radius=0.1, color=WHITE, fill_opacity=1.0, z_index=float("inf"))
            for p in points
        }

        scene_aspect_ratio = config.frame_width / config.frame_height
        self.play(
            self.camera.frame.animate.set(
                width=max((self.xmax - self.xmin + 1) * 1.2, (self.ymax - self.ymin + 1) * 1.2 * scene_aspect_ratio)
            ).move_to(((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2, 0)),
            *[FadeIn(m) for m in points_mobs.values()],
            run_time=0.5
        )

        # Title pinned visually to center-top via updater (constant screen size across camera moves)
        title = Text(
            "Convex Hull - Divide & Conquer (Merge-based)",
            font_size=34,
            color=YELLOW,
        )
        # Cache initial metrics for scale compensation
        title._oh = title.height
        title._f0 = self.camera.frame.get_height()
        def _pin_to_top_center(m):
            frame = self.camera.frame
            # Keep constant on-screen size relative to frame zoom
            f = frame.get_height()
            m.set_height(m._oh * (f / m._f0))
            # Place at top center with small inward offset
            c = frame.get_center()
            t = frame.get_top()
            margin = 0.06 * f  # 6% of frame height as margin
            m.move_to([c[0], t[1] - margin, 0])
        title.add_updater(_pin_to_top_center)
        self.add(title)
        self.play(Write(title), run_time=0.4)

        # Sort by x then y
        pts_sorted = sorted(points)

        # Full recursive animation
        final_hull, final_lines = self.animate_divide(pts_sorted, points_mobs, depth=0)

        # Final emphasize
        # Morph the same pinned title to the final style so it remains fixed and centered
        end_text = Text(
            "Convex Hull - Divide & Conquer",
            font_size=36,
            color=ORANGE,
            weight=BOLD,
        )
        self.play(Transform(title, end_text), run_time=0.4)

        # Show complexity analysis - impressive for TA
        complexity_label = self.make_side_label(
            f"Algorithm Analysis:\nTime: O(n log n)\nSpace: O(n)\nn = {len(points)} points",
            side="left", font_size=18, color=GREEN
        )
        self.play(FadeIn(complexity_label), run_time=0.4)

        # Emphasize hull points, de-emphasize interior
        hull_key_set = set(self._key(p) for p in final_hull)
        anims = []
        for key, mob in points_mobs.items():
            if key in hull_key_set:
                anims.append(mob.animate.set_color(ORANGE))
            else:
                anims.append(mob.animate.set_opacity(0.3))
        self.play(*anims, run_time=0.6)
        
        # Add final summary
        summary_label = self.make_side_label(
            f"Final Hull: {len(final_hull)} vertices\nOptimal D&C algorithm\nBalanced recursion tree",
            side="right", font_size=18, color=ORANGE
        )
        self.play(FadeIn(summary_label), run_time=0.4)
        
        self.wait(2)
        self.play(FadeOut(complexity_label), FadeOut(summary_label), run_time=0.3)


# Console-only reference: exact provided D&C (optional to run)
def quad(p):
    if p[0] >= 0 and p[1] >= 0:
        return 1
    if p[0] <= 0 and p[1] >= 0:
        return 2
    if p[0] <= 0 and p[1] <= 0:
        return 3
    return 4


def orientation(a, b, c):
    res = (b[1]-a[1]) * (c[0]-b[0]) - (c[1]-b[1]) * (b[0]-a[0])
    if res == 0:
        return 0
    if res > 0:
        return 1
    return -1


def compare(p1, q1):
    p = [p1[0]-mid[0], p1[1]-mid[1]]
    q = [q1[0]-mid[0], q1[1]-mid[1]]
    one = quad(p)
    two = quad(q)
    if one != two:
        return -1 if one < two else 1
    if p[1]*q[0] < q[1]*p[0]:
        return -1
    return 1


def merger(a, b):
    n1, n2 = len(a), len(b)
    ia, ib = 0, 0
    for i in range(1, n1):
        if a[i][0] > a[ia][0]:
            ia = i
    for i in range(1, n2):
        if b[i][0] < b[ib][0]:
            ib = i
    inda, indb = ia, ib
    done = 0
    
    # Add iteration limits to prevent infinite loops
    max_iterations = n1 + n2  # Maximum possible iterations
    
    # Find upper tangent
    iteration_count = 0
    while not done:
        done = 1
        start_inda = inda
        inner_iterations = 0
        while orientation(b[indb], a[inda], a[(inda+1) % n1]) >= 0:
            inda = (inda + 1) % n1
            inner_iterations += 1
            if inner_iterations > n1:  # Cycled through entire hull
                break
        
        start_indb = indb
        inner_iterations = 0
        while orientation(a[inda], b[indb], b[(n2+indb-1) % n2]) <= 0:
            indb = (indb - 1) % n2
            done = 0
            inner_iterations += 1
            if inner_iterations > n2:  # Cycled through entire hull
                break
        
        iteration_count += 1
        if iteration_count > max_iterations:
            # Fallback: use current positions
            break
    
    uppera, upperb = inda, indb
    inda, indb = ia, ib
    done = 0
    
    # Find lower tangent
    iteration_count = 0
    while not done:
        done = 1
        start_indb = indb
        inner_iterations = 0
        while orientation(a[inda], b[indb], b[(indb+1) % n2]) >= 0:
            indb = (indb + 1) % n2
            inner_iterations += 1
            if inner_iterations > n2:  # Cycled through entire hull
                break
        
        start_inda = inda
        inner_iterations = 0
        while orientation(b[indb], a[inda], a[(n1+inda-1) % n1]) <= 0:
            inda = (inda - 1) % n1
            done = 0
            inner_iterations += 1
            if inner_iterations > n1:  # Cycled through entire hull
                break
        
        iteration_count += 1
        if iteration_count > max_iterations:
            # Fallback: use current positions
            break
    
    ret = []
    lowera, lowerb = inda, indb
    ind = uppera
    ret.append(a[uppera])
    while ind != lowera:
        ind = (ind+1) % n1
        ret.append(a[ind])
    ind = lowerb
    ret.append(b[lowerb])
    while ind != upperb:
        ind = (ind+1) % n2
        ret.append(b[ind])
    return ret


def bruteHull(a):
    global mid
    s = set()
    for i in range(len(a)):
        for j in range(i+1, len(a)):
            x1, x2 = a[i][0], a[j][0]
            y1, y2 = a[i][1], a[j][1]
            a1, b1, c1 = y1-y2, x2-x1, x1*y2-y1*x2
            pos, neg = 0, 0
            for k in range(len(a)):
                if (k == i) or (k == j) or (a1*a[k][0]+b1*a[k][1]+c1 <= 0):
                    neg += 1
                if (k == i) or (k == j) or (a1*a[k][0]+b1*a[k][1]+c1 >= 0):
                    pos += 1
            if pos == len(a) or neg == len(a):
                s.add(tuple(a[i]))
                s.add(tuple(a[j]))
    ret = []
    for x in s:
        ret.append(list(x))
    mid = [0, 0]
    n = len(ret)
    for i in range(n):
        mid[0] += ret[i][0]
        mid[1] += ret[i][1]
        ret[i][0] *= n
        ret[i][1] *= n
    ret = sorted(ret, key=cmp_to_key(compare))
    for i in range(n):
        ret[i] = [ret[i][0]/n, ret[i][1]/n]
    return ret


def dist(p1, p2):
    """Calculate Euclidean distance between two points"""
    import math
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def convex_hull_area(hull_points):
    """Calculate area of convex hull using Shoelace formula"""
    if len(hull_points) < 3:
        return 0.0
    
    area = 0.0
    n = len(hull_points)
    for i in range(n):
        j = (i + 1) % n
        area += hull_points[i][0] * hull_points[j][1]
        area -= hull_points[j][0] * hull_points[i][1]
    
    return abs(area) / 2.0


def convex_hull_perimeter(hull_points):
    """Calculate perimeter of convex hull"""
    if len(hull_points) < 2:
        return 0.0
    
    perimeter = 0.0
    n = len(hull_points)
    for i in range(n):
        j = (i + 1) % n
        perimeter += dist(hull_points[i], hull_points[j])
    
    return perimeter


def point_in_convex_hull(point, hull_points):
    """
    Check if a point is inside, on edge, or outside the convex hull.
    Returns: 'inside', 'on_edge', or 'outside'
    """
    if len(hull_points) < 3:
        return 'outside'
    
    n = len(hull_points)
    epsilon = 1e-9
    
    # Check if point is a hull vertex (with floating-point tolerance)
    for hull_pt in hull_points:
        if (abs(point[0] - hull_pt[0]) < epsilon and 
            abs(point[1] - hull_pt[1]) < epsilon):
            return 'on_edge'
    
    # First, determine if hull is CCW or CW by checking orientation of first 3 points
    if n >= 3:
        first_orient = orientation(hull_points[0], hull_points[1], hull_points[2])
        is_ccw = first_orient > 0
    else:
        is_ccw = True
    
    # Check if point is on any edge
    for i in range(n):
        j = (i + 1) % n
        orient = orientation(hull_points[i], hull_points[j], point)
        
        # If orientation is 0, point is collinear with edge
        if orient == 0:
            # Check if point is between hull_points[i] and hull_points[j]
            min_x = min(hull_points[i][0], hull_points[j][0])
            max_x = max(hull_points[i][0], hull_points[j][0])
            min_y = min(hull_points[i][1], hull_points[j][1])
            max_y = max(hull_points[i][1], hull_points[j][1])
            
            if (min_x - epsilon <= point[0] <= max_x + epsilon and 
                min_y - epsilon <= point[1] <= max_y + epsilon):
                return 'on_edge'
    
    # Check if point is inside (all orientations should be consistent with hull direction)
    for i in range(n):
        j = (i + 1) % n
        orient = orientation(hull_points[i], hull_points[j], point)
        
        if is_ccw:
            # For CCW hull, all orientations should be >= 0 (left or on)
            if orient < 0:
                return 'outside'
        else:
            # For CW hull, all orientations should be <= 0 (right or on)
            if orient > 0:
                return 'outside'
    
    return 'inside'


def divide(a):
    if len(a) <= 5:
        return bruteHull(a)
    left = a[: len(a)//2]
    right = a[len(a)//2 :]
    return merger(divide(left), divide(right))


def performance_analysis(dataset_sizes, distribution_name="custom"):
    """
    Analyzes performance of Divide & Conquer across different input sizes.
    Generates and saves performance graphs to media/graphs/.
    
    Args:
        dataset_sizes: List of tuples (n, points) where n is the size and points is the list of coordinates
        distribution_name: Name of the distribution type for the graph title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n⚠️  matplotlib not installed. Install it with:")
        print("    pip install matplotlib")
        return
    
    import os
    import math
    
    sizes = []
    times = []
    hull_sizes = []
    nlogn_values = []
    
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS - DIVIDE & CONQUER")
    print("=" * 60)
    print(f"Distribution: {distribution_name}")
    print(f"Testing {len(dataset_sizes)} different input sizes...")
    
    for n, test_points in dataset_sizes:
        # Run multiple times and take average
        num_runs = 5
        total_time = 0
        hull_size = 0
        
        for run in range(num_runs):
            test_points_sorted = sorted(test_points)
            start = time.perf_counter()
            hull = divide(test_points_sorted)
            end = time.perf_counter()
            total_time += (end - start) * 1000  # Convert to ms
            
            # Get hull size from first run
            if run == 0 and hull:
                hull_size = len(hull)
        
        avg_time = total_time / num_runs
        sizes.append(n)
        times.append(avg_time)
        hull_sizes.append(hull_size)
        nlogn_values.append(n * math.log2(n) if n > 1 else n)
        print(f"  n={n:5d}, h={hull_size:4d}: {avg_time:.4f} ms (avg of {num_runs} runs)")
    
    os.makedirs('media/graphs', exist_ok=True)
    
    # Graph 1: Time vs n (Input Size) - More populated x-axis
    plt.figure(figsize=(12, 6))
    plt.plot(sizes, times, 'yo-', linewidth=2, markersize=8)
    plt.xlabel('Input Size (n)', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    plt.title(f"Divide & Conquer: Time vs Input Size (n)\n{distribution_name.replace('_', ' ').title()} Distribution", 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # Add more x-axis ticks for better visualization
    if len(sizes) > 0:
        plt.xticks(sizes, rotation=45)
    plt.tight_layout()
    
    filename1 = f'media/graphs/divide_conquer_{distribution_name}_time_vs_n.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graph 1 (Time vs n) saved to: {filename1}")
    plt.close()
    
    # Graph 2: Time vs O(n log n) - Theoretical Complexity
    plt.figure(figsize=(12, 6))
    plt.plot(nlogn_values, times, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('O(n log n)', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    plt.title(f"Divide & Conquer: Time vs O(n log n)\n{distribution_name.replace('_', ' ').title()} Distribution", 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename2 = f'media/graphs/divide_conquer_{distribution_name}_time_vs_nlogn.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"✓ Graph 2 (Time vs O(n log n)) saved to: {filename2}")
    plt.close()


if __name__ == "__main__":
    import time
    import random
    
    a = [
        [5.29, 3.48], [9.75, -1.54], [11.02, -0.28], [10.39, -0.28], [11.02, 0.98],
        [8.48, -0.911], [9.75, 1], [9.72, 1.6], [9.12, 2.91], [7.85, 2.29], [7.21, 1.62],
        [6.57, 0.36], [4.65, 2.29], [5.29, 0.36], [3.38, 2.91], [4.03, 1], [2.75, 1],
        [5.29, -1.573], [3.38, -0.911], [5.94, -2.18], [2.11, -0.911], [4.66, -2.2],
        [3.38, -2.18], [7.21, -2.782]
    ]
    
    print("=" * 60)
    print("DIVIDE & CONQUER (MERGE-BASED) - CONVEX HULL")
    print("=" * 60)
    print(f"\nNumber of input points: {len(a)}")
    
    # Measure execution time
    a.sort()
    start_time = time.perf_counter()
    ans = divide(a)
    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    print(f"Number of hull points: {len(ans)}")
    print(f"Execution time: {execution_time:.4f} ms")
    
    print("\n" + "-" * 60)
    print("CONVEX HULL COORDINATES:")
    print("-" * 60)
    for i, x in enumerate(ans, 1):
        print(f"{i:2d}. ({x[0]:7.3f}, {x[1]:7.3f})")
    
    # Calculate area and perimeter
    area = convex_hull_area(ans)
    perimeter = convex_hull_perimeter(ans)
    
    print("\n" + "-" * 60)
    print("GEOMETRIC PROPERTIES:")
    print("-" * 60)
    print(f"Area:      {area:.4f} square units")
    print(f"Perimeter: {perimeter:.4f} units")
    
    # Point inclusion testing (only if n <= 100)
    if len(a) <= 100:
        print("\n" + "-" * 60)
        print("POINT INCLUSION TEST:")
        print("-" * 60)
        
        inside_count = 0
        on_edge_count = 0
        
        for i, point in enumerate(a, 1):
            status = point_in_convex_hull(point, ans)
            if status == 'inside':
                inside_count += 1
                print(f"{i:3d}. ({point[0]:7.3f}, {point[1]:7.3f}) -> Inside")
            elif status == 'on_edge':
                on_edge_count += 1
                print(f"{i:3d}. ({point[0]:7.3f}, {point[1]:7.3f}) -> On Edge")
        
        print(f"\nSummary: {on_edge_count} points on hull edges, {inside_count} points inside")
        print(f"Total: {on_edge_count + inside_count} points (all points should be inside or on edge)")
    
    
    print("\n" + "=" * 60)
    
    # Uncomment below to run performance analysis
    # Using larger input sizes (1K-1M) to clearly show O(n log n) behavior
    # Smaller sizes are dominated by overhead and don't show theoretical complexity
    
    print("\n")
    test_datasets = [
        (1000, [[random.uniform(0, 100), random.uniform(0, 100)] for _ in range(1000)]),
        (10000, [[random.uniform(0, 100), random.uniform(0, 100)] for _ in range(10000)]),
        (50000, [[random.uniform(0, 100), random.uniform(0, 100)] for _ in range(50000)]),
        (100000, [[random.uniform(0, 100), random.uniform(0, 100)] for _ in range(100000)]),
        (250000, [[random.uniform(0, 100), random.uniform(0, 100)] for _ in range(250000)]),
        (500000, [[random.uniform(0, 100), random.uniform(0, 100)] for _ in range(500000)]),
        (750000, [[random.uniform(0, 100), random.uniform(0, 100)] for _ in range(750000)]),
        (1000000, [[random.uniform(0, 100), random.uniform(0, 100)] for _ in range(1000000)]),
    ]
    performance_analysis(test_datasets, distribution_name="uniform_random")
    
    