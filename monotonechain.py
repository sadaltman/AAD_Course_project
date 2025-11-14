from manim import *


class Algorithm(MovingCameraScene):

    def cross(self, o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        """2D cross product of OA and OB vectors, i.e. z-component of (a - o) x (b - o)."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def build_chain(self, points: list[tuple[float, float]], points_mobs: dict, color: str):
        """Build a monotone chain (lower or upper) with simple visualization.

        Returns the chain (list of points) and the list of line mobjects used to draw it.
        """
        chain: list[tuple[float, float]] = []
        lines: list = []

        for p in points:
            # Candidate point highlight
            self.play(points_mobs[p].animate.set_color(color), run_time=0.1)

            # Maintain convexity: while last turn is non-CCW, pop
            while len(chain) >= 2 and self.cross(chain[-2], chain[-1], p) <= 0:
                # Remove last segment visually
                if lines:
                    last_line = lines.pop()
                    self.play(Uncreate(last_line), run_time=0.15)
                popped = chain.pop()
                # De-emphasize popped vertex slightly
                self.play(points_mobs[popped].animate.set_color(WHITE), run_time=0.05)

            # Add new segment
            if chain:
                seg = Line(
                    points_mobs[chain[-1]].get_center(),
                    points_mobs[p].get_center(),
                    color=color,
                    stroke_width=3.0,
                )
                lines.append(seg)
                self.play(Create(seg), run_time=0.2)
            chain.append(p)

        # Reset candidate point colors to white; final hull is recolored later
        for p in points:
            self.play(points_mobs[p].animate.set_color(WHITE), run_time=0.02)

        return chain, lines

    def construct(self):
        # Sample points (same style as other files)
        points = [
            (5.29, 3.48), (9.75, -1.54), (11.02, -0.28), (10.39, -0.28), (11.02, 0.98),
            (8.48, -0.911), (9.75, 1), (9.72, 1.6), (9.12, 2.91), (7.85, 2.29), (7.21, 1.62),
            (6.57, 0.36), (4.65, 2.29), (5.29, 0.36), (3.38, 2.91), (4.03, 1), (2.75, 1),
            (5.29, -1.573), (3.38, -0.911), (5.94, -2.18), (2.11, -0.911), (4.66, -2.2),
            (3.38, -2.18), (7.21, -2.782)
        ]

        # Compute bounds for camera framing
        min_x = min(p[0] for p in points); max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points); max_y = max(p[1] for p in points)

        # Number plane
        number_plane = NumberPlane(
            x_range=(-100, 100),
            y_range=(-100, 100),
            background_line_style={"stroke_color": WHITE, "stroke_opacity": 0.2},
        )
        number_plane.x_axis.set_opacity(0.2)
        number_plane.y_axis.set_opacity(0.2)
        self.play(FadeIn(number_plane), run_time=0.4)

        # Points
        points_mobs = {p: Dot((*p, 0), radius=0.1, color=WHITE, fill_opacity=1.0, z_index=float("inf")) for p in points}

        # Frame camera
        scene_aspect_ratio = config.frame_width / config.frame_height
        self.play(
            self.camera.frame.animate.set(
                width=max((max_x - min_x + 1) * 1.2, (max_y - min_y + 1) * 1.2 * scene_aspect_ratio)
            ).move_to(((min_x + max_x) / 2, (min_y + max_y) / 2, 0)),
            *[FadeIn(points_mobs[p]) for p in points],
        )

        # Title
        title = Text("Monotone Chain (Andrew)", font_size=34, color=YELLOW, weight=BOLD).to_edge(UP, buff=0.3).shift(RIGHT * 2)
        self.play(Write(title), run_time=0.4)

        # Sort points by x, then y
        sorted_pts = sorted(points)

        # Phase 1: Lower hull
        phase1 = Text("Lower hull", font_size=22, color=BLUE).next_to(title, DOWN, buff=0.2)
        self.play(Write(phase1), run_time=0.3)
        lower, lower_lines = self.build_chain(sorted_pts, points_mobs, BLUE)

        # Phase 2: Upper hull
        self.play(FadeOut(phase1), run_time=0.2)
        phase2 = Text("Upper hull", font_size=22, color=TEAL).next_to(title, DOWN, buff=0.2)
        self.play(Write(phase2), run_time=0.3)
        upper, upper_lines = self.build_chain(list(reversed(sorted_pts)), points_mobs, TEAL)

        # Combine (omit last of each to avoid duplication of endpoints)
        hull = lower[:-1] + upper[:-1]

        # Clear chain helper lines
        self.play(
            *[FadeOut(l) for l in (lower_lines + upper_lines)],
            FadeOut(phase2),
            run_time=0.4,
        )

        # Final hull drawing
        hull_lines: list[Line] = []
        final_text = Text("Convex Hull", font_size=32, color=ORANGE, weight=BOLD).to_edge(UP, buff=0.3).shift(RIGHT * 2)
        self.play(ReplacementTransform(title, final_text), run_time=0.3)

        for i in range(len(hull)):
            a = hull[i]
            b = hull[(i + 1) % len(hull)]
            line = Line(points_mobs[a].get_center(), points_mobs[b].get_center(), color=ORANGE, stroke_width=4.0)
            hull_lines.append(line)
            self.play(Create(line), run_time=0.25)

        # Emphasize hull points, de-emphasize interior
        anims = []
        for p in points:
            if p in hull:
                anims.append(points_mobs[p].animate.set_color(ORANGE).scale(1.15))
            else:
                anims.append(points_mobs[p].animate.set_opacity(0.3))
        self.play(anims, run_time=0.6)
        self.wait(3)


# Non-visual reference implementation
def monotone_chain(points: list[tuple[float, float]]):
    """Andrew's monotone chain convex hull. Returns hull vertices in CCW order."""
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


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
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Check if point is on any edge
    for i in range(n):
        j = (i + 1) % n
        if abs(cross(hull_points[i], hull_points[j], point)) < epsilon:
            # Check if point is between hull_points[i] and hull_points[j]
            min_x = min(hull_points[i][0], hull_points[j][0])
            max_x = max(hull_points[i][0], hull_points[j][0])
            min_y = min(hull_points[i][1], hull_points[j][1])
            max_y = max(hull_points[i][1], hull_points[j][1])
            
            if (min_x - epsilon <= point[0] <= max_x + epsilon and 
                min_y - epsilon <= point[1] <= max_y + epsilon):
                return 'on_edge'
    
    # Check if point is inside using cross product method
    for i in range(n):
        j = (i + 1) % n
        if cross(hull_points[i], hull_points[j], point) < -epsilon:
            return 'outside'
    
    return 'inside'


def performance_analysis(dataset_sizes, distribution_name="custom"):
    """
    Analyzes performance of Monotone Chain across different input sizes.
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
    print("PERFORMANCE ANALYSIS - MONOTONE CHAIN")
    print("=" * 60)
    print(f"Distribution: {distribution_name}")
    print(f"Testing {len(dataset_sizes)} different input sizes...")
    
    for n, test_points in dataset_sizes:
        # Run multiple times and take average
        num_runs = 5
        total_time = 0
        hull_size = 0
        
        for run in range(num_runs):
            start = time.perf_counter()
            hull = monotone_chain(test_points)
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
    plt.plot(sizes, times, 'co-', linewidth=2, markersize=8)
    plt.xlabel('Input Size (n)', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    plt.title(f"Monotone Chain: Time vs Input Size (n)\n{distribution_name.replace('_', ' ').title()} Distribution", 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # Add more x-axis ticks for better visualization
    if len(sizes) > 0:
        plt.xticks(sizes, rotation=45)
    plt.tight_layout()
    
    filename1 = f'media/graphs/monotone_chain_{distribution_name}_time_vs_n.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graph 1 (Time vs n) saved to: {filename1}")
    plt.close()
    
    # Graph 2: Time vs O(n log n) - Theoretical Complexity
    plt.figure(figsize=(12, 6))
    plt.plot(nlogn_values, times, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('O(n log n)', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    plt.title(f"Monotone Chain: Time vs O(n log n)\n{distribution_name.replace('_', ' ').title()} Distribution", 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename2 = f'media/graphs/monotone_chain_{distribution_name}_time_vs_nlogn.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"✓ Graph 2 (Time vs O(n log n)) saved to: {filename2}")
    plt.close()


if __name__ == "__main__":
    import time
    import random
    
    pts = [
        (5.29, 3.48), (9.75, -1.54), (11.02, -0.28), (10.39, -0.28), (11.02, 0.98),
        (8.48, -0.911), (9.75, 1), (9.72, 1.6), (9.12, 2.91), (7.85, 2.29), (7.21, 1.62),
        (6.57, 0.36), (4.65, 2.29), (5.29, 0.36), (3.38, 2.91), (4.03, 1), (2.75, 1),
        (5.29, -1.573), (3.38, -0.911), (5.94, -2.18), (2.11, -0.911), (4.66, -2.2),
        (3.38, -2.18), (7.21, -2.782)
    ]
    
    print("=" * 60)
    print("MONOTONE CHAIN (ANDREW'S ALGORITHM) - CONVEX HULL")
    print("=" * 60)
    print(f"\nNumber of input points: {len(pts)}")
    
    # Measure execution time
    start_time = time.perf_counter()
    hull = monotone_chain(pts)
    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    print(f"Number of hull points: {len(hull)}")
    print(f"Execution time: {execution_time:.4f} ms")
    
    print("\n" + "-" * 60)
    print("CONVEX HULL COORDINATES:")
    print("-" * 60)
    for i, (x, y) in enumerate(hull, 1):
        print(f"{i:2d}. ({x:7.3f}, {y:7.3f})")
    
    # Calculate area and perimeter
    area = convex_hull_area(hull)
    perimeter = convex_hull_perimeter(hull)
    
    print("\n" + "-" * 60)
    print("GEOMETRIC PROPERTIES:")
    print("-" * 60)
    print(f"Area:      {area:.4f} square units")
    print(f"Perimeter: {perimeter:.4f} units")
    
    # Point inclusion testing (only if n <= 100)
    if len(pts) <= 100:
        print("\n" + "-" * 60)
        print("POINT INCLUSION TEST:")
        print("-" * 60)
        
        inside_count = 0
        on_edge_count = 0
        
        for i, point in enumerate(pts, 1):
            status = point_in_convex_hull(point, hull)
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
        (1000, [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(1000)]),
        (10000, [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10000)]),
        (50000, [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(50000)]),
        (100000, [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(100000)]),
        (250000, [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(250000)]),
        (500000, [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(500000)]),
        (750000, [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(750000)]),
        (1000000, [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(1000000)]),
    ]
    performance_analysis(test_datasets, distribution_name="uniform_random")
    

