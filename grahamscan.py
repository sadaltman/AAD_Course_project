from manim import *
import numpy as np
import math
import random


COUNTERCLOCKWISE = 1
COLLINEAR = 0
CLOCKWISE = -1


class Algorithm(MovingCameraScene):

    def orientation(self, p1: tuple[float,float], p2: tuple[float,float], p3: tuple[float,float]):
        x1, y1, x2, y2, x3, y3 = *p1, *p2, *p3
        diff = (y3-y2)*(x2-x1) - (y2-y1)*(x3-x2)
        return COUNTERCLOCKWISE if diff > 0 else (CLOCKWISE if diff < 0 else COLLINEAR)

    def dist(self, p1: tuple[float,float], p2: tuple[float,float]):
        x1, y1, x2, y2 = *p1, *p2
        return math.sqrt((y2-y1)**2 + (x2-x1)**2)
    
    def polar_angle(self, p1: tuple[float,float], p2: tuple[float,float]):
        dy = p1[1] - p2[1]
        dx = p1[0] - p2[0]
        return math.atan2(dy, dx)
    
    # finds the best direction to avoid drawing on top of lines from reference point to other points
    # by generating num_dirs directions then finding the farthest one from other points
    def find_best_direction(self, reference_point: VMobject, other_points: list[VMobject], num_dirs=36):
        angles = np.linspace(0, 2*np.pi, num_dirs)
        potential_directions = [np.array([np.cos(a), np.sin(a), 0]) for a in angles]
        center = reference_point.get_center()
        other_centers = [p.get_center() for p in other_points]
        return max(
            potential_directions, 
            key=lambda d: sum([np.linalg.norm(oc - (center + d)) for oc in other_centers])
        )
        
    def graham_scan(self, points: list[tuple[float,float]]):

        assert isinstance(points, list) and len(points) > 2, "points must be a list of size > 2"
        assert all([
            isinstance(p, tuple) 
            and len(p) == 2 
            and isinstance(p[0], (int, float))
            and isinstance(p[1], (int, float))
            for p in points
        ]), "a point must be a tuple of 2 numbers"

        min_x, max_x = min([p[0] for p in points]), max([p[0] for p in points])
        min_y, max_y = min([p[1] for p in points]), max([p[1] for p in points])

        # drawing the number plane
        number_plane = NumberPlane(
            x_range=(-100, 100),
            y_range=(-100, 100),
            background_line_style={"stroke_color": WHITE, "stroke_opacity": 0.2}
        )
        number_plane.x_axis.set_opacity(0.2)
        number_plane.y_axis.set_opacity(0.2)
        self.play(FadeIn(number_plane), run_time=0.5)
        
        # VDict that maps each point to its Dot object on the scene
        points_mobs = VDict({
            p: Dot((*p, 0), radius=0.1, color=WHITE, fill_opacity=1.0, z_index=float('inf')) for p in points
        })

        # draw points and center the camera on the appropriate area
        scene_aspect_ratio = config.frame_width / config.frame_height
        self.play(
            self.camera.frame.animate.set(
                width=max((max_x-min_x+1)*1.2, (max_y-min_y+1)*1.2*scene_aspect_ratio)
            ).move_to(
                ((min_x+max_x)/2, (min_y+max_y)/2, 0)
        ), FadeIn(points_mobs))

        # finding lowest point p0
        p0_idx = min(range(len(points)), key=lambda i: (points[i][1], points[i][0]))
        p0 = points[p0_idx]
        p0_label = MathTex("p_0", font_size=32).next_to(points_mobs[p0], DOWN, buff=0.2)
        horizontal_line = Line((min_x-1, p0[1], 0), (max_x+1, p0[1], 0))
        dashed_lines = [DashedLine((*p0, 0), (*p, 0), color="#00EFD1", dash_length=0.15, stroke_width=1.0, z_index=float('-inf')) for p in points if p0 != p]
        self.play(
            Create(horizontal_line),
            points_mobs[p0].animate.set_color("#00EFD1"),
            Write(p0_label),
            run_time=0.5
        )
        self.play(*[Create(l) for l in dashed_lines], run_time=0.5)

        # sort points according to their polar angle with p0, or distance if equal angles
        points.sort(key=lambda p: (self.polar_angle(p, p0), self.dist(p, p0)))
        angle = Sector(1.5, angle=self.polar_angle(points[1], p0), fill_opacity=0.6, stroke_color=PURPLE, stroke_width=1.5, color=BLACK)
        angle.shift(points_mobs[p0].get_center() - angle.get_arc_center())
        self.play(FadeIn(angle))

        orientation_icons = {
            COUNTERCLOCKWISE: Arc(radius=0.5, angle=2/3*TAU, start_angle=1/4*PI, stroke_width=4).add_tip().scale(0.3),
            COLLINEAR: Arrow(start=ORIGIN, end=RIGHT*3, stroke_width=4).scale(0.3),
            CLOCKWISE: Arc(radius=0.5, angle=2/3*TAU, start_angle=1/4*PI, stroke_width=4).add_tip().scale(0.3).flip()
        }
        orientation_labels = {
            COUNTERCLOCKWISE: Text("counter clockwise", fill_opacity=0.8, font_size=20),
            COLLINEAR: Text("collinear", fill_opacity=0.8, font_size=20),
            CLOCKWISE: Text("clockwise", fill_opacity=0.8, font_size=20)
        }
        
        hull_lines = []
        hull = [p0]
        for i in range(1, len(points)):
            new_angle = Sector(1.5, angle=self.polar_angle(points[i], p0), fill_opacity=0.6, stroke_color=PURPLE, stroke_width=1.5, color=BLACK)
            new_angle.shift(points_mobs[p0].get_center() - new_angle.get_arc_center())
            line = Line(points_mobs[hull[-1]].get_center(), points_mobs[points[i]].get_center(), color=BLUE if len(hull) >= 2 else ORANGE)
            animations = [
                ReplacementTransform(angle, new_angle),
                Create(line)
            ]
            if len(hull) >= 2:
                best_direction = self.find_best_direction(points_mobs[hull[-1]], [points_mobs[hull[-2]], points_mobs[points[i]]])
                orientation_icon = orientation_icons[self.orientation(hull[-2], hull[-1], points[i])]
                orientation_label = orientation_labels[self.orientation(hull[-2], hull[-1], points[i])]
                orientation_icon.next_to(points_mobs[hull[-1]], best_direction)
                orientation_label.next_to(orientation_icon, best_direction)
                animations.extend([Create(orientation_icon), FadeIn(orientation_label)])
            self.play(animations, run_time=0.5)
            angle = new_angle
            while len(hull) >= 2 and \
            self.orientation(hull[-2], hull[-1], points[i]) != COUNTERCLOCKWISE:
                self.wait(0.5)
                hull.pop()
                self.play(
                    Uncreate(hull_lines[-1]),
                    line.animate.put_start_and_end_on(points_mobs[hull[-1]].get_center(), points_mobs[points[i]].get_center()),
                    FadeOut(orientation_icon),
                    FadeOut(orientation_label),
                    run_time=0.5
                )
                hull_lines.pop()
                if len(hull) >= 2:
                    best_direction = self.find_best_direction(points_mobs[hull[-1]], [points_mobs[hull[-2]], points_mobs[points[i]]])
                    orientation_icon = orientation_icons[self.orientation(hull[-2], hull[-1], points[i])]
                    orientation_label = orientation_labels[self.orientation(hull[-2], hull[-1], points[i])]
                    orientation_icon.next_to(points_mobs[hull[-1]], best_direction)
                    orientation_label.next_to(orientation_icon, best_direction)
                    self.play(Create(orientation_icon), FadeIn(orientation_label), run_time=0.5)
  
            self.wait(0.5)
            if len(hull) >= 2:
                self.play(
                    line.animate.set_color(ORANGE), 
                    FadeOut(orientation_icon),
                    FadeOut(orientation_label),
                    run_time=0.5
                )
            hull.append(points[i])
            hull_lines.append(line)
            self.wait(0.5)
            
        line = Line(points_mobs[hull[-1]].get_center(), points_mobs[p0].get_center(), color=ORANGE)
        hull_lines.append(line)
        self.play(Create(line), run_time=0.5)
        convex_hull_text = Text("Convex Hull", font_size=48, color=ORANGE, weight=BOLD).next_to(points_mobs[p0], DOWN, buff=0.2)
        anims = [Write(convex_hull_text), Uncreate(angle)]
        for p in points:
            anims.append(
                points_mobs[p].animate.set_color(ORANGE) if p in hull
                else points_mobs[p].animate.set_opacity(0.3)
            )
        anims.extend([FadeOut(l) for l in dashed_lines])
        self.play(anims, run_time=0.5)
        self.wait(3)
        
    def construct(self):
        points = [
            (5.29, 3.48), (9.75, -1.54), (11.02, -0.28), (10.39, -0.28), (11.02, 0.98), 
            (8.48, -0.911), (9.75, 1), (9.72, 1.6), (9.12, 2.91), (7.85, 2.29), (7.21, 1.62), 
            (6.57, 0.36), (4.65, 2.29), (5.29, 0.36), (3.38, 2.91), (4.03, 1), (2.75, 1), (5.29, -1.573), 
            (3.38, -0.911), (5.94, -2.18), (2.11, -0.911), (4.66, -2.2), (3.38, -2.18), (7.21, -2.782)
        ]
        self.graham_scan(points)


def orientation(p1: tuple[float,float], p2: tuple[float,float], p3: tuple[float,float]):
    x1, y1, x2, y2, x3, y3 = *p1, *p2, *p3
    diff = (y3-y2)*(x2-x1) - (y2-y1)*(x3-x2)
    return COUNTERCLOCKWISE if diff > 0 else (CLOCKWISE if diff < 0 else COLLINEAR)


def dist(p1: tuple[float,float], p2: tuple[float,float]):
    x1, y1, x2, y2 = *p1, *p2
    return math.sqrt((y2-y1)**2 + (x2-x1)**2)


def polar_angle(p1: tuple[float,float], p2: tuple[float,float]):
    dy = p1[1] - p2[1]
    dx = p1[0] - p2[0]
    return math.atan2(dy, dx)

    
def convex_hull_area(hull_points: list[tuple[float, float]]) -> float:
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


def convex_hull_perimeter(hull_points: list[tuple[float, float]]) -> float:
    """Calculate perimeter of convex hull"""
    if len(hull_points) < 2:
        return 0.0
    
    perimeter = 0.0
    n = len(hull_points)
    for i in range(n):
        j = (i + 1) % n
        perimeter += dist(hull_points[i], hull_points[j])
    
    return perimeter


def point_in_convex_hull(point: tuple[float, float], hull_points: list[tuple[float, float]]) -> str:
    """
    Check if a point is inside, on edge, or outside the convex hull.
    Returns: 'inside', 'on_edge', or 'outside'
    """
    if len(hull_points) < 3:
        return 'outside'
    
    n = len(hull_points)
    epsilon = 1e-9  # Tolerance for floating-point comparison
    
    # Check if point is on any edge
    for i in range(n):
        j = (i + 1) % n
        # Check if point is collinear with edge and within bounds
        if orientation(hull_points[i], hull_points[j], point) == COLLINEAR:
            # Check if point is between hull_points[i] and hull_points[j]
            min_x = min(hull_points[i][0], hull_points[j][0])
            max_x = max(hull_points[i][0], hull_points[j][0])
            min_y = min(hull_points[i][1], hull_points[j][1])
            max_y = max(hull_points[i][1], hull_points[j][1])
            
            if (min_x - epsilon <= point[0] <= max_x + epsilon and 
                min_y - epsilon <= point[1] <= max_y + epsilon):
                return 'on_edge'
    
    # Check if point is inside using cross product method
    # All cross products should have the same sign for a point inside convex hull
    for i in range(n):
        j = (i + 1) % n
        if orientation(hull_points[i], hull_points[j], point) == CLOCKWISE:
            return 'outside'
    
    return 'inside'


def graham_scan(points: list[tuple[float,float]]):

    assert isinstance(points, list) and len(points) > 2, "points must be a list of size > 2"
    assert all([
        isinstance(p, tuple) 
        and len(p) == 2 
        and isinstance(p[0], (int, float))
        and isinstance(p[1], (int, float))
        for p in points
    ]), "a point must be a tuple of 2 numbers"

    p0 = min(points, key=lambda p: (p[1], p[0]))
    points.sort(key=lambda p: (polar_angle(p, p0), dist(p, p0)))

    hull = [p0]
    for i in range(1, len(points)):
        while len(hull) >= 2 and \
        orientation(hull[-2], hull[-1], points[i]) != COUNTERCLOCKWISE:
            hull.pop()
        hull.append(points[i])

    return hull


def performance_analysis(dataset_sizes, distribution_name="custom"):
    """
    Analyzes performance of Graham's Scan across different input sizes.
    Generates and saves performance graphs to media/graphs/.
    
    Args:
        dataset_sizes: List of tuples (n, points) where n is the size and points is the list of coordinates
        distribution_name: Name of the distribution type for the graph title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n" + "!" * 60)
        print("ERROR: matplotlib is not installed!")
        print("!" * 60)
        print("\nTo install matplotlib, run:")
        print("  pip install matplotlib")
        print("or")
        print("  conda install matplotlib")
        print("\nPerformance analysis requires matplotlib for graph generation.")
        print("!" * 60)
        return
    
    import os
    import time
    import math
    
    sizes = []
    hull_sizes = []
    times = []
    nlogn_values = []
    
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS - GRAHAM'S SCAN")
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
            hull = graham_scan(test_points)
            end = time.perf_counter()
            total_time += (end - start) * 1000  # Convert to ms
            if run == 0:  # Get hull size from first run
                hull_size = len(hull)
        
        avg_time = total_time / num_runs
        sizes.append(n)
        hull_sizes.append(hull_size)
        times.append(avg_time)
        nlogn_values.append(n * math.log2(n) if n > 1 else n)
        print(f"  n={n:5d}, h={hull_size:4d}: {avg_time:.4f} ms (avg of {num_runs} runs)")
    
    os.makedirs('media/graphs', exist_ok=True)
    
    # Graph 1: Time vs n (Input Size) - More populated x-axis
    plt.figure(figsize=(12, 6))
    plt.plot(sizes, times, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Input Size (n)', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    plt.title(f"Graham's Scan: Time vs Input Size (n)\n{distribution_name.replace('_', ' ').title()} Distribution", 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # Add more x-axis ticks for better visualization
    if len(sizes) > 0:
        plt.xticks(sizes, rotation=45)
    plt.tight_layout()
    
    filename1 = f'media/graphs/graham_scan_{distribution_name}_time_vs_n.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graph 1 (Time vs n) saved to: {filename1}")
    plt.close()
    
    # Graph 2: Time vs O(n log n) - Theoretical Complexity
    plt.figure(figsize=(12, 6))
    plt.plot(nlogn_values, times, 'go-', linewidth=2, markersize=8)
    plt.xlabel('O(n log n)', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    plt.title(f"Graham's Scan: Time vs O(n log n)\n{distribution_name.replace('_', ' ').title()} Distribution", 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename2 = f'media/graphs/graham_scan_{distribution_name}_time_vs_nlogn.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"✓ Graph 2 (Time vs O(n log n)) saved to: {filename2}")
    plt.close()


if __name__ == '__main__':
    import time
    import random
    
    points = [
        (5.29, 3.48), (9.75, -1.54), (11.02, -0.28), (10.39, -0.28), (11.02, 0.98), 
        (8.48, -0.911), (9.75, 1), (9.72, 1.6), (9.12, 2.91), (7.85, 2.29), (7.21, 1.62), 
        (6.57, 0.36), (4.65, 2.29), (5.29, 0.36), (3.38, 2.91), (4.03, 1), (2.75, 1), (5.29, -1.573), 
        (3.38, -0.911), (5.94, -2.18), (2.11, -0.911), (4.66, -2.2), (3.38, -2.18), (7.21, -2.782)
    ]
    
    print("=" * 60)
    print("GRAHAM'S SCAN - CONVEX HULL ALGORITHM")
    print("=" * 60)
    print(f"\nNumber of input points: {len(points)}")
    
    # Measure execution time
    start_time = time.perf_counter()
    hull = graham_scan(points)
    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    print(f"Number of hull points: {len(hull)}")
    print(f"Execution time: {execution_time:.4f} ms")
    
    print("\n" + "-" * 60)
    print("CONVEX HULL COORDINATES:")
    print("-" * 60)
    for i, point in enumerate(hull, 1):
        print(f"{i:2d}. ({point[0]:7.3f}, {point[1]:7.3f})")
    
    # Calculate area and perimeter
    area = convex_hull_area(hull)
    perimeter = convex_hull_perimeter(hull)
    
    print("\n" + "-" * 60)
    print("GEOMETRIC PROPERTIES:")
    print("-" * 60)
    print(f"Area:      {area:.4f} square units")
    print(f"Perimeter: {perimeter:.4f} units")
    
    # Point inclusion testing (only if n <= 100)
    if len(points) <= 100:
        print("\n" + "-" * 60)
        print("POINT INCLUSION TEST:")
        print("-" * 60)
        
        inside_count = 0
        on_edge_count = 0
        
        for i, point in enumerate(points, 1):
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
    # NOTE: Requires matplotlib: pip install matplotlib
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

    