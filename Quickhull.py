from manim import *
import numpy as np
import math


class Algorithm(MovingCameraScene):

    def findSide(self, p1: tuple[float, float], p2: tuple[float, float], p: tuple[float, float]):
        """
        Returns the side of point p with respect to line joining points p1 and p2.
        Returns: 1 if left side, -1 if right side, 0 if collinear
        """
        val = (p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0])
        
        if val > 0:
            return 1
        if val < 0:
            return -1
        return 0

    def lineDist(self, p1: tuple[float, float], p2: tuple[float, float], p: tuple[float, float]):
        """Returns a value proportional to the distance between point p and the line joining p1 and p2"""
        return abs((p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0]))

    def quickhull_visual(self, points: list[tuple[float, float]]):
        """QuickHull algorithm with Manim visualization"""
        
        assert isinstance(points, list) and len(points) >= 3, "points must be a list of size >= 3"
        
        n = len(points)
        min_x = min([p[0] for p in points])
        max_x = max([p[0] for p in points])
        min_y = min([p[1] for p in points])
        max_y = max([p[1] for p in points])

        # Drawing the number plane
        number_plane = NumberPlane(
            x_range=(-100, 100),
            y_range=(-100, 100),
            background_line_style={"stroke_color": WHITE, "stroke_opacity": 0.2}
        )
        number_plane.x_axis.set_opacity(0.2)
        number_plane.y_axis.set_opacity(0.2)
        self.play(FadeIn(number_plane), run_time=0.5)

        # Create Dot objects for each point
        points_mobs = {
            p: Dot((*p, 0), radius=0.1, color=WHITE, fill_opacity=1.0, z_index=float('inf'))
            for p in points
        }

        # Draw points and center camera
        scene_aspect_ratio = config.frame_width / config.frame_height
        self.play(
            self.camera.frame.animate.set(
                width=max((max_x-min_x+1)*1.2, (max_y-min_y+1)*1.2*scene_aspect_ratio)
            ).move_to(
                ((min_x+max_x)/2, (min_y+max_y)/2, 0)
            ),
            *[FadeIn(points_mobs[p]) for p in points]
        )

        # Find points with minimum and maximum x-coordinate
        min_x_point = min(points, key=lambda p: p[0])
        max_x_point = max(points, key=lambda p: p[0])

        # Highlight min and max points
        min_label = MathTex("p_{min}", font_size=28, color="#FF6B6B").next_to(points_mobs[min_x_point], DOWN, buff=0.2)
        max_label = MathTex("p_{max}", font_size=28, color="#4ECDC4").next_to(points_mobs[max_x_point], UP, buff=0.2)
        
        self.play(
            points_mobs[min_x_point].animate.set_color("#FF6B6B"),
            points_mobs[max_x_point].animate.set_color("#4ECDC4"),
            Write(min_label),
            Write(max_label),
            run_time=0.8
        )
        self.wait(0.5)

        # Draw the initial dividing line
        dividing_line = Line(
            points_mobs[min_x_point].get_center(),
            points_mobs[max_x_point].get_center(),
            color=YELLOW,
            stroke_width=3.0
        )
        self.play(Create(dividing_line), run_time=0.8)
        self.wait(0.5)

        # Data structures to track the algorithm
        self.hull_set = set()
        self.hull_lines = []
        self.recursion_depth = 0
        self.max_recursion_depth = 10  # Safety limit

        # Recursively find convex hull
        title = Text("QuickHull Algorithm", font_size=32, color=YELLOW, weight=BOLD).to_edge(UP, buff=0.3).shift(RIGHT * 2)
        self.play(Write(title), run_time=0.5)

        # Process both sides
        self.quickhull_recursive(
            points, n, min_x_point, max_x_point, 1, 
            points_mobs, dividing_line, "#95E1D3"
        )
        
        self.quickhull_recursive(
            points, n, min_x_point, max_x_point, -1, 
            points_mobs, dividing_line, "#F38181"
        )

        # Final hull visualization
        self.play(FadeOut(dividing_line), FadeOut(min_label), FadeOut(max_label), run_time=0.5)
        
        convex_hull_text = Text("Convex Hull", font_size=40, color=ORANGE, weight=BOLD).to_edge(UP, buff=0.3).shift(RIGHT * 2)
        anims = [ReplacementTransform(title, convex_hull_text)]
        
        for p in points:
            if p in self.hull_set:
                anims.append(points_mobs[p].animate.set_color(ORANGE))
            else:
                anims.append(points_mobs[p].animate.set_opacity(0.3))
        
        # Highlight hull lines
        for line in self.hull_lines:
            anims.append(line.animate.set_color(ORANGE).set_stroke_width(4))
        
        self.play(anims, run_time=1.0)
        self.wait(3)

    def quickhull_recursive(self, points, n, p1, p2, side, points_mobs, parent_line, color):
        """Recursive QuickHull with visualization"""
        
        self.recursion_depth += 1
        if self.recursion_depth > self.max_recursion_depth:
            return
        
        ind = -1
        max_dist = 0
        farthest_point = None

        # Find the point with maximum distance from line p1-p2 on the specified side
        distance_lines = []
        for p in points:
            if p == p1 or p == p2:
                continue
                
            temp = self.lineDist(p1, p2, p)
            point_side = self.findSide(p1, p2, p)
            
            if point_side == side and temp > max_dist:
                farthest_point = p
                max_dist = temp

        # If no point found, add endpoints to hull
        if farthest_point is None:
            self.hull_set.add(p1)
            self.hull_set.add(p2)
            
            hull_line = Line(
                points_mobs[p1].get_center(),
                points_mobs[p2].get_center(),
                color=ORANGE,
                stroke_width=3.0,
                z_index=2
            )
            
            if parent_line:
                self.play(
                    ReplacementTransform(parent_line.copy(), hull_line),
                    run_time=0.5
                )
            else:
                self.play(Create(hull_line), run_time=0.5)
            
            self.hull_lines.append(hull_line)
            self.recursion_depth -= 1
            return

        # Visualize the farthest point
        self.play(
            points_mobs[farthest_point].animate.set_color(color).scale(1.3),
            run_time=0.4
        )

        # Draw perpendicular distance line
        dist_line = DashedLine(
            points_mobs[farthest_point].get_center(),
            self.project_point_on_line(farthest_point, p1, p2),
            color=color,
            dash_length=0.1,
            stroke_width=2.0
        )
        dist_label = MathTex(f"d_{{max}}", font_size=24, color=color).next_to(
            dist_line.get_center(), 
            RIGHT if side > 0 else LEFT, 
            buff=0.1
        )
        
        self.play(Create(dist_line), Write(dist_label), run_time=0.5)
        self.wait(0.3)

        # Draw new triangle
        line1 = Line(
            points_mobs[p1].get_center(),
            points_mobs[farthest_point].get_center(),
            color=color,
            stroke_width=2.5
        )
        line2 = Line(
            points_mobs[farthest_point].get_center(),
            points_mobs[p2].get_center(),
            color=color,
            stroke_width=2.5
        )
        
        self.play(
            Create(line1),
            Create(line2),
            FadeOut(dist_line),
            FadeOut(dist_label),
            run_time=0.6
        )
        self.wait(0.3)

        # Recursively process the two new segments
        self.quickhull_recursive(
            points, n, farthest_point, p1, 
            -self.findSide(farthest_point, p1, p2),
            points_mobs, line1, color
        )
        
        self.quickhull_recursive(
            points, n, farthest_point, p2,
            -self.findSide(farthest_point, p2, p1),
            points_mobs, line2, color
        )
        
        self.recursion_depth -= 1

    def project_point_on_line(self, p, p1, p2):
        """Project point p onto line segment p1-p2"""
        x0, y0 = p
        x1, y1 = p1
        x2, y2 = p2
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return (x1, y1, 0)
        
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        
        return (x1 + t * dx, y1 + t * dy, 0)

    def construct(self):
        points = [
            (5.29, 3.48), (9.75, -1.54), (11.02, -0.28), (10.39, -0.28), (11.02, 0.98),
            (8.48, -0.911), (9.75, 1), (9.72, 1.6), (9.12, 2.91), (7.85, 2.29), (7.21, 1.62),
            (6.57, 0.36), (4.65, 2.29), (5.29, 0.36), (3.38, 2.91), (4.03, 1), (2.75, 1),
            (5.29, -1.573), (3.38, -0.911), (5.94, -2.18), (2.11, -0.911), (4.66, -2.2),
            (3.38, -2.18), (7.21, -2.782)
        ]
        self.quickhull_visual(points)


# Non-visual implementation
def findSide(p1, p2, p):
    """Returns the side of point p with respect to line joining points p1 and p2"""
    val = (p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0])
    
    if val > 0:
        return 1
    if val < 0:
        return -1
    return 0


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
    
    # Check if point is on any edge
    for i in range(n):
        j = (i + 1) % n
        if findSide(hull_points[i], hull_points[j], point) == 0:
            # Check if point is between hull_points[i] and hull_points[j]
            min_x = min(hull_points[i][0], hull_points[j][0])
            max_x = max(hull_points[i][0], hull_points[j][0])
            min_y = min(hull_points[i][1], hull_points[j][1])
            max_y = max(hull_points[i][1], hull_points[j][1])
            
            if (min_x - epsilon <= point[0] <= max_x + epsilon and 
                min_y - epsilon <= point[1] <= max_y + epsilon):
                return 'on_edge'
    
    # Check if point is inside - all points should be on the same side
    # We'll check if point has consistent orientation with all edges
    first_side = None
    for i in range(n):
        j = (i + 1) % n
        side = findSide(hull_points[i], hull_points[j], point)
        if side != 0:
            if first_side is None:
                first_side = side
            elif side != first_side:
                return 'outside'
    
    return 'inside'


def lineDist(p1, p2, p):
    """Returns distance between point p and line joining p1 and p2"""
    return abs((p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0]))


def quickHull(a, n, p1, p2, side, hull):
    """QuickHull recursive function"""
    ind = -1
    max_dist = 0

    # Find the point with maximum distance from line p1-p2
    for i in range(n):
        temp = lineDist(p1, p2, a[i])
        
        if (findSide(p1, p2, a[i]) == side) and (temp > max_dist):
            ind = i
            max_dist = temp

    # If no point found, add endpoints to hull
    if ind == -1:
        hull.add(tuple(p1))
        hull.add(tuple(p2))
        return

    # Recursively process the two parts divided by a[ind]
    quickHull(a, n, a[ind], p1, -findSide(a[ind], p1, p2), hull)
    quickHull(a, n, a[ind], p2, -findSide(a[ind], p2, p1), hull)


def printHull(a, n):
    """Main function to find and print convex hull"""
    if n < 3:
        print("Convex hull not possible")
        return []

    hull = set()

    # Find points with minimum and maximum x-coordinate
    min_x = 0
    max_x = 0
    for i in range(1, n):
        if a[i][0] < a[min_x][0]:
            min_x = i
        if a[i][0] > a[max_x][0]:
            max_x = i

    # Recursively find convex hull on both sides
    quickHull(a, n, a[min_x], a[max_x], 1, hull)
    quickHull(a, n, a[min_x], a[max_x], -1, hull)

    print("The points in Convex Hull are:")
    for point in hull:
        print(f"({point[0]}, {point[1]})", end=" ")
    print()
    
    return list(hull)


def performance_analysis(dataset_sizes, distribution_name="custom"):
    """
    Analyzes performance of QuickHull across different input sizes.
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
    print("PERFORMANCE ANALYSIS - QUICKHULL")
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
            hull = printHull(test_points, len(test_points))
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
    plt.plot(sizes, times, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Input Size (n)', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    plt.title(f"QuickHull: Time vs Input Size (n)\n{distribution_name.replace('_', ' ').title()} Distribution", 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # Add more x-axis ticks for better visualization
    if len(sizes) > 0:
        plt.xticks(sizes, rotation=45)
    plt.tight_layout()
    
    filename1 = f'media/graphs/quickhull_{distribution_name}_time_vs_n.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graph 1 (Time vs n) saved to: {filename1}")
    plt.close()
    
    # Graph 2: Time vs O(n log n) - Theoretical Complexity
    plt.figure(figsize=(12, 6))
    plt.plot(nlogn_values, times, 'co-', linewidth=2, markersize=8)
    plt.xlabel('O(n log n)', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    plt.title(f"QuickHull: Time vs O(n log n)\n{distribution_name.replace('_', ' ').title()} Distribution", 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename2 = f'media/graphs/quickhull_{distribution_name}_time_vs_nlogn.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"✓ Graph 2 (Time vs O(n log n)) saved to: {filename2}")
    plt.close()


if __name__ == '__main__':
    import time
    import math
    import random
    
    points = [
        [5.29, 3.48], [9.75, -1.54], [11.02, -0.28], [10.39, -0.28], [11.02, 0.98],
        [8.48, -0.911], [9.75, 1], [9.72, 1.6], [9.12, 2.91], [7.85, 2.29], [7.21, 1.62],
        [6.57, 0.36], [4.65, 2.29], [5.29, 0.36], [3.38, 2.91], [4.03, 1], [2.75, 1],
        [5.29, -1.573], [3.38, -0.911], [5.94, -2.18], [2.11, -0.911], [4.66, -2.2],
        [3.38, -2.18], [7.21, -2.782]
    ]
    n = len(points)
    
    print("=" * 60)
    print("QUICKHULL - CONVEX HULL ALGORITHM")
    print("=" * 60)
    print(f"\nNumber of input points: {n}")
    
    # Measure execution time
    start_time = time.perf_counter()
    hull = printHull(points, n)
    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    # Sort hull points in counter-clockwise order for consistent output
    # Find centroid
    cx = sum(p[0] for p in hull) / len(hull)
    cy = sum(p[1] for p in hull) / len(hull)
    
    # Sort by polar angle
    hull_sorted = sorted(hull, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    
    print(f"Number of hull points: {len(hull_sorted)}")
    print(f"Execution time: {execution_time:.4f} ms")
    
    # print("\n" + "-" * 60)
    # print("CONVEX HULL COORDINATES:")
    # print("-" * 60)
    # for i, point in enumerate(hull_sorted, 1):
    #     print(f"{i:2d}. ({point[0]:7.3f}, {point[1]:7.3f})")
    
    # Calculate area and perimeter
    area = convex_hull_area(hull_sorted)
    perimeter = convex_hull_perimeter(hull_sorted)
    
    print("\n" + "-" * 60)
    print("GEOMETRIC PROPERTIES:")
    print("-" * 60)
    print(f"Area:      {area:.4f} square units")
    print(f"Perimeter: {perimeter:.4f} units")
    
    # Point inclusion testing (only if n <= 100)
    if n <= 100:
        print("\n" + "-" * 60)
        print("POINT INCLUSION TEST:")
        print("-" * 60)
        
        inside_count = 0
        on_edge_count = 0
        
        for i, point in enumerate(points, 1):
            status = point_in_convex_hull(point, hull_sorted)
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