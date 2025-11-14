from manim import *
import numpy as np
import math
import random


CLOCKWISE = 1
COLLINEAR = 0
COUNTERCLOCKWISE = 2


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Algorithm(MovingCameraScene):

    def Left_index(self, points: list[Point]):
        """Finding the left most point"""
        minn = 0
        for i in range(1, len(points)):
            if points[i].x < points[minn].x:
                minn = i
            elif points[i].x == points[minn].x:
                if points[i].y > points[minn].y:
                    minn = i
        return minn

    def orientation(self, p: Point, q: Point, r: Point):
        """
        To find orientation of ordered triplet (p, q, r).
        The function returns following values:
        0 --> p, q and r are collinear
        1 --> Clockwise
        2 --> Counterclockwise
        """
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        
        if val == 0:
            return COLLINEAR
        elif val > 0:
            return CLOCKWISE
        else:
            return COUNTERCLOCKWISE

    def find_best_direction(self, reference_point: VMobject, other_points: list[VMobject], num_dirs=36):
        """Finds the best direction to avoid drawing on top of lines"""
        angles = np.linspace(0, 2*np.pi, num_dirs)
        potential_directions = [np.array([np.cos(a), np.sin(a), 0]) for a in angles]
        center = reference_point.get_center()
        other_centers = [p.get_center() for p in other_points]
        return max(
            potential_directions,
            key=lambda d: sum([np.linalg.norm(oc - (center + d)) for oc in other_centers])
        )

    def jarvis_march(self, points: list[Point]):
        """Jarvis March (Gift Wrapping) Algorithm with Manim visualization"""
        
        assert isinstance(points, list) and len(points) > 2, "points must be a list of size > 2"
        
        n = len(points)
        min_x = min([p.x for p in points])
        max_x = max([p.x for p in points])
        min_y = min([p.y for p in points])
        max_y = max([p.y for p in points])

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
        points_mobs = {}
        for i, p in enumerate(points):
            points_mobs[i] = Dot((p.x, p.y, 0), radius=0.1, color=WHITE, fill_opacity=1.0, z_index=float('inf'))

        # Draw points and center the camera
        scene_aspect_ratio = config.frame_width / config.frame_height
        self.play(
            self.camera.frame.animate.set(
                width=max((max_x-min_x+1)*1.2, (max_y-min_y+1)*1.2*scene_aspect_ratio)
            ).move_to(
                ((min_x+max_x)/2, (min_y+max_y)/2, 0)
            ),
            *[FadeIn(points_mobs[i]) for i in range(n)]
        )

        # Find the leftmost point
        l = self.Left_index(points)
        
        # Highlight leftmost point
        leftmost_label = MathTex("l", font_size=28, color="#00EFD1").next_to(points_mobs[l], DOWN, buff=0.2)
        vertical_line = DashedLine((points[l].x, min_y-1, 0), (points[l].x, max_y+1, 0), color="#00EFD1", dash_length=0.15, stroke_width=2.0)
        
        self.play(
            Create(vertical_line),
            points_mobs[l].animate.set_color("#00EFD1"),
            Write(leftmost_label),
            run_time=0.8
        )
        self.wait(0.5)
        self.play(FadeOut(vertical_line), run_time=0.5)

        # Initialize hull
        hull = []
        hull_lines = []
        p = l
        p_label = MathTex("p", font_size=28, color=YELLOW).next_to(points_mobs[p], UP, buff=0.2)
        
        self.play(
            FadeOut(leftmost_label),
            Write(p_label),
            points_mobs[p].animate.set_color(YELLOW),
            run_time=0.5
        )

        # Orientation icons
        orientation_icons = {
            COUNTERCLOCKWISE: Arc(radius=0.5, angle=2/3*TAU, start_angle=1/4*PI, stroke_width=4, color=GREEN).add_tip().scale(0.3),
            COLLINEAR: Arrow(start=ORIGIN, end=RIGHT*3, stroke_width=4, color=YELLOW).scale(0.3),
            CLOCKWISE: Arc(radius=0.5, angle=-2/3*TAU, start_angle=1/4*PI, stroke_width=4, color=RED).add_tip().scale(0.3)
        }
        orientation_labels = {
            COUNTERCLOCKWISE: Text("counterclockwise", fill_opacity=0.8, font_size=20, color=GREEN),
            COLLINEAR: Text("collinear", fill_opacity=0.8, font_size=20, color=YELLOW),
            CLOCKWISE: Text("clockwise", fill_opacity=0.8, font_size=20, color=RED)
        }

        iteration = 0
        max_iterations = n + 1  # Safety limit
        
        while iteration < max_iterations:
            hull.append(p)
            
            # Initialize q as next point
            q = (p + 1) % n
            q_label = MathTex("q", font_size=28, color=BLUE).next_to(points_mobs[q], DOWN, buff=0.2)
            candidate_line = DashedLine(
                points_mobs[p].get_center(),
                points_mobs[q].get_center(),
                color=BLUE,
                dash_length=0.15,
                stroke_width=2.0
            )
            
            self.play(
                points_mobs[q].animate.set_color(BLUE),
                Write(q_label),
                Create(candidate_line),
                run_time=0.5
            )
            self.wait(0.3)

            # Check all points to find the most counterclockwise
            for i in range(n):
                if i == p or i == q:
                    continue
                
                # Highlight test point
                i_label = MathTex("i", font_size=28, color=PURPLE).next_to(points_mobs[i], UP, buff=0.2)
                test_line = Line(
                    points_mobs[p].get_center(),
                    points_mobs[i].get_center(),
                    color=PURPLE,
                    stroke_width=2.0
                )
                
                self.play(
                    points_mobs[i].animate.set_color(PURPLE),
                    Write(i_label),
                    Create(test_line),
                    run_time=0.3
                )

                # Check orientation
                orient = self.orientation(points[p], points[i], points[q])
                
                # Show orientation indicator
                best_dir = self.find_best_direction(points_mobs[p], [points_mobs[q], points_mobs[i]])
                orient_icon = orientation_icons[orient].copy()
                orient_label = orientation_labels[orient].copy()
                orient_icon.next_to(points_mobs[p], best_dir, buff=0.3)
                orient_label.next_to(orient_icon, best_dir, buff=0.1)
                
                self.play(
                    Create(orient_icon),
                    FadeIn(orient_label),
                    run_time=0.3
                )
                self.wait(0.3)

                if orient == COUNTERCLOCKWISE:
                    # Update q to i
                    self.play(
                        FadeOut(orient_icon),
                        FadeOut(orient_label),
                        FadeOut(q_label),
                        points_mobs[q].animate.set_color(WHITE),
                        run_time=0.3
                    )
                    
                    q = i
                    
                    # Transform test line to candidate line
                    new_candidate_line = DashedLine(
                        points_mobs[p].get_center(),
                        points_mobs[q].get_center(),
                        color=BLUE,
                        dash_length=0.15,
                        stroke_width=2.0
                    )
                    
                    self.play(
                        Uncreate(candidate_line),
                        test_line.animate.become(new_candidate_line),
                        FadeOut(i_label),
                        points_mobs[i].animate.set_color(BLUE),
                        run_time=0.3
                    )
                    
                    candidate_line = new_candidate_line
                    q_label = MathTex("q", font_size=28, color=BLUE).next_to(points_mobs[q], DOWN, buff=0.2)
                    self.play(Write(q_label), run_time=0.2)
                else:
                    # i is not more counterclockwise, remove it
                    self.play(
                        FadeOut(orient_icon),
                        FadeOut(orient_label),
                        Uncreate(test_line),
                        FadeOut(i_label),
                        points_mobs[i].animate.set_color(WHITE),
                        run_time=0.3
                    )
                
                self.wait(0.2)

            # q is the most counterclockwise point, add to hull
            hull_line = Line(
                points_mobs[p].get_center(),
                points_mobs[q].get_center(),
                color=ORANGE,
                stroke_width=3.0,
                z_index=1
            )
            
            self.play(
                ReplacementTransform(candidate_line, hull_line),
                run_time=0.5
            )
            hull_lines.append(hull_line)
            
            # Set p as q for next iteration
            self.play(
                FadeOut(p_label),
                points_mobs[p].animate.set_color(ORANGE),
                run_time=0.3
            )
            
            p = q
            
            # Check if we've returned to the starting point
            if p == l:
                self.play(FadeOut(q_label), run_time=0.3)
                break
            
            self.play(
                FadeOut(q_label),
                points_mobs[p].animate.set_color(YELLOW),
                run_time=0.3
            )
            
            p_label = MathTex("p", font_size=28, color=YELLOW).next_to(points_mobs[p], UP, buff=0.2)
            self.play(Write(p_label), run_time=0.3)
            
            iteration += 1
            self.wait(0.3)

        # Final visualization
        convex_hull_text = Text("Convex Hull - Jarvis March", font_size=40, color=ORANGE, weight=BOLD)
        convex_hull_text.to_edge(UP, buff=0.5).shift(RIGHT * 2)
        
        anims = [Write(convex_hull_text)]
        for i in range(n):
            if i in hull:
                anims.append(points_mobs[i].animate.set_color(ORANGE))
            else:
                anims.append(points_mobs[i].animate.set_opacity(0.3))
        
        self.play(anims, run_time=0.8)
        self.wait(3)

    def construct(self):
        points = [
            Point(5.29, 3.48), Point(9.75, -1.54), Point(11.02, -0.28), Point(10.39, -0.28), 
            Point(11.02, 0.98), Point(8.48, -0.911), Point(9.75, 1), Point(9.72, 1.6), 
            Point(9.12, 2.91), Point(7.85, 2.29), Point(7.21, 1.62), Point(6.57, 0.36), 
            Point(4.65, 2.29), Point(5.29, 0.36), Point(3.38, 2.91), Point(4.03, 1), 
            Point(2.75, 1), Point(5.29, -1.573), Point(3.38, -0.911), Point(5.94, -2.18), 
            Point(2.11, -0.911), Point(4.66, -2.2), Point(3.38, -2.18), Point(7.21, -2.782)
        ]
        self.jarvis_march(points)


def Left_index(points: list[Point]):
    """Finding the left most point"""
    minn = 0
    for i in range(1, len(points)):
        if points[i].x < points[minn].x:
            minn = i
        elif points[i].x == points[minn].x:
            if points[i].y > points[minn].y:
                minn = i
    return minn


def orientation(p: Point, q: Point, r: Point):
    """
    To find orientation of ordered triplet (p, q, r).
    The function returns following values:
    0 --> p, q and r are collinear
    1 --> Clockwise
    2 --> Counterclockwise
    """
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    
    if val == 0:
        return COLLINEAR
    elif val > 0:
        return CLOCKWISE
    else:
        return COUNTERCLOCKWISE


def dist(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)


def convex_hull_area(hull_points: list[Point]) -> float:
    """Calculate area of convex hull using Shoelace formula"""
    if len(hull_points) < 3:
        return 0.0
    
    area = 0.0
    n = len(hull_points)
    for i in range(n):
        j = (i + 1) % n
        area += hull_points[i].x * hull_points[j].y
        area -= hull_points[j].x * hull_points[i].y
    
    return abs(area) / 2.0


def convex_hull_perimeter(hull_points: list[Point]) -> float:
    """Calculate perimeter of convex hull"""
    if len(hull_points) < 2:
        return 0.0
    
    perimeter = 0.0
    n = len(hull_points)
    for i in range(n):
        j = (i + 1) % n
        perimeter += dist(hull_points[i], hull_points[j])
    
    return perimeter


def point_in_convex_hull(point: Point, hull_points: list[Point]) -> str:
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
            min_x = min(hull_points[i].x, hull_points[j].x)
            max_x = max(hull_points[i].x, hull_points[j].x)
            min_y = min(hull_points[i].y, hull_points[j].y)
            max_y = max(hull_points[i].y, hull_points[j].y)
            
            if (min_x - epsilon <= point.x <= max_x + epsilon and 
                min_y - epsilon <= point.y <= max_y + epsilon):
                return 'on_edge'
    
    # Check if point is inside using cross product method
    # All cross products should have the same sign for a point inside convex hull
    for i in range(n):
        j = (i + 1) % n
        if orientation(hull_points[i], hull_points[j], point) == CLOCKWISE:
            return 'outside'
    
    return 'inside'


def convexHull(points: list[Point], n: int):
    """Jarvis March algorithm without visualization"""
    
    if n < 3:
        return []
    
    l = Left_index(points)
    hull = []
    
    p = l
    while True:
        hull.append(p)
        q = (p + 1) % n
        
        for i in range(n):
            if orientation(points[p], points[i], points[q]) == COUNTERCLOCKWISE:
                q = i
        
        p = q
        
        if p == l:
            break
    
    return [points[i] for i in hull]


def performance_analysis(dataset_sizes, distribution_name="custom"):
    """
    Analyzes performance of Jarvis March across different input sizes.
    Generates and saves performance graphs to media/graphs/.
    
    Args:
        dataset_sizes: List of tuples (n, points) where n is the size and points is the list of Point objects
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
    
    sizes = []
    hull_sizes = []
    times = []
    nh_values = []
    
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS - JARVIS MARCH")
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
            hull = convexHull(test_points, len(test_points))
            end = time.perf_counter()
            total_time += (end - start) * 1000  # Convert to ms
            if run == 0:  # Get hull size from first run
                hull_size = len(hull)
        
        avg_time = total_time / num_runs
        sizes.append(n)
        hull_sizes.append(hull_size)
        times.append(avg_time)
        nh_values.append(n * hull_size)
        print(f"  n={n:5d}, h={hull_size:4d}: {avg_time:.4f} ms (avg of {num_runs} runs)")
    
    os.makedirs('media/graphs', exist_ok=True)
    
    # Graph 1: Time vs n (Input Size) - More populated x-axis
    plt.figure(figsize=(12, 6))
    plt.plot(sizes, times, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Input Size (n)', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    plt.title(f"Jarvis March: Time vs Input Size (n)\n{distribution_name.replace('_', ' ').title()} Distribution", 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # Add more x-axis ticks for better visualization
    if len(sizes) > 0:
        plt.xticks(sizes, rotation=45)
    plt.tight_layout()
    
    filename1 = f'media/graphs/jarvis_march_{distribution_name}_time_vs_n.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graph 1 (Time vs n) saved to: {filename1}")
    plt.close()
    
    # Graph 2: Time vs O(nh) - Theoretical Complexity
    plt.figure(figsize=(12, 6))
    plt.plot(nh_values, times, 'mo-', linewidth=2, markersize=8)
    plt.xlabel('O(nh)', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    plt.title(f"Jarvis March: Time vs O(nh)\n{distribution_name.replace('_', ' ').title()} Distribution", 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename2 = f'media/graphs/jarvis_march_{distribution_name}_time_vs_nh.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"✓ Graph 2 (Time vs O(nh)) saved to: {filename2}")
    plt.close()


if __name__ == '__main__':
    import time
    import random
    
    points = [
        Point(5.29, 3.48), Point(9.75, -1.54), Point(11.02, -0.28), Point(10.39, -0.28), 
        Point(11.02, 0.98), Point(8.48, -0.911), Point(9.75, 1), Point(9.72, 1.6), 
        Point(9.12, 2.91), Point(7.85, 2.29), Point(7.21, 1.62), Point(6.57, 0.36), 
        Point(4.65, 2.29), Point(5.29, 0.36), Point(3.38, 2.91), Point(4.03, 1), 
        Point(2.75, 1), Point(5.29, -1.573), Point(3.38, -0.911), Point(5.94, -2.18), 
        Point(2.11, -0.911), Point(4.66, -2.2), Point(3.38, -2.18), Point(7.21, -2.782)
    ]
    
    print("=" * 60)
    print("JARVIS MARCH (GIFT WRAPPING) - CONVEX HULL ALGORITHM")
    print("=" * 60)
    print(f"\nNumber of input points: {len(points)}")
    
    # Measure execution time
    start_time = time.perf_counter()
    hull = convexHull(points, len(points))
    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    print(f"Number of hull points: {len(hull)}")
    print(f"Execution time: {execution_time:.4f} ms")
    
    print("\n" + "-" * 60)
    print("CONVEX HULL COORDINATES:")
    print("-" * 60)
    for i, point in enumerate(hull, 1):
        print(f"{i:2d}. ({point.x:7.3f}, {point.y:7.3f})")
    
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
                print(f"{i:3d}. ({point.x:7.3f}, {point.y:7.3f}) -> Inside")
            elif status == 'on_edge':
                on_edge_count += 1
                print(f"{i:3d}. ({point.x:7.3f}, {point.y:7.3f}) -> On Edge")
        
        print(f"\nSummary: {on_edge_count} points on hull edges, {inside_count} points inside")
        print(f"Total: {on_edge_count + inside_count} points (all points should be inside or on edge)")
    
    
    print("\n" + "=" * 60)
    
    # Uncomment below to run performance analysis
    # NOTE: Requires matplotlib: pip install matplotlib
    # Using larger input sizes (1K-100K) to clearly show O(nh) behavior
    # Smaller sizes are dominated by overhead and don't show theoretical complexity
    
    print("\n")
    test_datasets = [
        (1000, [Point(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(1000)]),
        (5000, [Point(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(5000)]),
        (10000, [Point(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10000)]),
        (50000, [Point(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(50000)]),
        (100000, [Point(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(100000)]),
        (250000, [Point(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(250000)]),
        (500000, [Point(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(500000)]),
        (750000, [Point(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(750000)]),
        (1000000, [Point(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(1000000)]),
    ]
    performance_analysis(test_datasets, distribution_name="uniform_random")
    
    

    
