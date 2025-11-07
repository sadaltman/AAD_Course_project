from manim import *
import numpy as np
import math


COUNTERCLOCKWISE = 1
COLLINEAR = 0
CLOCKWISE = -1


class Algorithm(MovingCameraScene):

    def orientation(self, p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]):
        """Determine orientation of triplet (p1, p2, p3)"""
        x1, y1, x2, y2, x3, y3 = *p1, *p2, *p3
        diff = (y3-y2)*(x2-x1) - (y2-y1)*(x3-x2)
        return COUNTERCLOCKWISE if diff > 0 else (CLOCKWISE if diff < 0 else COLLINEAR)

    def dist(self, p1: tuple[float, float], p2: tuple[float, float]):
        """Calculate Euclidean distance between two points"""
        x1, y1, x2, y2 = *p1, *p2
        return math.sqrt((y2-y1)**2 + (x2-x1)**2)
    
    def polar_angle(self, p1: tuple[float, float], p2: tuple[float, float]):
        """Calculate polar angle from p2 to p1"""
        dy = p1[1] - p2[1]
        dx = p1[0] - p2[0]
        return math.atan2(dy, dx)
    
    def graham_scan(self, points: list[tuple[float, float]]):
        """Graham Scan algorithm to find convex hull of a set of points"""
        if len(points) < 3:
            return points
        
        # Find lowest point
        p0 = min(points, key=lambda p: (p[1], p[0]))
        
        # Sort points by polar angle
        sorted_points = sorted(points, key=lambda p: (self.polar_angle(p, p0), self.dist(p, p0)))
        
        hull = [sorted_points[0]]
        for i in range(1, len(sorted_points)):
            while len(hull) >= 2 and self.orientation(hull[-2], hull[-1], sorted_points[i]) != COUNTERCLOCKWISE:
                hull.pop()
            hull.append(sorted_points[i])
        
        return hull
    
    def jarvis_march(self, points: list[tuple[float, float]], m: int):
        """
        Jarvis march (gift wrapping) on a set of points
        Returns hull if successful, False if hull size exceeds m
        """
        n = len(points)
        if n < 3:
            return points
        
        # Find leftmost point
        l = min(range(n), key=lambda i: (points[i][0], points[i][1]))
        
        hull = []
        p = l
        
        # Keep wrapping until we come back to first point
        for _ in range(m + 1):
            hull.append(points[p])
            
            # Find most counterclockwise point
            q = (p + 1) % n
            for i in range(n):
                if self.orientation(points[p], points[i], points[q]) == COUNTERCLOCKWISE:
                    q = i
            
            p = q
            
            # If we've wrapped back to start
            if p == l:
                return hull
        
        # Hull has more than m vertices
        return False

    def split_points(self, points: list[tuple[float, float]], k: int):
        """Split points into k subsets"""
        n = len(points)
        avg = n / k
        subsets = []
        last = 0.0
        
        while last < n:
            subsets.append(points[int(last):int(last + avg)])
            last += avg
        
        return subsets

    def chans_algorithm(self, points: list[tuple[float, float]]):
        """Chan's Algorithm with Manim visualization"""
        
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

        # Title
        title = Text("Chan's Algorithm", font_size=36, color=YELLOW, weight=BOLD).to_edge(UP, buff=0.3).shift(RIGHT * 2)
        self.play(Write(title), run_time=0.5)
        self.wait(0.5)

        # Algorithm iterations
        m = 3
        iteration = 0
        max_iterations = 5
        
        while iteration < max_iterations:
            # Calculate m value for this iteration
            if iteration == 0:
                m_value = m
            else:
                m_value = min(m ** (2 ** iteration), n)
            
            iteration_text = Text(f"Iteration {iteration + 1}: m = {m_value}", 
                                font_size=24, color=BLUE).next_to(title, DOWN, buff=0.2)
            self.play(Write(iteration_text), run_time=0.5)
            
            # Calculate number of subsets
            k = int(np.floor(n / m_value))
            if k < 1:
                k = 1
            
            subset_text = Text(f"Splitting into {k} subsets", 
                             font_size=20, color=GREEN).next_to(iteration_text, DOWN, buff=0.2)
            self.play(Write(subset_text), run_time=0.5)
            self.wait(0.5)

            # Split points into k subsets
            subsets = self.split_points(points, k)
            
            # Colors for different subsets
            subset_colors = [RED, BLUE, GREEN, PURPLE, ORANGE, PINK, TEAL, MAROON]
            
            # Compute sub-hulls using Graham Scan
            sub_hulls = []
            sub_hull_lines = []
            
            phase_text = Text("Phase 1: Computing Sub-Hulls (Graham Scan)", 
                            font_size=20, color=ORANGE).next_to(subset_text, DOWN, buff=0.2)
            self.play(Write(phase_text), run_time=0.5)
            
            for idx, subset in enumerate(subsets):
                if len(subset) < 2:
                    continue
                
                color = subset_colors[idx % len(subset_colors)]
                
                # Highlight subset points
                for p in subset:
                    self.play(points_mobs[p].animate.set_color(color), run_time=0.1)
                
                # Compute hull for this subset
                hull = self.graham_scan(subset)
                sub_hulls.append(hull)
                
                # Draw sub-hull
                hull_mobjects = []
                for i in range(len(hull)):
                    line = Line(
                        points_mobs[hull[i]].get_center(),
                        points_mobs[hull[(i+1) % len(hull)]].get_center(),
                        color=color,
                        stroke_width=2.0,
                        z_index=1
                    )
                    hull_mobjects.append(line)
                    self.play(Create(line), run_time=0.2)
                
                sub_hull_lines.extend(hull_mobjects)
                self.wait(0.3)
            
            # Phase 2: Jarvis March on sub-hulls
            self.play(FadeOut(phase_text), run_time=0.3)
            phase2_text = Text("Phase 2: Jarvis March on Sub-Hulls", 
                             font_size=20, color=ORANGE).next_to(subset_text, DOWN, buff=0.2)
            self.play(Write(phase2_text), run_time=0.5)
            
            # Flatten all sub-hull points
            all_sub_hull_points = []
            for hull in sub_hulls:
                all_sub_hull_points.extend(hull)
            
            # Remove duplicates
            all_sub_hull_points = list(set(all_sub_hull_points))
            
            # Run Jarvis march with limit m
            final_hull = self.jarvis_march(all_sub_hull_points, m_value)
            
            if final_hull is not False:
                # Success! Draw final hull
                self.play(
                    FadeOut(iteration_text),
                    FadeOut(subset_text),
                    FadeOut(phase2_text),
                    *[FadeOut(line) for line in sub_hull_lines],
                    run_time=0.5
                )
                
                success_text = Text("Final Convex Hull Found!", 
                                  font_size=32, color=GREEN, weight=BOLD).to_edge(UP, buff=0.3).shift(RIGHT * 2)
                self.play(ReplacementTransform(title, success_text), run_time=0.5)
                
                # Draw final hull
                final_hull_lines = []
                for i in range(len(final_hull)):
                    line = Line(
                        points_mobs[final_hull[i]].get_center(),
                        points_mobs[final_hull[(i+1) % len(final_hull)]].get_center(),
                        color=ORANGE,
                        stroke_width=4.0,
                        z_index=2
                    )
                    final_hull_lines.append(line)
                    self.play(Create(line), run_time=0.3)
                
                # Highlight hull points
                anims = []
                for p in points:
                    if p in final_hull:
                        anims.append(points_mobs[p].animate.set_color(ORANGE).scale(1.2))
                    else:
                        anims.append(points_mobs[p].animate.set_opacity(0.3))
                
                self.play(anims, run_time=0.8)
                self.wait(3)
                return
            else:
                # Need to increase m
                self.play(
                    FadeOut(iteration_text),
                    FadeOut(subset_text),
                    FadeOut(phase2_text),
                    *[FadeOut(line) for line in sub_hull_lines],
                    run_time=0.5
                )
                
                fail_text = Text(f"Hull > {m_value} vertices. Increasing m...", 
                               font_size=22, color=RED).next_to(title, DOWN, buff=0.2)
                self.play(Write(fail_text), run_time=0.5)
                self.wait(0.8)
                self.play(FadeOut(fail_text), run_time=0.3)
                
                # Reset point colors
                for p in points:
                    self.play(points_mobs[p].animate.set_color(WHITE), run_time=0.05)
                
                iteration += 1

    def construct(self):
        points = [
            (5.29, 3.48), (9.75, -1.54), (11.02, -0.28), (10.39, -0.28), (11.02, 0.98),
            (8.48, -0.911), (9.75, 1), (9.72, 1.6), (9.12, 2.91), (7.85, 2.29), (7.21, 1.62),
            (6.57, 0.36), (4.65, 2.29), (5.29, 0.36), (3.38, 2.91), (4.03, 1), (2.75, 1),
            (5.29, -1.573), (3.38, -0.911), (5.94, -2.18), (2.11, -0.911), (4.66, -2.2),
            (3.38, -2.18), (7.21, -2.782)
        ]
        self.chans_algorithm(points)


# Non-visual implementations
def orientation(p1, p2, p3):
    """Determine orientation of triplet"""
    x1, y1, x2, y2, x3, y3 = *p1, *p2, *p3
    diff = (y3-y2)*(x2-x1) - (y2-y1)*(x3-x2)
    return COUNTERCLOCKWISE if diff > 0 else (CLOCKWISE if diff < 0 else COLLINEAR)


def dist(p1, p2):
    """Calculate distance"""
    x1, y1, x2, y2 = *p1, *p2
    return math.sqrt((y2-y1)**2 + (x2-x1)**2)


def polar_angle(p1, p2):
    """Calculate polar angle"""
    dy = p1[1] - p2[1]
    dx = p1[0] - p2[0]
    return math.atan2(dy, dx)


def graham_scan(points):
    """Graham Scan algorithm"""
    if len(points) < 3:
        return points
    
    p0 = min(points, key=lambda p: (p[1], p[0]))
    sorted_points = sorted(points, key=lambda p: (polar_angle(p, p0), dist(p, p0)))
    
    hull = [sorted_points[0]]
    for i in range(1, len(sorted_points)):
        while len(hull) >= 2 and orientation(hull[-2], hull[-1], sorted_points[i]) != COUNTERCLOCKWISE:
            hull.pop()
        hull.append(sorted_points[i])
    
    return hull


def jarvis_march(points, m):
    """Jarvis march with limit m"""
    n = len(points)
    if n < 3:
        return points
    
    l = min(range(n), key=lambda i: (points[i][0], points[i][1]))
    
    hull = []
    p = l
    
    for _ in range(m + 1):
        hull.append(points[p])
        
        q = (p + 1) % n
        for i in range(n):
            if orientation(points[p], points[i], points[q]) == COUNTERCLOCKWISE:
                q = i
        
        p = q
        
        if p == l:
            return hull
    
    return False


def split_points(points, k):
    """Split points into k subsets"""
    n = len(points)
    avg = n / k
    subsets = []
    last = 0.0
    
    while last < n:
        subsets.append(points[int(last):int(last + avg)])
        last += avg
    
    return subsets


def chans_algorithm(points):
    """Chan's Algorithm"""
    n = len(points)
    m = 3
    
    for iteration in range(10):  # Max iterations
        if iteration == 0:
            m_value = m
        else:
            m_value = min(m ** (2 ** iteration), n)
        
        # Split into subsets
        k = max(1, int(np.floor(n / m_value)))
        subsets = split_points(points, k)
        
        # Compute sub-hulls
        sub_hulls = []
        for subset in subsets:
            if len(subset) >= 2:
                hull = graham_scan(subset)
                sub_hulls.append(hull)
        
        # Flatten sub-hull points
        all_sub_hull_points = []
        for hull in sub_hulls:
            all_sub_hull_points.extend(hull)
        all_sub_hull_points = list(set(all_sub_hull_points))
        
        # Run Jarvis march
        final_hull = jarvis_march(all_sub_hull_points, m_value)
        
        if final_hull is not False:
            return final_hull
    
    return None


if __name__ == '__main__':
    points = [
        (0, 3), (2, 2), (1, 1), (2, 1),
        (3, 0), (0, 0), (3, 3)
    ]
    
    hull = chans_algorithm(points)
    if hull:
        print("Convex Hull points:")
        for point in hull:
            print(f"({point[0]}, {point[1]})")
