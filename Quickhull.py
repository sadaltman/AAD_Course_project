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


if __name__ == '__main__':
    points = [
        [0, 3], [1, 1], [2, 2], [4, 4],
        [0, 0], [1, 2], [3, 1], [3, 3]
    ]
    n = len(points)
    printHull(points, n)
