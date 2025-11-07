from manim import *
import numpy as np
import math


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


if __name__ == '__main__':
    points = [
        Point(0, 3),
        Point(2, 2),
        Point(1, 1),
        Point(2, 1),
        Point(3, 0),
        Point(0, 0),
        Point(3, 3)
    ]
    
    hull = convexHull(points, len(points))
    print("Convex Hull points:")
    for point in hull:
        print(f"({point.x}, {point.y})")
