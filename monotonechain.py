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


if __name__ == "__main__":
    pts = [
        (0, 3), (2, 2), (1, 1), (2, 1), (3, 0), (0, 0), (3, 3)
    ]
    hull = monotone_chain(pts)
    print("Convex Hull points:")
    for x, y in hull:
        print(f"({x}, {y})")
