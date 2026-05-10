// graphics.cppm - primary interface
export module graphics;

export import :shapes;
export import :colors;

export double distance(const Point& a, const Point& b) {
  double dx = b.x - a.x;
  double dy = b.y - a.y;
  return dx * dx + dy * dy;  // squared distance
}
