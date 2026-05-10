// graphics-colors.cppm - partition
export module graphics:colors;

export struct Color {
  unsigned char r, g, b, a;
};

export constexpr Color Red{255, 0, 0, 255};
export constexpr Color Green{0, 255, 0, 255};
export constexpr Color Blue{0, 0, 255, 255};
