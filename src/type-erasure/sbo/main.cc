#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <utility>

namespace sbo {

// A type-erased "Drawable" with a small-buffer optimization (SBO):
// types up to `kBuf` bytes are stored inline; larger types fall back to
// heap storage. Compare with `any-drawable/`, which always heap-allocates.
class AnyDrawable {
  static constexpr std::size_t kBuf = 32;

  struct VTable {
    std::string (*draw)(const void*);
    void (*copy)(void* dst, const void* src);
    void (*move)(void* dst, void* src) noexcept;
    void (*destroy)(void*) noexcept;
    bool inline_storage;
  };

  template <typename T>
  static const VTable* vtable_for() {
    static constexpr bool fits = sizeof(T) <= kBuf && alignof(T) <= alignof(std::max_align_t) && std::is_nothrow_move_constructible_v<T>;
    if constexpr (fits) {
      static const VTable v{
          /*draw*/ [](const void* p) -> std::string { return static_cast<const T*>(p)->draw(); },
          /*copy*/ [](void* dst, const void* src) { ::new (dst) T(*static_cast<const T*>(src)); },
          /*move*/ [](void* dst, void* src) noexcept { ::new (dst) T(std::move(*static_cast<T*>(src))); },
          /*destroy*/ [](void* p) noexcept { static_cast<T*>(p)->~T(); },
          /*inline_storage*/ true,
      };
      return &v;
    } else {
      static const VTable v{
          /*draw*/ [](const void* p) -> std::string { return (*static_cast<const T* const*>(p))->draw(); },
          /*copy*/ [](void* dst, const void* src) { *static_cast<T**>(dst) = new T(**static_cast<const T* const*>(src)); },
          /*move*/
          [](void* dst, void* src) noexcept {
            *static_cast<T**>(dst) = *static_cast<T**>(src);
            *static_cast<T**>(src) = nullptr;
          },
          /*destroy*/ [](void* p) noexcept { delete *static_cast<T**>(p); },
          /*inline_storage*/ false,
      };
      return &v;
    }
  }

  alignas(std::max_align_t) std::byte storage_[kBuf];
  const VTable* vt_ = nullptr;

 public:
  template <typename T>
  AnyDrawable(T x) {
    using U = std::decay_t<T>;
    vt_ = vtable_for<U>();
    if (vt_->inline_storage) {
      ::new (storage_) U(std::move(x));
    } else {
      *reinterpret_cast<U**>(storage_) = new U(std::move(x));
    }
  }

  AnyDrawable(const AnyDrawable& other) : vt_(other.vt_) { vt_->copy(storage_, other.storage_); }

  AnyDrawable(AnyDrawable&& other) noexcept : vt_(other.vt_) {
    vt_->move(storage_, other.storage_);
    other.vt_ = nullptr;
  }

  ~AnyDrawable() {
    if (vt_) vt_->destroy(storage_);
  }

  std::string draw() const { return vt_->draw(storage_); }
  bool stored_inline() const { return vt_->inline_storage; }
};

struct Tiny {
  int n;
  std::string draw() const { return "tiny(" + std::to_string(n) + ")"; }
};

struct Big {
  char pad[256];
  int n;
  std::string draw() const { return "big(" + std::to_string(n) + ")"; }
};

}  // namespace sbo

TEST(SBO, SmallTypeStaysInline) {
  sbo::AnyDrawable d = sbo::Tiny{42};
  EXPECT_TRUE(d.stored_inline());
  EXPECT_EQ(d.draw(), "tiny(42)");
}

TEST(SBO, LargeTypeFallsBackToHeap) {
  sbo::AnyDrawable d = sbo::Big{{}, 7};
  EXPECT_FALSE(d.stored_inline());
  EXPECT_EQ(d.draw(), "big(7)");
}

TEST(SBO, CopyIsIndependentForBothPaths) {
  sbo::AnyDrawable s1 = sbo::Tiny{1};
  sbo::AnyDrawable s2 = s1;
  EXPECT_EQ(s1.draw(), "tiny(1)");
  EXPECT_EQ(s2.draw(), "tiny(1)");

  sbo::AnyDrawable b1 = sbo::Big{{}, 2};
  sbo::AnyDrawable b2 = b1;
  EXPECT_EQ(b1.draw(), "big(2)");
  EXPECT_EQ(b2.draw(), "big(2)");
}
