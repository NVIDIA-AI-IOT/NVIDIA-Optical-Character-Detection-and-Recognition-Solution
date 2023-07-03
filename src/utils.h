#ifndef NVOCDR_UTIL_HEADER
#define NVOCDR_UTIL_HEADER
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <iostream>

namespace nvocdr
{
    
typedef struct
{
    int x1, y1, x2, y2, x3, y3, x4, y4;
} Polygon;

typedef struct
{
    int x, y;
} Point2d;

typedef struct
{
    Point2d leftTop;
    Point2d rightTop;
    Point2d leftBottom;
    Point2d rightBottom;
} Box2d;

class AutoProfiler {
 public:
  AutoProfiler(std::string name)
      : m_name(std::move(name)),
        m_beg(std::chrono::high_resolution_clock::now()) { }
  ~AutoProfiler() {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_beg);
    std::cout << m_name << " : " << dur.count() << " ms\n";
  }
 private:
  std::string m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
};

// template<typename Duration = std::chrono::milliseconds,
//          typename F,
//          typename ... Args>
// typename Duration::rep profile(int times, F&& fun,  Args&&... args) {
//   const auto beg = std::chrono::high_resolution_clock::now();
//   for (int i = 0; i < times; ++i)
//     std::forward<F>(fun)(std::forward<Args>(args)...);
//   const auto end = std::chrono::high_resolution_clock::now();
//   return std::chrono::duration_cast<Duration>(end - beg).count();
// }

}

#endif