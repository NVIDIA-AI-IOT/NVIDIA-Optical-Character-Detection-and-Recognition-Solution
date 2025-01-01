#include <chrono>
#include <deque>
#include  <numeric>

namespace nvocdr
{
    
template <typename T, size_t Capacity>
class CyclicBuffer : public std::deque<T> {
public:
    void push(const T& value) {
        if (this->size() == Capacity) {
           this->pop_front();
        }
        this->push_back(value);
    }
};


template<size_t HistorySize>
class Timer{
public:
  inline void Start() { mStartTime = std::chrono::high_resolution_clock::now(); }
  inline void Stop() {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - mStartTime);
    mHistory.push(duration);
  }
  inline float getMean() {

    if (mHistory.empty()){
        return 0;
    }
    auto mean = std::accumulate(mHistory.begin(), mHistory.end(), std::chrono::milliseconds(0)) / mHistory.size();
    return mean.count();
  }
  inline float getLast() {
    return mHistory.back().count();
  }
private:
  friend std::ostream& operator<<(std::ostream& o,  Timer& timer) {
    timer.Stop();
    o << timer.getLast() << "ms";
    return o;
  }
  std::chrono::high_resolution_clock::time_point mStartTime;
  CyclicBuffer<std::chrono::milliseconds, HistorySize> mHistory;
};
    
} // namespace nvocdr