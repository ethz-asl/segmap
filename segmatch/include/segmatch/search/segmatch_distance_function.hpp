#ifndef SEGMATCH_DISTANCE_FUNCTION_H_
#define SEGMATCH_DISTANCE_FUNCTION_H_

namespace pcl {
template<class T>
struct L2_Segmatch {
  public:
    typedef bool is_kdtree_distance;

    typedef T ElementType;
    typedef typename flann::Accumulator<T>::Type ResultType;

    template <typename K> std::string type_name();
    template <typename Iterator1, typename Iterator2>
    ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType /*worst_dist*/ = -1) const
    {
        ResultType result = ResultType();
        ResultType diff;

        // use unweighted squared difference for x, y and z coords
        for(size_t i = 0; i < 3; ++i ) {
            diff = *a++ - *b++;
            result += diff*diff;
        }

        if (size > 3 && abs(*a++ - *b++) > 1) {
            result += class_penalty;
        }

        if (size > 4) {
            diff = abs(*a++ - *b++);
            if (diff < 0.1 || diff > 0.9) {
              //result += color_weight*diff*diff;
              result += color_weight;
            }
        }

        return result;
    }

    template <typename U, typename V>
    inline ResultType accum_dist(const U& a, const V& b, int) const
    {
        return (a-b)*(a-b);
    }

  private:
    // TODO(smauq): parameter tuniung required
    static constexpr ResultType class_penalty = static_cast<ResultType>(0.0);
    static constexpr ResultType color_weight = static_cast<ResultType>(0.0);
};
}

#endif  // SEGMATCH_DISTANCE_FUNCTION_H_
