#ifndef SEGMATCH_ID_PAIR_HASH_HPP_
#define SEGMATCH_ID_PAIR_HASH_HPP_

#include <functional>

#include "common.hpp"

namespace segmatch {

/// \brief Struct providing an hashing function for pairs of segment IDs.
struct IdPairHash {
  /// \brief Hashing function for pairs of segment IDs.
  /// \param pair Pair of IDs to be hashed.
  /// \returns The hash of the ID pair.
  size_t operator() (const IdPair& pair) const {
    static_assert(std::is_same<IdPair, std::pair<int64_t, int64_t>>::value,
                  "The hashing function is valid only if IdPair is defined as "
                  "std::pair<int64_t, int64_t>");
    // We expect IDs to be always positive, which enables this trick for combining the two IDs. If
    // that would not be the case the hashing function could be less efficient, but still
    // functional.
    return std::hash<uint64_t>{}(static_cast<uint64_t>(pair.first) << 1 +
                                 static_cast<uint64_t>(pair.second));
  }
};

} // namespace segmatch

#endif // SEGMATCH_ID_PAIR_HASH_HPP_
