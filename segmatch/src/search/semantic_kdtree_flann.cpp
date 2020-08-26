#include "segmatch/search/impl/semantic_kdtree_flann.hpp"


namespace search {
    template class SemanticKdTreeFLANN<segmatch::MapPoint>;
    template class SemanticKdTreeFLANN<segmatch::MapPoint, typename flann::L2<float>>;
    // template class SemanticKdTreeFLANN<segmatch::PclPoint>;
    template class SemanticKdTreeFLANN<segmatch::MapPoint, typename search::L2_Color<float>>;
    template class SemanticKdTreeFLANN<segmatch::MapPoint, typename search::L2_Copy<float>>;
    template class SemanticKdTreeFLANN<segmatch::MapPoint, typename search::L2_Semantics<float>>;
}
