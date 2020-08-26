#ifndef SEGMATCH_SEMANTIC_KDTREE_H_
#define SEGMATCH_SEMANTIC_KDTREE_H_

#include <vector>
#include "segmatch/search/point_representation_spec.hpp"

#include <flann/flann.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/PointIndices.h>
#include <pcl/search/search.h>
#include <boost/shared_array.hpp>

#include "segmatch/common.hpp"


// TODO(ben): some of this should go in the impl/*.hpp file
namespace search {

  /**
 * Squared Euclidean distance functor.
 *
 * This is the simpler, unrolled version. This is preferable for
 * very low dimensionality data (eg 3D points)
 */
// TODO(ben): move this somewhere else, header or so?
//TODO: specialize for 4D color distance, 4D semantics and 5D color + semantics
template<class T>
struct L2_Color
{
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

        // weight difference in hue by a tunable factor
        // max difference for hue will be 180 (due to symmetry)
        float xyz_result = result;
        float hue_diff = *(a + 3) - *(b + 3);
        result += hue_weight * hue_diff * hue_diff;

        return result;
    }

    template <typename U, typename V>
    inline ResultType accum_dist(const U& a, const V& b, int) const
    {
        return (a-b)*(a-b);
    }

    // TODO: tune this parameter
    static constexpr float hue_weight = 0.01;
};

template<class T>
struct L2_Semantics
{
    typedef bool is_kdtree_distance;

    typedef T ElementType;
    typedef typename flann::Accumulator<T>::Type ResultType;

    ResultType semantic_class_penalty = static_cast<ResultType>(5.0);
    double c1 = 0;


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

        // add penalty if semantic class is different
        // either additive or multiplicative
        if (abs(*a - *b) > 1) {
          result += semantic_class_penalty;
          // results *= semantic_class_penalty;
        }

        return result;
    }

    template <typename U, typename V>
    inline ResultType accum_dist(const U& a, const V& b, int) const
    {
        return (a-b)*(a-b);
    }
};


template<typename PointT, typename Dist = ::flann::L2_Simple<float>>
class SemanticKdTreeFLANN : public pcl::search::Search<PointT> /* : public pcl::KdTree<PointT> */
{
  public:
    //   using KdTree<PointT>::input_;
    //   using KdTree<PointT>::indices_;
    //   using KdTree<PointT>::epsilon_;
    //   using KdTree<PointT>::sorted_;
    //   using KdTree<PointT>::point_representation_;
    //   using KdTree<PointT>::nearestKSearch;
    //   using KdTree<PointT>::radiusSearch;
    typedef PointT PtT;

    // typedef boost::shared_ptr <std::vector<int> > IndicesPtr;
    // typedef boost::shared_ptr <const std::vector<int> > IndicesConstPtr;

    typedef pcl::PointCloud<PointT> PointCloud;
    // typedef boost::shared_ptr<PointCloud> PointCloudPtr;
    typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;

    typedef pcl::SemanticPointRepresentation<PointT> PointRepresentation;
    // //typedef boost::shared_ptr<PointRepresentation> PointRepresentationPtr;
    typedef boost::shared_ptr<const PointRepresentation> PointRepresentationConstPtr;

    // // Boost shared pointers
    // typedef boost::shared_ptr<KdTree<PointT> > Ptr;
    // typedef boost::shared_ptr<const KdTree<PointT> > ConstPtr;

    // typedef typename KdTree<PointT>::PointCloud PointCloud;
    // typedef typename KdTree<PointT>::PointCloudConstPtr PointCloudConstPtr;

    typedef boost::shared_ptr<std::vector<int>> IndicesPtr;
    typedef boost::shared_ptr<const std::vector<int>> IndicesConstPtr;

    typedef ::flann::Index<Dist> FLANNIndex;

      /** \brief Default Constructor for KdTreeFLANN.
    * \param[in] sorted set to true if the application that the tree will be used for requires sorted nearest neighbor indices (default). False otherwise.
    *
    * By setting sorted to false, the \ref radiusSearch operations will be faster.
    */
    SemanticKdTreeFLANN (bool sorted = true);

    // TODO(ben): add other constructors?

    /** \brief Destructor for KdTreeFLANN.
    * Deletes all allocated data arrays and destroys the kd-tree structures.
    */
    virtual ~SemanticKdTreeFLANN ()
    {
      cleanup ();
    }


    bool getSortedResults();


    /** \brief Get a pointer to the input point cloud dataset. */
    inline PointCloudConstPtr
    getInputCloud () const
    {
      return (input_);
    }

    // setInputCloud()
    /** \brief Provide a pointer to the input dataset.
     * \param[in] cloud the const boost shared pointer to a PointCloud message
     * \param[in] indices the point indices subset that is to be used from \a cloud - if NULL the whole cloud is used
     */
    // template<typename PointT, typename Dist>
    void setInputCloud(const PointCloudConstPtr& cloud, const IndicesConstPtr& indices = IndicesConstPtr());


    // TODO: just a placeholder function, requried only s.t. definition for this function inherited from search.h exists. Fix that.
    int nearestKSearch(const PointT& point,
                       int k,
                       std::vector<int>& k_indices,
                       std::vector<float>& k_sqr_distances) const;

    /** \brief Search for all the nearest neighbors of the query point in a given radius.
     *
     * \attention This method does not do any bounds checking for the input index
     * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
     *
     * \param[in] point a given \a valid (i.e., finite) query point
     * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
     * \param[out] k_indices the resultant indices of the neighboring points
     * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
     * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
     * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
     * returned.
     * \return number of neighbors found in radius
     *
     * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
     */
    int radiusSearch(const PointT& point,
                     double radius,
                     std::vector<int>& k_indices,
                     std::vector<float>& k_sqr_distances,
                     unsigned int max_nn = 0) const;


      /** \brief Search for all the nearest neighbors of the query point in a given radius (zero-copy).
      *
      * \attention This method does not do any bounds checking for the input index
      * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
      *
      * \param[in] index a \a valid index representing a \a valid query point in the dataset given
      * by \a setInputCloud. If indices were given in setInputCloud, index will be the position in
      * the indices vector.
      *
      * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
      * \param[out] k_indices the resultant indices of the neighboring points
      * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
      * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
      * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
      * returned.
      * \return number of neighbors found in radius
      *
      * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
      */
      int radiusSearch (int index, double radius, std::vector<int> &k_indices,
                    std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const
      {
        if (indices_ == NULL)
        {
          assert (index >= 0 && index < static_cast<int> (input_->points.size ()) && "Out-of-bounds error in radiusSearch!");
          return (radiusSearch (input_->points[index], radius, k_indices, k_sqr_distances, max_nn));
        }
        else
        {
          assert (index >= 0 && index < static_cast<int> (indices_->size ()) && "Out-of-bounds error in radiusSearch!");
          return (radiusSearch (input_->points[(*indices_)[index]], radius, k_indices, k_sqr_distances, max_nn));
        }
      }

    // simple euclidean clustering using custom distance for color & semantics. Based on PCL version.
      /** \brief Decompose a region of space into clusters based on the Euclidean distance between points
    * \param cloud the point cloud message
    * \param tree the spatial locator (e.g., kd-tree) used for nearest neighbors searching
    * \note the tree has to be created as a spatial locator on \a cloud
    * \param tolerance the spatial cluster tolerance as a measure in L2 Euclidean space
    * \param clusters the resultant clusters containing point indices (as a vector of PointIndices)
    * \param min_pts_per_cluster minimum number of points that a cluster may contain (default: 1)
    * \param max_pts_per_cluster maximum number of points that a cluster may contain (default: max int)
    * \ingroup segmentation
    */
  // template <typename PointT>
  void extractEuclideanClusters (
      const PointCloud &cloud, /* const boost::shared_ptr<segmatch::search::<PointT> > &tree,  */
      float tolerance, std::vector<pcl::PointIndices> &clusters,
      unsigned int min_pts_per_cluster = 1, unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max) ());

    /** \brief A FLANN index object. */
    boost::shared_ptr<FLANNIndex> flann_index_;
  protected:
    /** \brief The input point cloud dataset containing the points we need to use. */
    PointCloudConstPtr input_;

    /** \brief A pointer to the vector of point indices to use. */
    IndicesConstPtr indices_;

    /** \brief Epsilon precision (error bound) for nearest neighbors searches. */
    float epsilon_;

    /** \brief Minimum allowed number of k nearest neighbors points that a viable result must contain. */
    int min_pts_;

    /** \brief Return the radius search neighbours sorted **/
    bool sorted_;

    /** \brief For converting different point structures into k-dimensional vectors for nearest-neighbor search. */
    PointRepresentationConstPtr point_representation_;

  private:
    /** \brief Internal cleanup method. */
    void cleanup();

    /** \brief Converts a PointCloud to the internal FLANN point array representation. Returns the number
     * of points.
     * \param cloud the PointCloud
     */
    void convertCloudToArray(const PointCloud& cloud);

    /** \brief Converts a PointCloud with a given set of indices to the internal FLANN point array
     * representation. Returns the number of points.
     * \param[in] cloud the PointCloud data
     * \param[in] indices the point cloud indices
     */
    void convertCloudToArray(const PointCloud& cloud, const std::vector<int>& indices);

    /** \brief Class getName method. */
    virtual const std::string& getName() const { return ("SemanticKdTreeFLANN"); }


    /** \brief Internal pointer to data. */
    boost::shared_array<float> cloud_;

    /** \brief mapping between internal and external indices. */
    std::vector<int> index_mapping_;

    /** \brief whether the mapping bwwteen internal and external indices is identity */
    bool identity_mapping_;

    /** \brief Tree dimensionality (i.e. the number of dimensions per point). */
    int dim_;

    /** \brief The total size of the data (either equal to the number of points in the input cloud or to the number of
     * indices - if passed). */
    int total_nr_points_;

    /** \brief The KdTree search parameters for K-nearest neighbors. */
    ::flann::SearchParams param_k_;

    /** \brief The KdTree search parameters for radius search. */
    ::flann::SearchParams param_radius_;
    };
    }  // namespace search
#endif // SEGMATCH_SEMANTIC_KDTREE_H_
