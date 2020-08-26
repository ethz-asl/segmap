#ifndef SEMANTIC_KDTREE_FLANN_IMPL_H_
#define SEMANTIC_KDTREE_FLANN_IMPL_H_

#include "segmatch/search/semantic_kdtree_flann.hpp"

namespace search {

template<typename PointT, typename Dist>
SemanticKdTreeFLANN<PointT, Dist>::SemanticKdTreeFLANN(bool sorted)
  : sorted_(sorted)
  , flann_index_()
  , cloud_()
  , index_mapping_()
  , identity_mapping_(false)
  , dim_(0)
  , total_nr_points_(0)
  , param_k_(::flann::SearchParams(-1, epsilon_))
  , param_radius_(::flann::SearchParams(-1, epsilon_, sorted))
  , pcl::search::Search<PointT> ("KdTree", sorted)
  // , point_representation_(new pcl::DefaultPointRepresentation<PointT>())
  , point_representation_(new pcl::SemanticPointRepresentation<PointT>())
{
}


template <typename PointT, typename Dist> bool
SemanticKdTreeFLANN<PointT, Dist>::getSortedResults()
{
  return (sorted_);
}


template <typename PointT, typename Dist>
void SemanticKdTreeFLANN<PointT, Dist>::setInputCloud (const PointCloudConstPtr &cloud, const IndicesConstPtr &indices)
{

  cleanup ();   // Perform an automatic cleanup of structures

  epsilon_ = 0.0f;   // default error bound value
  // TODO(ben): improve how this works. Could use the following like in original but note that PCL POint will contain 6
  // or 7 dimensions instead of 4 or 5.
  dim_ = point_representation_->getNumberOfDimensions (); // Number of dimensions
  //point_representation_->printNumDims();

  input_   = cloud;
  indices_ = indices;

  // Allocate enough data
  if (!input_)
  {
    // PCL_ERROR ("[pcl::KdTreeFLANN::setInputCloud] Invalid input!\n");
    std::cout << "ERROR! Invalid input!" << std::endl;
    return;
  }
  if (indices != NULL)
  {
    convertCloudToArray (*input_, *indices_);
  }
  else
  {
    convertCloudToArray (*input_);
  }
  total_nr_points_ = static_cast<int> (index_mapping_.size ());
  if (total_nr_points_ == 0)
  {
    // PCL_ERROR ("[pcl::KdTreeFLANN::setInputCloud] Cannot create a KDTree with an empty input cloud!\n");
    std::cout << "ERROR! total_nr_points == 0!" << std::endl;
    return;
  }
  flann_index_.reset (new FLANNIndex (::flann::Matrix<float> (cloud_.get (),
                                                              index_mapping_.size (),
                                                              dim_),
                                      ::flann::KDTreeSingleIndexParams (15))); // max 15 points/leaf
  flann_index_->buildIndex ();
  // std::cout << "flann_index_ type: " << flann_index_->getType() << std::endl;
}

template <typename PointT, typename Dist> int
search::SemanticKdTreeFLANN<PointT, Dist>::nearestKSearch (const PointT &point, int k,
                                                std::vector<int> &k_indices,
                                                std::vector<float> &k_distances) const
{
  std::cout << "WARNING: Calling placeholder function nearestKSearch - implement if it needs to be used or remove the reference!!!!\n";
  return -1;
}


template<typename PointT, typename Dist>
int
search::SemanticKdTreeFLANN<PointT, Dist>::radiusSearch(const PointT& point,
                                                        double radius,
                                                        std::vector<int>& k_indices,
                                                        std::vector<float>& k_sqr_dists,
                                                        unsigned int max_nn) const
{
    assert(point_representation_->isValid(point) && "Invalid (NaN, Inf) point coordinates given to radiusSearch!");

    std::vector<float> query(dim_);
    point_representation_->vectorize(static_cast<PointT>(point), query);

    // Has max_nn been set properly?
    if (max_nn == 0 || max_nn > static_cast<unsigned int>(total_nr_points_))
        max_nn = total_nr_points_;

    std::vector<std::vector<int>> indices(1);
    std::vector<std::vector<float>> dists(1);

    ::flann::SearchParams params(param_radius_);
    if (max_nn == static_cast<unsigned int>(total_nr_points_))
        params.max_neighbors = -1; // return all neighbors in radius
    else
        params.max_neighbors = max_nn;

    int neighbors_in_radius = flann_index_->radiusSearch(
      ::flann::Matrix<float>(&query[0], 1, dim_), indices, dists, static_cast<float>(radius * radius), params);

    k_indices = indices[0];
    k_sqr_dists = dists[0];

    // Do mapping to original point cloud
    if (!identity_mapping_) {
        for (int i = 0; i < neighbors_in_radius; ++i) {
            int& neighbor_index = k_indices[i];
            neighbor_index = index_mapping_[neighbor_index];
        }
    }

    return (neighbors_in_radius);
}

template <typename PointT, typename Dist> void
search::SemanticKdTreeFLANN<PointT, Dist>::cleanup ()
{
  // Data array cleanup
  index_mapping_.clear ();

  if (indices_)
    indices_.reset ();
}


template <typename PointT, typename Dist> void
search::SemanticKdTreeFLANN<PointT, Dist>::convertCloudToArray (const PointCloud &cloud)
{
  // No point in doing anything if the array is empty
  if (cloud.points.empty ())
  {
    cloud_.reset ();
    return;
  }

  int original_no_of_points = static_cast<int> (cloud.points.size ());

  cloud_.reset (new float[original_no_of_points * dim_]);
  float* cloud_ptr = cloud_.get ();
  index_mapping_.reserve (original_no_of_points);
  identity_mapping_ = true;

  for (int cloud_index = 0; cloud_index < original_no_of_points; ++cloud_index)
  {
    // Check if the point is invalid
    if (!point_representation_->isValid (cloud.points[cloud_index]))
    {
      identity_mapping_ = false;
      continue;
    }

    index_mapping_.push_back (cloud_index);

    point_representation_->vectorize (cloud.points[cloud_index], cloud_ptr);
    cloud_ptr += dim_;
  }
}


template <typename PointT, typename Dist> void
SemanticKdTreeFLANN<PointT, Dist>::convertCloudToArray (const PointCloud &cloud, const std::vector<int> &indices)
{
  // No point in doing anything if the array is empty
  if (cloud.points.empty ())
  {
    cloud_.reset ();
    return;
  }

  int original_no_of_points = static_cast<int> (indices.size ());

  cloud_.reset (new float[original_no_of_points * dim_]);
  float* cloud_ptr = cloud_.get ();
  index_mapping_.reserve (original_no_of_points);
  // its a subcloud -> false
  // true only identity:
  //     - indices size equals cloud size
  //     - indices only contain values between 0 and cloud.size - 1
  //     - no index is multiple times in the list
  //     => index is complete
  // But we can not guarantee that => identity_mapping_ = false
  identity_mapping_ = false;

  for (std::vector<int>::const_iterator iIt = indices.begin (); iIt != indices.end (); ++iIt)
  {
    // Check if the point is invalid
    if (!point_representation_->isValid (cloud.points[*iIt]))
      continue;

    // map from 0 - N -> indices [0] - indices [N]
    index_mapping_.push_back (*iIt);  // If the returned index should be for the indices vector

    point_representation_->vectorize (cloud.points[*iIt], cloud_ptr);
    cloud_ptr += dim_;
  }
}


template <typename PointT, typename Dist> void
SemanticKdTreeFLANN<PointT, Dist>::extractEuclideanClusters (const PointCloud &cloud,
                               /* const boost::shared_ptr<search::Search<PointT> > &tree, */
                               float tolerance, std::vector<pcl::PointIndices> &clusters,
                               unsigned int min_pts_per_cluster,
                               unsigned int max_pts_per_cluster)
{
  if (getInputCloud ()->points.size () != cloud.points.size ())
  {
      std::cout << "Tree built for a different point cloud dataset (" << getInputCloud()->points.size()
                << ") than the input cloud (" << cloud.points.size() << ")!\n";
      return;
  }
  // Check if the tree is sorted -- if it is we don't need to check the first element
  // int nn_start_idx = 0;  // tree->getSortedResults () ? 1 : 0;
  int nn_start_idx = getSortedResults () ? 1 : 0;
  // Create a bool vector of processed point indices, and initialize it to false
  std::vector<bool> processed (cloud.points.size (), false);

  std::vector<int> nn_indices;
  std::vector<float> nn_distances;
  // Process all points in the indices vector
  for (int i = 0; i < static_cast<int> (cloud.points.size ()); ++i)
  {
    if (processed[i])
      continue;

    std::vector<int> seed_queue;
    int sq_idx = 0;
    seed_queue.push_back (i);

    processed[i] = true;
    while (sq_idx < static_cast<int> (seed_queue.size ()))
    {
      // Search for sq_idx
      if (!radiusSearch (seed_queue[sq_idx], tolerance, nn_indices, nn_distances))
      {
        sq_idx++;
        continue;
      }

      for (size_t j = nn_start_idx; j < nn_indices.size (); ++j)             // can't assume sorted (default isn't!)
      {
        if (nn_indices[j] == -1 || processed[nn_indices[j]])        // Has this point been processed before ?
          continue;

        // Perform a simple Euclidean clustering
        seed_queue.push_back (nn_indices[j]);
        processed[nn_indices[j]] = true;
      }

      sq_idx++;
    }

    // If this queue is satisfactory, add to the clusters
    if (seed_queue.size () >= min_pts_per_cluster && seed_queue.size () <= max_pts_per_cluster)
    {
      pcl::PointIndices r;
      r.indices.resize (seed_queue.size ());
      for (size_t j = 0; j < seed_queue.size (); ++j)
        r.indices[j] = seed_queue[j];

      // These two lines should not be needed: (can anyone confirm?) -FF
      std::sort (r.indices.begin (), r.indices.end ());
      r.indices.erase (std::unique (r.indices.begin (), r.indices.end ()), r.indices.end ());

      r.header = cloud.header;
      clusters.push_back (r);   // We could avoid a copy by working directly in the vector
      // std::string cyan = "\033[0;36m";
      // std::string reset = "\033[0m";
      // std::cout << cyan << "cluster size: " << r.indices.size() << reset << std::endl;
    }
  }
}

// #define PCL_INSTANTIATE_SemanticKdTreeFLANN(T) template class PCL_EXPORTS search::SemanticKdTreeFLANN<T>;
}
#endif  //#ifndef _SEMANTIC_KDTREE_FLANN_IMPL_H_
