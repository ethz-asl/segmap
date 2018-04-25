#ifndef SEGMATCH_GRAPH_UTILITIES_HPP_
#define SEGMATCH_GRAPH_UTILITIES_HPP_

#include <iostream>
#include <fstream>

#include <boost/graph/connected_components.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/graphviz.hpp>
#include <glog/logging.h>

namespace segmatch {

/// \brief Provide generic graph utility functions.
class GraphUtilities {
 public:
  /// \brief Prevent instantiation of static class.
  GraphUtilities() = delete;

  /// \brief Writes the specified graph to a .dot file that can be visualized in graphviz
  /// ( http://www.graphviz.org/ ).
  /// \param graph The graph that needs to be saved.
  /// \param file_name The destination file.
  template<typename Graph>
  static void saveGraphForGraphviz(const Graph& graph, const std::string& file_name) {
    std::ofstream output_file;
    output_file.open(file_name);
    if (output_file.is_open()) {
      boost::write_graphviz(output_file, graph);
    } else {
      LOG(ERROR) << "Unable to write graph to file: " << file_name;
    }
  }

  /// \brief Finds the vertices of a graph belonging to the a maximum clique. Only one maximum
  /// clique is returned.
  /// Closely follows the exact algorithm described in:
  /// "Fast Algorithms for the Maximum Clique Problem on Massive Sparse Graphs"
  /// Pattabiraman, Bharath Mostofa Ali Patwary, Md Gebremedhin, Assefaw H Liao, Wei-keng
  /// Choudhary, Alok ( https://arxiv.org/pdf/1209.5818.pdf )
  /// The algorithm is modified so that vertices are visited in increasing degeneracy order. This
  /// limits the search depth to the degeneracy of the graph.
  /// \param graph The input graph. The graph must be indirected and the underlying data structure
  /// must support random access.
  /// \param min_clique_size The minimum size of the maximum clique, smaller cliques will be
  /// ignored. Must be greater or equal 2.
  /// \returns Vector containing the vertices belonging to a maximum clique. If the vector is
  /// empty, no clique with the specified minimum size exists.
  template<typename Graph>
  static std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> findMaximumClique(
      const Graph& graph, const size_t min_clique_size) {
    // Ensure that the graph type is supported and define type shortcuts.
    CHECK(min_clique_size >= 2);
    assertIsUndirectedAndRandomAccessGraph(graph);
    typedef boost::graph_traits<Graph> GraphTraits;
    typedef typename GraphTraits::vertex_descriptor Vertex;

    const size_t n_vertices = boost::num_vertices(graph);
    std::vector<Vertex> neighbors;
    neighbors.reserve(n_vertices);

    std::vector<Vertex> maximum_clique_tmp;
    std::vector<Vertex> maximum_clique;
    maximum_clique_tmp.reserve(n_vertices);
    size_t max_found_size = min_clique_size - 1u;

    // Use bin-sort to sort the vertex indices in increasing degree order.
    std::vector<size_t> bin_starts;
    std::vector<Vertex> sorted_vertices;
    std::vector<size_t> vertex_positions;
    std::vector<size_t> vertex_degrees;
    binSortVerticesByDegree(graph, bin_starts, sorted_vertices, vertex_positions, vertex_degrees);

    // Try to find a clique starting from each vertex.
    for (size_t i = 0u; i < sorted_vertices.size(); ++i) {
      const Vertex vertex = sorted_vertices[i];
      const size_t vertex_degree = vertex_degrees[vertex];

      // Skip the vertex if it doesn't have enough neighbors to be a maximum clique.
      if (vertex_degree >= max_found_size) {
        neighbors.clear();

        // Collect all the neighbors that have enough neighbors to be a maximum clique.
        typename GraphTraits::out_edge_iterator e_it, e_end;
        for (boost::tie(e_it, e_end) = boost::out_edges(vertex, graph); e_it != e_end; ++e_it) {
          const Vertex neighbor = boost::target(*e_it, graph);
          if(vertex_positions[neighbor] > vertex_positions[vertex] &&
              vertex_degrees[neighbor] >= max_found_size)
            neighbors.push_back(neighbor);
        }

        // Get the size of the maximum clique contained in the subgraph defined by the current vertex
        // and its neighbors.
        const size_t new_found_size = findMaximumCliqueSubset(graph, neighbors, vertex_degrees, 1u,
                                                              max_found_size, maximum_clique_tmp);

        // If a bigger clique is found, set it as the new maximum clique.
        if(new_found_size > max_found_size) {
          max_found_size = new_found_size;
          maximum_clique_tmp.push_back(vertex);
          maximum_clique = std::move(maximum_clique_tmp);
        } else {
          maximum_clique_tmp.clear();
        }
      }

      // Decrease the degree of neighbor vertices of higher degree. This is equivalent to removing
      // this vertex and the incident edges.
      typename GraphTraits::out_edge_iterator e_it, e_end;
      for (boost::tie(e_it, e_end) = boost::out_edges(sorted_vertices[i], graph);
           e_it != e_end; ++e_it) {
        const Vertex neighbor = boost::target(*e_it, graph);
        const size_t neighbor_degree = vertex_degrees[neighbor];
        if (neighbor_degree > vertex_degree) {
          const size_t neighbor_position = vertex_positions[neighbor];
          const size_t swapped_neighbor_position = bin_starts[neighbor_degree];
          const Vertex swapped_neighbor = sorted_vertices[swapped_neighbor_position];
          if (neighbor != swapped_neighbor) {
            vertex_positions[neighbor] = swapped_neighbor_position;
            vertex_positions[swapped_neighbor] = neighbor_position;
            sorted_vertices[neighbor_position] = swapped_neighbor;
            sorted_vertices[swapped_neighbor_position] = neighbor;
          }
          ++bin_starts[neighbor_degree];
          --vertex_degrees[neighbor];
        }
      }
    }

    return maximum_clique;
  }

  /// \brief Finds the vertex degrees and the maximum vertex degree in the graph.
  /// \param graph The input graph. The graph must be indirected and the underlying data structure
  /// must support random access.
  /// \param vertex_degrees Vector in which the vertex degrees will be stored.
  /// \returns Maximum vertex degree in the graph.
  template<typename Graph>
  static size_t getVertexDegreesAndGraphMaxDegree(const Graph& graph,
                                                  std::vector<size_t>& vertex_degrees) {
    // Ensure that the graph type is supported and define type shortcuts.
    assertIsUndirectedAndRandomAccessGraph(graph);

    // Get and store the vertex degrees.
    vertex_degrees.clear();
    vertex_degrees.resize(num_vertices(graph));
    size_t maximum_degree = 0u;
    typename boost::graph_traits<Graph>::vertex_iterator v_it, v_end;
    for (boost::tie(v_it, v_end) = boost::vertices(graph); v_it != v_end; ++v_it) {
      vertex_degrees[*v_it] = boost::out_degree(*v_it, graph);
      maximum_degree = std::max(maximum_degree, vertex_degrees[*v_it]);
    }
    return maximum_degree;
  }

 private:
  // Statically verify that a graph is undirected and based on data structures that allow random
  // access.
  template<typename Graph>
  static void assertIsUndirectedAndRandomAccessGraph(const Graph& graph) {
    BOOST_CONCEPT_ASSERT((boost::concepts::GraphConcept<Graph>));
    typedef boost::graph_traits<Graph> GraphTraits;
    typedef typename GraphTraits::vertex_descriptor Vertex;
    static_assert(std::is_same<typename GraphTraits::directed_category,
                               boost::undirected_tag>::value,
                  "GraphUtilities::findMaximumKCore only supports undirected graphs");
    static_assert(std::is_same<Vertex, size_t>::value,
                  "GraphUtilities::findMaximumKCore only supports graphs with vertex descriptors "
                  "of type size_t (usually graphs based on random access containers).");
  }

  // Sort the vertices of a graph in increasing vertex degree order using bin-sorting.
  template<typename Graph>
  static size_t binSortVerticesByDegree(
      const Graph& graph, std::vector<size_t>& bin_starts,
      std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>& sorted_vertices,
      std::vector<size_t>& vertex_positions, std::vector<size_t>& vertex_degrees) {

    // Ensure that the graph type is supported.
    assertIsUndirectedAndRandomAccessGraph(graph);
    typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;

    // Get and store the vertex degrees.
    size_t maximum_degree = getVertexDegreesAndGraphMaxDegree(graph, vertex_degrees);

    // Use bin-sort to sort the vertex indices in increasing degree order.
    // 1) Find the size of each bin.
    std::vector<size_t> bin_sizes(maximum_degree + 1u);
    for (const auto degree : vertex_degrees) ++bin_sizes[degree];

    // 2) Find the starting index of each bin.
    bin_starts.resize(maximum_degree + 1u);
    size_t next_bin_start = 0u;
    for (size_t i = 0u; i < bin_sizes.size(); ++i) {
      bin_starts[i] = next_bin_start;
      next_bin_start += bin_sizes[i];
    }

    // 3) Sort vertex indices
    std::vector<size_t> bin_offsets(bin_starts);
    sorted_vertices.resize(boost::num_vertices(graph));
    vertex_positions.resize(boost::num_vertices(graph));
    typename boost::graph_traits<Graph>::vertex_iterator v_it, v_end;
    for (boost::tie(v_it, v_end) = boost::vertices(graph); v_it != v_end; ++v_it) {
      vertex_positions[*v_it] = bin_offsets[vertex_degrees[*v_it]]++;
      sorted_vertices[vertex_positions[*v_it]] = *v_it;
    }
  }

  // Helper recursive function for the findMaximumClique() function.
  template<typename Graph>
  static size_t findMaximumCliqueSubset(
      const Graph& graph,
      std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>& subset,
      const std::vector<size_t>& vertex_degrees,
      const size_t clique_size, size_t max_found_size,
      std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>& maximum_clique_tmp) {
    // Ensure that the graph type is supported and define type shortcuts.
    assertIsUndirectedAndRandomAccessGraph(graph);
    typedef boost::graph_traits<Graph> GraphTraits;
    typedef typename GraphTraits::vertex_descriptor Vertex;

    const size_t n_vertices = boost::num_vertices(graph);
    std::vector<Vertex> neighbors;
    neighbors.reserve(n_vertices);

    // Final step of the recursion: if there are no more vertices to process, the search is
    // complete.
    if(subset.empty()) {
      if(clique_size > max_found_size) {
        maximum_clique_tmp.clear();
        return clique_size;
      }
      return max_found_size;
    }

    // Process the given subset of vertices.
    while(!subset.empty()) {
      // Continue the search only if there are enough remaining candidates.
      if(clique_size + subset.size() <= max_found_size) break;
      Vertex vertex = subset.back();
      subset.pop_back();

      // Collect the vertices that have enough neighbors and are connected to the current vertex.
      for (const Vertex candidate : subset) {
        if (vertex_degrees[candidate] >= max_found_size &&
            boost::edge(vertex, candidate, graph).second)
            neighbors.push_back(candidate);
      }

      // Get the size of the maximum clique contained in the subgraph defined by the current vertex
      // and its neighbors.
      const size_t new_found_size = findMaximumCliqueSubset(graph, neighbors, vertex_degrees,
                                                            clique_size + 1u, max_found_size,
                                                            maximum_clique_tmp);

      // If a bigger clique is found, use the current vertex.
      if(new_found_size > max_found_size) {
        max_found_size = new_found_size;
        maximum_clique_tmp.push_back(vertex);
      }
      neighbors.clear();
    }

    return max_found_size;
  }
}; // class GraphUtilities

} // namespace segmatch

#endif // SEGMATCH_GRAPH_UTILITIES_HPP_
