#include <boost/graph/adjacency_list.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "segmatch/recognizers/graph_utilities.hpp"
using namespace boost;
using namespace segmatch;

// Initialize common objects needed by multiple tests.
class GraphUtilitiesTest : public ::testing::Test {
 protected:
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
  typedef graph_traits<Graph>::vertex_descriptor Vertex;
  typedef std::vector<Vertex> Vertices;

  Graph graph_1;
  Graph graph_2;
  Graph graph_3;

  void SetUp() override {
    graph_1 = buildGraph(4, { { 0, 2 }, { 1, 2 }, { 1, 3 }, { 2, 3 } });
    graph_2 = buildGraph(8, {
        { 0, 5 }, { 0, 7 }, { 1, 3 }, { 1, 5 }, { 2, 3 }, { 2, 4 }, { 2, 6 }, { 2, 7 }, { 3, 4 },
        { 3, 6 }, { 4, 6 }
    });
    // Graph from page 2 of http://hbanaszak.mjr.uw.edu.pl/TempTxt/Batagelj/cores.pdf
    graph_3 = buildGraph(21, {
        { 0, 5 }, { 0, 7 }, { 0, 9 }, { 0, 12 }, { 1, 3 }, { 1, 6 }, { 2, 8 }, { 2, 10 },
        { 2, 16 }, { 3, 6 }, { 3, 7 }, { 3, 13 }, { 3, 15 }, { 3, 20 }, { 4, 17 }, { 5, 7 },
        { 5, 12 }, { 6, 7 }, { 6, 11 }, { 7, 12 }, { 8, 19 }, { 9, 15 }, { 9, 17 }, { 13, 15 },
        { 13, 20 }, { 14, 17 }, { 15, 17 }, { 15, 20 }, { 16, 19 }
    });
  }

  // Build a BGL graph from a vector of edges.
  Graph buildGraph(const size_t num_vertices,
                   const std::vector<std::pair<size_t, size_t>>& edges) {
    Graph graph(num_vertices);
    for (const auto& edge : edges) {
      boost::add_edge(edge.first, edge.second, graph);
    }
    return graph;
  }
};

TEST_F(GraphUtilitiesTest, test_find_max_clique_1) {
  // Arrange
  Vertices expected_max_clique = { 1, 2, 3 };

  // Act
  Vertices max_clique = GraphUtilities::findMaximumClique(graph_1, 3);

  // Assert
  std::sort(max_clique.begin(), max_clique.end());
  EXPECT_EQ(expected_max_clique, max_clique);
}

TEST_F(GraphUtilitiesTest, test_find_max_clique_2) {
  // Arrange
  Vertices expected_max_clique = { 2, 3, 4, 6 };

  // Act
  Vertices max_clique = GraphUtilities::findMaximumClique(graph_2, 3);

  // Assert
  std::sort(max_clique.begin(), max_clique.end());
  EXPECT_EQ(expected_max_clique, max_clique);
}

TEST_F(GraphUtilitiesTest, test_find_max_clique_3) {
  // Arrange
  Vertices expected_max_clique = { 3, 13, 15, 20 };
  // The graph contains two maximum cliques. Which one is found depends on the algorithm.
  // Vertices expected_max_clique = { 0, 5, 7, 12 };

  // Act
  Vertices max_clique = GraphUtilities::findMaximumClique(graph_3, 4);

  // Assert
  std::sort(max_clique.begin(), max_clique.end());
  EXPECT_EQ(expected_max_clique, max_clique);
}

TEST_F(GraphUtilitiesTest, test_find_max_clique_no_such_clique) {
  // Arrange
  Vertices expected_max_clique = { };

  // Act
  Vertices max_clique = GraphUtilities::findMaximumClique(graph_3, 5);

  // Assert
  std::sort(max_clique.begin(), max_clique.end());
  EXPECT_EQ(expected_max_clique, max_clique);
}
