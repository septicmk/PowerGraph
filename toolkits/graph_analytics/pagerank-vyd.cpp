/*
 * Copyright (c) 2009 Carnegie Mellon University.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://www.graphlab.ml.cmu.edu
 *
 */

#include <vector>
#include <string>
#include <fstream>

#include <graphlab.hpp>
// #include <graphlab/macros_def.hpp>

#include "vineyard/client/client.h"
#include "vineyard/graph/fragment/arrow_fragment.h"
#include "vineyard/graph/fragment/graph_schema.h"
#include "vineyard/graph/loader/arrow_fragment_loader.h"
#include <graphlab/vyd/arrow_projected_fragment.h>

using namespace gs;
// Global random reset probability
double RESET_PROB = 0.15;

double TOLERANCE = 1.0E-2;

size_t ITERATIONS = 0;

bool USE_DELTA_CACHE = false;

// The vertex data is just the pagerank value (a double)
typedef double vertex_data_type;

// There is no edge data in the pagerank application
typedef graphlab::empty edge_data_type;

// The graph type is determined by the vertex and edge data types
typedef graphlab::distributed_graph<vertex_data_type, edge_data_type> graph_type;

/*
 * A simple function used by graph.transform_vertices(init_vertex);
 * to initialize the vertes data.
 */
void init_vertex(graph_type::vertex_type& vertex) { vertex.data() = 1; }



/*
 * The factorized page rank update function extends ivertex_program
 * specifying the:
 *
 *   1) graph_type
 *   2) gather_type: double (returned by the gather function). Note
 *      that the gather type is not strictly needed here since it is
 *      assumed to be the same as the vertex_data_type unless
 *      otherwise specified
 *
 * In addition ivertex program also takes a message type which is
 * assumed to be empty. Since we do not need messages no message type
 * is provided.
 *
 * pagerank also extends graphlab::IS_POD_TYPE (is plain old data type)
 * which tells graphlab that the pagerank program can be serialized
 * (converted to a byte stream) by directly reading its in memory
 * representation.  If a vertex program does not exted
 * graphlab::IS_POD_TYPE it must implement load and save functions.
 */
class pagerank :
  public graphlab::ivertex_program<graph_type, double> {

  double last_change;
public:

  /**
   * Gather only in edges.
   */
  edge_dir_type gather_edges(icontext_type& context,
                              const vertex_type& vertex) const {
    return graphlab::IN_EDGES;
  } // end of Gather edges


  /* Gather the weighted rank of the adjacent page   */
  double gather(icontext_type& context, const vertex_type& vertex,
               edge_type& edge) const {
    return (edge.source().data() / edge.source().num_out_edges());
  }

  /* Use the total rank of adjacent pages to update this page */
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& total) {

    const double newval = (1.0 - RESET_PROB) * total + RESET_PROB;
    last_change = (newval - vertex.data());
    vertex.data() = newval;
    if (ITERATIONS) context.signal(vertex);
  }

  /* The scatter edges depend on whether the pagerank has converged */
  edge_dir_type scatter_edges(icontext_type& context,
                              const vertex_type& vertex) const {
    // If an iteration counter is set then
    if (ITERATIONS) return graphlab::NO_EDGES;
    // In the dynamic case we run scatter on out edges if the we need
    // to maintain the delta cache or the tolerance is above bound.
    if(USE_DELTA_CACHE || std::fabs(last_change) > TOLERANCE ) {
      return graphlab::OUT_EDGES;
    } else {
      return graphlab::NO_EDGES;
    }
  }

  /* The scatter function just signal adjacent pages */
  void scatter(icontext_type& context, const vertex_type& vertex,
               edge_type& edge) const {
    if(USE_DELTA_CACHE) {
      context.post_delta(edge.target(), last_change);
    }

    if(last_change > TOLERANCE || last_change < -TOLERANCE) {
        context.signal(edge.target());
    } else {
      context.signal(edge.target()); //, std::fabs(last_change));
    }
  }

  void save(graphlab::oarchive& oarc) const {
    // If we are using iterations as a counter then we do not need to
    // move the last change in the vertex program along with the
    // vertex data.
    if (ITERATIONS == 0) oarc << last_change;
  }

  void load(graphlab::iarchive& iarc) {
    if (ITERATIONS == 0) iarc >> last_change;
  }

}; // end of factorized_pagerank update functor


/*
 * We want to save the final graph so we define a write which will be
 * used in graph.save("path/prefix", pagerank_writer()) to save the graph.
 */
struct pagerank_writer {
  std::string save_vertex(graph_type::vertex_type v) {
    std::stringstream strm;
    strm << v.id() << "\t" << v.data() << "\n";
    return strm.str();
  }
  std::string save_edge(graph_type::edge_type e) { return ""; }
}; // end of pagerank writer


double map_rank(const graph_type::vertex_type& v) { return v.data(); }


double pagerank_sum(graph_type::vertex_type v) {
  return v.data();
}

int main(int argc, char** argv) {
  // Initialize control plain using mpi
  graphlab::mpi_tools::init(argc, argv);
  graphlab::distributed_control dc;
  global_logger().set_log_level(LOG_INFO);

  // Parse command line options -----------------------------------------------
  graphlab::command_line_options clopts("PageRank algorithm.");
  std::string graph_dir;
  std::string format = "adj";
  std::string exec_type = "synchronous";
  clopts.attach_option("graph", graph_dir,
                       "The graph file.  If none is provided "
                       "then a toy graph will be created");
  clopts.add_positional("graph");
  clopts.attach_option("engine", exec_type,
                       "The engine type synchronous or asynchronous");
  clopts.attach_option("tol", TOLERANCE,
                       "The permissible change at convergence.");
  clopts.attach_option("format", format,
                       "The graph file format");
  size_t powerlaw = 0;
  clopts.attach_option("powerlaw", powerlaw,
                       "Generate a synthetic powerlaw out-degree graph. ");
  clopts.attach_option("iterations", ITERATIONS,
                       "If set, will force the use of the synchronous engine"
                       "overriding any engine option set by the --engine parameter. "
                       "Runs complete (non-dynamic) PageRank for a fixed "
                       "number of iterations. Also overrides the iterations "
                       "option in the engine");
  clopts.attach_option("use_delta", USE_DELTA_CACHE,
                       "Use the delta cache to reduce time in gather.");
  std::string saveprefix;
  clopts.attach_option("saveprefix", saveprefix,
                       "If set, will save the resultant pagerank to a "
                       "sequence of files with prefix saveprefix");
  std::string ipc_socket;
  clopts.attach_option("ipc_socket", ipc_socket,
                       "ipc_socket for vineyard.");
  std::string object_id;
  clopts.attach_option("object_id", object_id,
                       "object_id in vineyard.");
  std::string app_type;
  clopts.attach_option("app_type", app_type,
                       "write/read to/from vineyard.");

  std::string efile;
  clopts.attach_option("efile", efile,
                       "edgefile for vineyard.");

  std::string vfile;
  clopts.attach_option("vfile", vfile,
                       "vertexfile for vineyard.");

  if(!clopts.parse(argc, argv)) {
    dc.cout() << "Error in parsing command line arguments." << std::endl;
    return EXIT_FAILURE;
  }

  // Enable gather caching in the engine
  clopts.get_engine_args().set_option("use_cache", USE_DELTA_CACHE);

  if (ITERATIONS) {
    // make sure this is the synchronous engine
    dc.cout() << "--iterations set. Forcing Synchronous engine, and running "
              << "for " << ITERATIONS << " iterations." << std::endl;
    clopts.get_engine_args().set_option("type", "synchronous");
    clopts.get_engine_args().set_option("max_iterations", ITERATIONS);
    clopts.get_engine_args().set_option("sched_allv", true);
  }

  using OID_T = int64_t;
  using VID_T = uint32_t;
  using VDATA_T = grape::EmptyType;
  using EDATA_T = grape::EmptyType;

  // Build the graph ----------------------------------------------------------
  graph_type graph(dc, clopts);
  if (app_type == "reader") { // read graph from vineyard
    using GraphType = ArrowProjectedFragment<OID_T, VID_T, VDATA_T, EDATA_T>; // no LoadStrategy

    vineyard::Client client;
    VINEYARD_CHECK_OK(client.Connect(ipc_socket));

    LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

    vineyard::ObjectID object_id_ = vineyard::ObjectIDFromString(object_id);
    LOG(INFO) << "Searching the object_id " << object_id << " in vineyard, found " << object_id_;

    std::shared_ptr<GraphType> fragment_ =
        std::dynamic_pointer_cast<GraphType>(client.GetObject(object_id_));
    LOG(INFO) << "get ArrowProjectedFragment nv= " << fragment_->GetTotalVerticesNum(); 
    LOG(INFO) << "get ArrowProjectedFragment ne= " << fragment_->GetEdgeNum(); 
    // transfer
    LOG(INFO) << " Pouring fragment into PowerGraph... "; 
    {// TODO: add edge to add edges
      auto iv = fragment_->InnerVertices();
      for(auto v: iv){
        auto oe = fragment_->GetOutgoingAdjList(v);
        for(auto& e : oe) {
          auto u = e.get_neighbor();
          graph.add_edge(static_cast<graphlab::vertex_id_type>(v.GetValue()), 
                         static_cast<graphlab::vertex_id_type>(u.GetValue()));
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  } 
  else if (powerlaw > 0) { // make a synthetic graph
    dc.cout() << "Loading synthetic Powerlaw graph." << std::endl;
    graph.load_synthetic_powerlaw(powerlaw, false, 2.1, 100000000);
  }
  else if (graph_dir.length() > 0) { // Load the graph from a file
    dc.cout() << "Loading graph in format: "<< format << std::endl;
    graph.load_format(graph_dir, format);
  }
  else if (app_type == "writer") { // allocated the graph in vineyard
    using VydGraphType = vineyard::ArrowFragment<OID_T, VID_T>;
    using GraphType = ArrowProjectedFragment<OID_T, VID_T, VDATA_T, EDATA_T>; // no LoadStrategy

    std::vector<std::string> efiles = {efile};
    std::vector<std::string> vfiles = {vfile};
    vineyard::Client client;
    VINEYARD_CHECK_OK(client.Connect(ipc_socket));
    LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

    vineyard::ObjectID fragment_id = vineyard::InvalidObjectID();
    MPI_Barrier(MPI_COMM_WORLD);

    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);
    auto loader =
      std::make_unique<vineyard::ArrowFragmentLoader<OID_T, VID_T>>(
           client, comm_spec, efiles, vfiles, false);

    fragment_id = boost::leaf::try_handle_all(
        [&loader]() { return loader->LoadFragment(); },
        [](const vineyard::GSError& e) {
          LOG(FATAL) << e.error_msg;
          return 0;
        },
        [](const boost::leaf::error_info& unmatched) {
          LOG(FATAL) << "Unmatched error " << unmatched;
          return 0;
        });

    MPI_Barrier(MPI_COMM_WORLD);

    std::shared_ptr<VydGraphType> fragment_ =
        std::dynamic_pointer_cast<VydGraphType>(client.GetObject(fragment_id));
    LOG(INFO) << "[frag-" << fragment_->fid() << "]: " << vineyard::ObjectIDToString(fragment_id);
    std::string z = "0";
    std::string o = "-1";
    auto project_frag = GraphType::Project(fragment_, z, o, z, o);
    LOG(INFO) << "projected check: nv=" << fragment_->GetTotalVerticesNum() << " " << fragment_->GetEdgeNum();;
    return 0;
  } else {
    dc.cout() << "graph or powerlaw option must be specified" << std::endl;
    clopts.print_description();
    return 0;
  }

  // must call finalize before querying the graph
  graph.finalize();
  dc.cout() << " #vertices: " << graph.num_vertices()
            << " #edges:" << graph.num_edges() << std::endl;

  

  // Initialize the vertex data
  graph.transform_vertices(init_vertex);

  // Running The Engine -------------------------------------------------------
  graphlab::omni_engine<pagerank> engine(dc, graph, exec_type, clopts);
  engine.signal_all();
  engine.start();
  const double runtime = engine.elapsed_seconds();
  dc.cout() << "Finished Running engine in " << runtime
            << " seconds." << std::endl;


  const double total_rank = graph.map_reduce_vertices<double>(map_rank);
  std::cout << "Total rank: " << total_rank << std::endl;

  // Save the final graph -----------------------------------------------------
  if (saveprefix != "") {
    graph.save(saveprefix, pagerank_writer(),
               false,    // do not gzip
               true,     // save vertices
               false);   // do not save edges
  }

  double totalpr = graph.map_reduce_vertices<double>(pagerank_sum);
  std::cout << "Totalpr = " << totalpr << "\n";

  // Tear-down communication layer and quit -----------------------------------
  graphlab::mpi_tools::finalize();
  return EXIT_SUCCESS;
} // End of main


// We render this entire program in the documentation


