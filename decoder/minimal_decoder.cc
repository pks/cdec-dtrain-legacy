#include <fstream>
#include <iostream>
#include <sstream>

#include "fdict.h"
#include "filelib.h"
#include "hg.h"
#include "hg_io.h"
#include "sparse_vector.h"
#include "viterbi.h"

using namespace std;

/*
 * Reads hypergraph from JSON file argv[1],
 * reweights it using weights from argv[2],
 * and outputs viterbi translation.
 *
 */
int
main(int argc, char** argv)
{
  clock_t begin_total = clock();

  // read hg
  clock_t begin_read = clock();
  ReadFile rf(argv[1]);
  Hypergraph hg;
  HypergraphIO::ReadFromJSON(rf.stream(), &hg);
  clock_t end_read = clock();
  double elapsed_secs_read = double(end_read - begin_read) / CLOCKS_PER_SEC;
  cerr << "read hg " << elapsed_secs_read << " s" << endl;

  // read weights
  clock_t begin_weights = clock();
  SparseVector<double> v;
  ifstream f(argv[2]);
  string line;
  while (getline(f, line)) {
    istringstream ss(line);
    string k; weight_t w;
    ss >> k >> w;
    v.add_value(FD::Convert(k), w);
  }
  clock_t end_weights = clock();
  double elapsed_secs_weights = double(end_weights - begin_weights) / CLOCKS_PER_SEC;
  cerr << "read weights " << elapsed_secs_weights << " s" << endl;

  // reweight hg
  clock_t begin_reweight = clock();
  hg.Reweight(v);
  clock_t end_reweight = clock();
  double elapsed_secs_reweight = double(end_reweight - begin_reweight) / CLOCKS_PER_SEC;
  cerr << "reweight " << elapsed_secs_reweight << " s" << endl;

  // topsort
  clock_t begin_top = clock();
  hg.TopologicallySortNodesAndEdges(hg.NumberOfNodes()-1);
  clock_t end_top = clock();
  double elapsed_secs_top = double(end_top - begin_top) / CLOCKS_PER_SEC;
  cerr << "topsort " << elapsed_secs_top << " s" << endl;

  // viterbi
  clock_t begin_viterbi = clock();
  vector<WordID> trans;
  ViterbiESentence(hg, &trans);
  cout << TD::GetString(trans) << endl << flush;
  clock_t end_viterbi = clock();
  double elapsed_secs_viterbi = double(end_viterbi - begin_viterbi) / CLOCKS_PER_SEC;
  cerr << "viterbi " << elapsed_secs_viterbi << " s" << endl;

  // total
  clock_t end_total = clock();
  double elapsed_secs = double(end_total - begin_total) / CLOCKS_PER_SEC;
  cerr << "total " << elapsed_secs << " s" << endl;
  
  return 0;
}

