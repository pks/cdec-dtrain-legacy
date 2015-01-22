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
int main(int argc, char** argv)
{
  ReadFile rf(argv[1]);
  Hypergraph hg;
  HypergraphIO::ReadFromBinary(rf.stream(), &hg);
  SparseVector<double> v;
  ifstream f(argv[2]);
  string line;
  while (getline(f, line)) {
    istringstream ss(line);
    string k; weight_t w;
    ss >> k >> w;
    v.add_value(FD::Convert(k), w);
  }
  hg.Reweight(v);
  clock_t begin = clock();
  hg.TopologicallySortNodesAndEdges(hg.NumberOfNodes()-1);
  vector<WordID> trans;
  ViterbiESentence(hg, &trans);
  cout << TD::GetString(trans) << endl << flush;
  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << elapsed_secs << " s" << endl;

  return 0;
}

