#ifndef _DTRAIN_H_
#define _DTRAIN_H_

#define DTRAIN_DOTS 10 // after how many inputs to display a '.'
#define DTRAIN_SCALE 100000

#include <iomanip>
#include <climits>
#include <string.h>

#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/program_options.hpp>

#include "decoder.h"
#include "ff_register.h"
#include "sentence_metadata.h"
#include "verbose.h"
#include "viterbi.h"

using namespace std;
namespace po = boost::program_options;

namespace dtrain
{

typedef double score_t;

struct ScoredHyp
{
  vector<WordID> w;
  SparseVector<weight_t> f;
  score_t model, score;
  unsigned rank;
};

inline void
RegisterAndConvert(const vector<string>& strs, vector<WordID>& ids)
{
  vector<string>::const_iterator it;
  for (auto s: strs)
    ids.push_back(TD::Convert(s));
}

inline void
PrintWordIDVec(vector<WordID>& v, ostream& os=cerr)
{
  for (unsigned i = 0; i < v.size(); i++) {
    os << TD::Convert(v[i]);
    if (i < v.size()-1) os << " ";
  }
}

inline ostream& _np(ostream& out) { return out << resetiosflags(ios::showpos); }
inline ostream& _p(ostream& out)  { return out << setiosflags(ios::showpos); }
inline ostream& _p2(ostream& out) { return out << setprecision(2); }
inline ostream& _p5(ostream& out) { return out << setprecision(5); }

template<typename T>
inline T
sign(T z)
{
  if (z == 0) return 0;
  return z < 0 ? -1 : +1;
}

} // namespace

#endif

