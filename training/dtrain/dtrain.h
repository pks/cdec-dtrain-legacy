#ifndef _DTRAIN_H_
#define _DTRAIN_H_

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

struct ScoredHyp
{
  vector<WordID>         w;
  SparseVector<weight_t> f;
  weight_t               model, gold;
  size_t               rank;
};

inline void
RegisterAndConvert(const vector<string>& strs, vector<WordID>& ids)
{
  for (auto s: strs)
    ids.push_back(TD::Convert(s));
}

inline void
PrintWordIDVec(vector<WordID>& v, ostream& os=cerr)
{
  for (size_t i = 0; i < v.size(); i++) {
    os << TD::Convert(v[i]);
    if (i < v.size()-1) os << " ";
  }
}

inline ostream& _np(ostream& out) { return out << resetiosflags(ios::showpos); }
inline ostream& _p(ostream& out)  { return out << setiosflags(ios::showpos); }
inline ostream& _p2(ostream& out) { return out << setprecision(2); }
inline ostream& _p5(ostream& out) { return out << setprecision(5); }

bool
dtrain_init(int argc, char** argv, po::variables_map* conf)
{
  po::options_description ini("Configuration File Options");
  ini.add_options()
    ("bitext,b",          po::value<string>(),                                                   "bitext")
    ("decoder_config,C",  po::value<string>(),                           "configuration file for decoder")
    ("iterations,T",      po::value<size_t>()->default_value(10),    "number of iterations T (per shard)")
    ("k",                 po::value<size_t>()->default_value(100),                   "size of kbest list")
    ("learning_rate,l",   po::value<weight_t>()->default_value(1.0),                      "learning rate")
    ("l1_reg,r",          po::value<weight_t>()->default_value(0.),          "l1 regularization strength")
    ("error_margin,m",    po::value<weight_t>()->default_value(0.),        "margin for margin perceptron")
    ("N",                 po::value<size_t>()->default_value(4),               "N for BLEU approximation")
    ("input_weights,w",   po::value<string>(),                                       "input weights file")
    ("average,a",         po::value<bool>()->default_value(false),               "output average weights")
    ("keep,K",            po::value<bool>()->default_value(false),   "output a weight file per iteration")
    ("output,o",          po::value<string>()->default_value("-"),  "output weights file, '-' for STDOUT")
    ("print_weights,P",   po::value<string>()->default_value("EgivenFCoherent SampleCountF CountEF MaxLexFgivenE MaxLexEgivenF IsSingletonF IsSingletonFE Glue WordPenalty PassThrough LanguageModel LanguageModel_OOV"),
                                                          "list of weights to print after each iteration");
  po::options_description cl("Command Line Options");
  cl.add_options()
    ("config,c", po::value<string>(), "dtrain config file");
  cl.add(ini);
  po::store(parse_command_line(argc, argv, cl), *conf);
  if (conf->count("config")) {
    ifstream f((*conf)["config"].as<string>().c_str());
    po::store(po::parse_config_file(f, ini), *conf);
  }
  po::notify(*conf);
  if (!conf->count("decoder_config")) {
    cerr << "Missing decoder configuration." << endl;
    return false;
  }
  if (!conf->count("bitext")) {
    cerr << "No training data given." << endl;
    return false;
  }

  return true;
}

} // namespace

#endif

