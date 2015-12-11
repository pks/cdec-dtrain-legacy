#ifndef _DTRAIN_NET_INTERFACE_H_
#define _DTRAIN_NET_INTERFACE_H_

#include "dtrain.h"

namespace dtrain
{

inline void
weightsToJson(SparseVector<weight_t>& w, ostringstream& os)
{
  vector<string> strs;
  for (typename SparseVector<weight_t>::iterator it=w.begin(),e=w.end(); it!=e; ++it) {
    ostringstream a;
    a << "\"" << FD::Convert(it->first) << "\":" << it->second;
    strs.push_back(a.str());
  }
  for (vector<string>::const_iterator it=strs.begin(); it!=strs.end(); it++) {
    os << *it;
    if ((it+1) != strs.end())
      os << ",";
    os << endl;
  }
}

template<typename T>
inline void
vectorAsString(SparseVector<T>& v, ostringstream& os)
{
  SparseVector<weight_t>::iterator it = v.begin();
  for (; it != v.end(); ++it) {
    os << FD::Convert(it->first) << "=" << it->second;
    auto peek = it;
    if (++peek != v.end())
      os << " ";
  }
}

template<typename T>
inline void
updateVectorFromString(string& s, SparseVector<T>& v)
{
  string buf;
  istringstream ss;
  while (ss >> buf) {
    size_t p = buf.find_last_of("=");
    istringstream c(buf.substr(p+1,buf.size()));
    weight_t val;
    c >> val;
    v[FD::Convert(buf.substr(0,p))] = val;
  }
}

bool
dtrain_net_init(int argc, char** argv, po::variables_map* conf)
{
  po::options_description ini("Configuration File Options");
  ini.add_options()
    ("decoder_conf,C",         po::value<string>(),                          "configuration file for decoder")
    ("k",                      po::value<size_t>()->default_value(100),                  "size of kbest list")
    ("N",                      po::value<size_t>()->default_value(4),              "N for BLEU approximation")
    ("margin,m",               po::value<weight_t>()->default_value(0.),       "margin for margin perceptron")
    ("output,o",               po::value<string>()->default_value(""),                   "final weights file")
    ("input_weights,w",        po::value<string>(),                                      "input weights file")
    ("learning_rate,l",        po::value<weight_t>()->default_value(0.001),                   "learning rate")
    ("learning_rate_sparse,l", po::value<weight_t>()->default_value(0.00001), "learning rate for sparse features")
    ("output_derivation,E",    po::bool_switch()->default_value(false),  "output derivation, not viterbi str")
    ("output_rules,R",    po::bool_switch()->default_value(false),                        "also output rules")
    ("dense_features,D",       po::value<string>()->default_value("EgivenFCoherent SampleCountF CountEF MaxLexFgivenE MaxLexEgivenF IsSingletonF IsSingletonFE Glue WordPenalty PassThrough LanguageModel LanguageModel_OOV Shape_S01111_T11011 Shape_S11110_T11011 Shape_S11100_T11000 Shape_S01110_T01110 Shape_S01111_T01111 Shape_S01100_T11000 Shape_S10000_T10000 Shape_S11100_T11100 Shape_S11110_T11110 Shape_S11110_T11010 Shape_S01100_T11100 Shape_S01000_T01000 Shape_S01010_T01010 Shape_S01111_T01011 Shape_S01100_T01100 Shape_S01110_T11010 Shape_S11000_T11000 Shape_S11000_T01100 IsSupportedOnline ForceRule"),
                                                                                             "dense features")
    ("debug_output,d",   po::value<string>()->default_value(""),                      "file for debug output");
  po::options_description cl("Command Line Options");
  cl.add_options()
    ("conf,c", po::value<string>(), "dtrain configuration file")
    ("addr,a", po::value<string>(),         "address of master");
  cl.add(ini);
  po::store(parse_command_line(argc, argv, cl), *conf);
  if (conf->count("conf")) {
    ifstream f((*conf)["conf"].as<string>().c_str());
    po::store(po::parse_config_file(f, ini), *conf);
  }
  po::notify(*conf);
  if (!conf->count("decoder_conf")) {
    cerr << "Missing decoder configuration. Exiting." << endl;
    return false;
  }
  if (!conf->count("addr")) {
    cerr << "No master address given! Exiting." << endl;
    return false;
  }

  return true;
}

} // namespace

#endif

