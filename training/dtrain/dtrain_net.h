#ifndef _DTRAIN_NET_H_
#define _DTRAIN_NET_H_

#include "dtrain.h"

namespace dtrain
{

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
  istringstream ss(s);
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
    ("decoder_conf,C", po::value<string>(),                      "configuration file for decoder")
    ("k",              po::value<size_t>()->default_value(100),              "size of kbest list")
    ("N",              po::value<size_t>()->default_value(4),          "N for BLEU approximation")
    ("margin,m",       po::value<weight_t>()->default_value(0.),   "margin for margin perceptron")
    ("output,o",       po::value<string>()->default_value(""),               "final weights file")
    ("input_weights,w",   po::value<string>(),                               "input weights file");
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

