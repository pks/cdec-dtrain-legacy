#include <iostream>
#include <cmath>
#include <utility>
#ifndef HAVE_OLD_CPP
# include <unordered_map>
#else
# include <tr1/unordered_map>
namespace std { using std::tr1::unordered_map; }
#endif

#include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

#include "m.h"
#include "corpus_tools.h"
#include "stringlib.h"
#include "filelib.h"
#include "ttables.h"
#include "tdict.h"
#include "da.h"

#include <nanomsg/nn.h>
#include <nanomsg/pair.h>
#include "nn.hpp"

namespace po = boost::program_options;
using namespace std;

bool InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
    ("diagonal_tension,T", po::value<double>()->default_value(4.0), "How sharp or flat around the diagonal is the alignment distribution (<1 = flat >1 = sharp)")
		("mean_srclen_multiplier,m",po::value<double>()->default_value(1), "When --force_align, use this source length multiplier")
		("force_align,f",po::value<string>(), "Load previously written parameters to 'force align' input. Set --diagonal_tension and --mean_srclen_multiplier as estimated during training.")
    ("favor_diagonal,d", "Use a static alignment distribution that assigns higher probabilities to alignments near the diagonal")
    ("prob_align_null", po::value<double>()->default_value(0.08), "When --favor_diagonal is set, what's the probability of a null alignment?")
    ("no_null_word,N","Do not generate from a null token")
    ("sock_url", po::value<string>()->default_value("tcp://127.0.0.1:60666"), "Socket url.");
  po::options_description clo("Command line options");
  clo.add_options()
        ("help,h", "Print this help message and exit");
  po::options_description dconfig_options, dcmdline_options;
  dconfig_options.add(opts);
  dcmdline_options.add(opts).add(clo);

  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  po::notify(*conf);

  if (conf->count("help") || conf->count("force_align")==0) {
    cerr << "Usage " << argv[0] << " [OPTIONS] -f params\n";
    cerr << dcmdline_options << endl;
    return false;
  }

  return true;
}

int main(int argc, char** argv) {
  po::variables_map conf;
  if (!InitCommandLine(argc, argv, &conf)) return 1;
  const double diagonal_tension = conf["diagonal_tension"].as<double>();
	const double mean_srclen_multiplier = conf["mean_srclen_multiplier"].as<double>();
  const bool use_null = (conf.count("no_null_word") == 0);
  const bool favor_diagonal = conf.count("favor_diagonal");
  const double prob_align_null = conf["prob_align_null"].as<double>();
  const double prob_align_not_null = 1.0 - prob_align_null;
  const WordID kNULL = TD::Convert("<eps>");
	ReadFile s2t_f(conf["force_align"].as<string>());
  TTable s2t;
	s2t.DeserializeLogProbsFromText(s2t_f.stream());

  nn::socket sock(AF_SP, NN_PAIR);
  string url = conf["sock_url"].as<string>();
  sock.bind(url.c_str());
  int to = 100;
  sock.setsockopt(NN_SOL_SOCKET, NN_RCVTIMEO, &to, sizeof (to));

  while (true)
  {
    char *buf = NULL;
    size_t sz = sock.recv(&buf, NN_MSG, 0);
    if (!buf)
      continue;
    string line(buf, buf+sz);
    if (line == "shutdown") {
      cerr << "shutting down" << endl;
      break;
    }
    cerr << "got '" << line << "'" << endl;
    nn::freemsg(buf);
    vector<WordID> src, trg;
    CorpusTools::ReadLine(line, &src, &trg);
    double log_prob = Md::log_poisson(trg.size(), 0.05 + src.size() * mean_srclen_multiplier);

    // compute likelihood
    ostringstream ss;
    for (unsigned j = 0; j < trg.size(); ++j) {
      const WordID& f_j = trg[j];
      double sum = 0;
      int a_j = 0;
      double max_pat = 0;
      double prob_a_i = 1.0 / (src.size() + use_null);  // uniform (model 1)
      if (use_null) {
        if (favor_diagonal) prob_a_i = prob_align_null;
        max_pat = s2t.prob(kNULL, f_j) * prob_a_i;
        sum += max_pat;
      }
      double az = 0;
      if (favor_diagonal)
        az = DiagonalAlignment::ComputeZ(j+1, trg.size(), src.size(), diagonal_tension) / prob_align_not_null;
      for (unsigned i = 1; i <= src.size(); ++i) {
        if (favor_diagonal)
          prob_a_i = DiagonalAlignment::UnnormalizedProb(j + 1, i, trg.size(), src.size(), diagonal_tension) / az;
        double pat = s2t.prob(src[i-1], f_j) * prob_a_i;
        if (pat > max_pat) { max_pat = pat; a_j = i; }
        sum += pat;
      }
      log_prob += log(sum);
      if (a_j > 0)
        ss << ' ' << (a_j - 1) << '-' << j;
    }
    string a = ss.str();
    cerr << "sending '" << a << "'" << endl;
    sock.send(a.c_str(), a.size()+1, 0);
  } // loop

  return 0;
}

