#include "dtrain_net.h"
#include "sample_net.h"
#include "score.h"
#include "update.h"

#include <nanomsg/nn.h>
#include <nanomsg/pair.h>
#include "nn.hpp"

using namespace dtrain;

int
main(int argc, char** argv)
{
  // get configuration
  po::variables_map conf;
  if (!dtrain_net_init(argc, argv, &conf))
    exit(1); // something is wrong
  const size_t k              = conf["k"].as<size_t>();
  const size_t N              = conf["N"].as<size_t>();
  const weight_t margin       = conf["margin"].as<weight_t>();
  const string master_addr    = conf["addr"].as<string>();

  // setup decoder
  register_feature_functions();
  SetSilent(true);
  ReadFile f(conf["decoder_conf"].as<string>());
  Decoder decoder(f.stream());
  ScoredKbest* observer = new ScoredKbest(k, new PerSentenceBleuScorer(N));

  // weights
  vector<weight_t>& decoder_weights = decoder.CurrentWeightVector();
  SparseVector<weight_t> lambdas, w_average;
  if (conf.count("input_weights")) {
    Weights::InitFromFile(conf["input_weights"].as<string>(), &decoder_weights);
    Weights::InitSparseVector(decoder_weights, &lambdas);
  }

  cerr << _p4;
  // output configuration
  cerr << "dtrain_net" << endl << "Parameters:" << endl;
  cerr << setw(25) << "k " << k << endl;
  cerr << setw(25) << "N " << N << endl;
  cerr << setw(25) << "margin " << margin << endl;
  cerr << setw(25) << "decoder conf " << "'"
       << conf["decoder_conf"].as<string>() << "'" << endl;

  // socket
  nn::socket sock(AF_SP, NN_PAIR);
  sock.connect(master_addr.c_str());

  size_t i = 0;
  while(true)
  {
    char *buf = NULL;
    string source;
    vector<Ngrams> refs;
    vector<size_t> rsz;
    bool next = true;
    size_t sz = sock.recv(&buf, NN_MSG, 0);
    if (buf) {
      const string in(buf, buf+sz);
      nn::freemsg(buf);
      if (in == "shutdown") {
        next = false;
      } else {
        vector<string> parts;
        boost::algorithm::split_regex(parts, in, boost::regex(" \\|\\|\\| "));
        if (parts[0] == "act:translate") {
          cerr << "translating ..." << endl;
          lambdas.init_vector(&decoder_weights);
          observer->dont_score = true;
          decoder.Decode(parts[1], observer);
          observer->dont_score = false;
          vector<ScoredHyp>* samples = observer->GetSamples();
          ostringstream os;
          PrintWordIDVec((*samples)[0].w, os);
          sock.send(os.str().c_str(), os.str().size()+1, 0);
          cerr << "done" << endl;
          continue;
        } else {
          cerr << "learning ..." << endl;
          source = parts[0];
          parts.erase(parts.begin());
          for (auto s: parts) {
            vector<WordID> r;
            vector<string> toks;
            boost::split(toks, s, boost::is_any_of(" "));
            for (auto tok: toks)
              r.push_back(TD::Convert(tok));
            refs.emplace_back(MakeNgrams(r, N));
            rsz.push_back(r.size());
          }
        }
      }
    }
    
    if (!next)
      break;

    // decode
    lambdas.init_vector(&decoder_weights);
    observer->SetReference(refs, rsz);
    decoder.Decode(source, observer);
    vector<ScoredHyp>* samples = observer->GetSamples();

    // get pairs and update
    SparseVector<weight_t> updates;
    CollectUpdates(samples, updates, margin);
    lambdas.plus_eq_v_times_s(updates, 1.0); // fixme
    string s = "x";
    sock.send(s.c_str(), s.size()+1, 0);
    i++;

    cerr << "done" << endl;
  } // input loop

  return 0;
}

