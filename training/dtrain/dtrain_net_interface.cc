#include "dtrain_net_interface.h"
#include "sample_net_interface.h"
#include "score_net_interface.h"
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
  weight_t eta                = conf["learning_rate"].as<weight_t>();
  weight_t eta_sparse         = conf["learning_rate_sparse"].as<weight_t>();
  const weight_t margin       = conf["margin"].as<weight_t>();
  const string master_addr    = conf["addr"].as<string>();
  const string output_fn      = conf["output"].as<string>();
  const string debug_fn       = conf["debug_output"].as<string>();
  vector<string> dense_features;
  boost::split(dense_features, conf["dense_features"].as<string>(),
               boost::is_any_of(" "));

  // setup decoder
  register_feature_functions();
  SetSilent(true);
  ReadFile f(conf["decoder_conf"].as<string>());
  Decoder decoder(f.stream());
  ScoredKbest* observer = new ScoredKbest(k, new PerSentenceBleuScorer(N));

  // weights
  vector<weight_t>& decoder_weights = decoder.CurrentWeightVector();
  SparseVector<weight_t> lambdas, w_average, original_lambdas;
  if (conf.count("input_weights")) {
    Weights::InitFromFile(conf["input_weights"].as<string>(), &decoder_weights);
    Weights::InitSparseVector(decoder_weights, &lambdas);
    Weights::InitSparseVector(decoder_weights, &original_lambdas);
  }

  cerr << _p4;
  // output configuration
  cerr << "dtrain_net_interface" << endl << "Parameters:" << endl;
  cerr << setw(25) << "k " << k << endl;
  cerr << setw(25) << "N " << N << endl;
  cerr << setw(25) << "eta " << eta << endl;
  cerr << setw(25) << "eta (sparse) " << eta_sparse << endl;
  cerr << setw(25) << "margin " << margin << endl;
  cerr << setw(25) << "decoder conf " << "'"
       << conf["decoder_conf"].as<string>() << "'" << endl;
  cerr << setw(25) << "output " << output_fn << endl;

  // setup socket
  nn::socket sock(AF_SP, NN_PAIR);
  sock.bind(master_addr.c_str());
  string hello = "hello";
  sock.send(hello.c_str(), hello.size()+1, 0);

  // debug
  ostringstream debug_output;

  string done = "done";

  size_t i = 0;
  while(true)
  {
    // debug --
    debug_output.str(string());
    debug_output.clear();
    debug_output << "{" << endl; // hack us a nice JSON output
    // -- debug

    char *buf = NULL;
    string source;
    vector<Ngrams> refs;
    vector<size_t> rsz;
    bool next = true;
    size_t sz = sock.recv(&buf, NN_MSG, 0);
    if (buf) {
      const string in(buf, buf+sz);
      nn::freemsg(buf);
      cerr << "[dtrain] got input '" << in << "'" << endl;
      if        (boost::starts_with(in, "set_learning_rate")) { // set learning rate
        stringstream ss(in);
        string x; weight_t w;
        ss >> x; ss >> w;
        cerr << "[dtrain] setting (dense) learning rate to " << w << " (was: " << eta << ")" << endl;
        eta = w;
        cerr << "[dtrain] done, looping again" << endl;
        sock.send(done.c_str(), done.size()+1, 0);
        continue;
      } else if (boost::starts_with(in, "set_sparse_learning_rate")) { // set sparse learning rate
        stringstream ss(in);
        string x; weight_t w;
        ss >> x; ss >> w;
        cerr << "[dtrain] setting sparse learning rate to " << w << " (was: " << eta_sparse << ")" << endl;
        eta_sparse = w;
        cerr << "[dtrain] done, looping again" << endl;
        sock.send(done.c_str(), done.size()+1, 0);
        continue;
      } else if (boost::starts_with(in, "reset_weights")) { // reset weights
        cerr << "[dtrain] resetting weights" << endl;
        lambdas = original_lambdas;
        cerr << "[dtrain] done, looping again" << endl;
        sock.send(done.c_str(), done.size()+1, 0);
        continue;
      } else if (in == "shutdown") { // shut down
        cerr << "[dtrain] got shutdown signal" << endl;
        next = false;
      } else { // translate
        vector<string> parts;
        boost::algorithm::split_regex(parts, in, boost::regex(" \\|\\|\\| "));
        if (parts[0] == "act:translate") {
          cerr << "[dtrain] translating ..." << endl;
          lambdas.init_vector(&decoder_weights);
          observer->dont_score = true;
          decoder.Decode(parts[1], observer);
          observer->dont_score = false;
          vector<ScoredHyp>* samples = observer->GetSamples();
          ostringstream os;
          cerr << "[dtrain] 1best features " << (*samples)[0].f << endl;
          PrintWordIDVec((*samples)[0].w, os);
          sock.send(os.str().c_str(), os.str().size()+1, 0);
          cerr << "[dtrain] done translating, looping again" << endl;
          continue;
        } else { // learn
          cerr << "[dtrain] learning ..." << endl;
          source = parts[0];
          // debug --
          debug_output << "\"source\":\"" <<  source.substr(source.find_first_of(">")+1, source.find_last_of("<")-3) <<  "\"," << endl;
          debug_output << "\"target\":\"" << parts[1] <<  "\"," << endl;
          // -- debug
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

    // debug --
    debug_output << "\"1best\":\"";
    PrintWordIDVec((*samples)[0].w, debug_output);
    debug_output << "\"," << endl;
    debug_output << "\"kbest\":[" << endl;
    size_t h = 0;
    for (auto s: *samples) {
      debug_output << "\"" << s.gold << " ||| " << s.model << " ||| " << s.rank << " ||| ";
      for (auto o: s.f)
        debug_output << FD::Convert(o.first) << "=" << o.second << " ";
      debug_output << " ||| ";
      PrintWordIDVec(s.w, debug_output);
      h += 1;
      debug_output << "\"";
      if (h < samples->size()) {
        debug_output << ",";
      }
      debug_output << endl;
    }
    debug_output << "]," << endl;
    debug_output << "\"samples_size\":" << samples->size() << "," << endl;
    debug_output << "\"weights_before\":{" << endl;
    weightsToJson(lambdas, debug_output);
    debug_output << "}," << endl;
    // -- debug

    // get pairs and update
    SparseVector<weight_t> updates;
    size_t num_up = CollectUpdates(samples, updates, margin);
    updates *= eta_sparse; // apply learning rate for sparse features
    for (auto feat: dense_features) { // apply learning rate for dense features
      updates[FD::Convert(feat)] /= eta_sparse;
      updates[FD::Convert(feat)] *= eta;
    }
    // debug --
    debug_output << "\"num_up\":" << num_up << "," << endl;
    debug_output << "\"updated_features\":" << updates.size() << "," << endl;
    debug_output << "\"learning_rate\":" << eta << "," << endl;
    debug_output << "\"learning_rate_sparse\":" << eta_sparse << "," << endl;
    debug_output << "\"best_match\":\"";
    PrintWordIDVec((*samples)[0].w, debug_output);
    debug_output << "\"," << endl;
    debug_output << "\"best_match_score\":" << (*samples)[0].gold << "," << endl ;
    // -- debug
    lambdas.plus_eq_v_times_s(updates, 1.0);
    i++;

    // debug --
    debug_output << "\"weights_after\":{" << endl;
    weightsToJson(lambdas, debug_output);
    debug_output << "}" << endl;
    debug_output << "}" << endl;
    // -- debug

    cerr << "[dtrain] done learning, looping again" << endl;
    sock.send(done.c_str(), done.size()+1, 0);

    // debug --
    WriteFile f(debug_fn);
    *f << debug_output.str();
    // -- debug

    // write current weights
    lambdas.init_vector(decoder_weights);
    ostringstream fn;
    fn << output_fn << "." << i << ".gz";
    Weights::WriteToFile(fn.str(), decoder_weights, true);
  } // input loop

  string shutdown = "off";
  sock.send(shutdown.c_str(), shutdown.size()+1, 0);

  cerr << "[dtrain] shutting down, goodbye" << endl;

  return 0;
}

