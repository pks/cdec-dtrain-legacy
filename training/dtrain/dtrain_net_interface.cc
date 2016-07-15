#include "dtrain_net_interface.h"
#include "sample_net_interface.h"
#include "score_net_interface.h"
#include "update.h"

#include <nanomsg/nn.h>
#include <nanomsg/pair.h>
#include "nn.hpp"

#include <sys/types.h>  // mkfifo
#include <sys/stat.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>


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
  const string output_fn      = conf["output"].as<string>();
  const string debug_fn       = conf["debug_output"].as<string>();
  vector<string> dense_features;
  boost::split(dense_features, conf["dense_features"].as<string>(),
               boost::is_any_of(" "));
  const bool output_derivation = conf["output_derivation"].as<bool>();
  const bool output_rules      = conf["output_rules"].as<bool>();

  // update lm
  /*if (conf["update_lm_fn"].as<string>() != "")
    mkfifo(conf["update_lm_fn"].as<string>().c_str(), 0666);*/

  // setup socket
  nn::socket sock(AF_SP, NN_PAIR);
  sock.bind(master_addr.c_str());
  string hello = "hello";
  sock.send(hello.c_str(), hello.size()+1, 0);

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

  // learning rates
  SparseVector<weight_t> learning_rates, original_learning_rates;
  weight_t learning_rate_R, original_learning_rate_R;
  weight_t learning_rate_RB, original_learning_rate_RB;
  weight_t learning_rate_Shape, original_learning_rate_Shape;
  vector<weight_t> l;
  Weights::InitFromFile(conf["learning_rates"].as<string>(), &l);
  Weights::InitSparseVector(l, &learning_rates);
  original_learning_rates      = learning_rates;
  learning_rate_R              = conf["learning_rate_R"].as<weight_t>();
  original_learning_rate_R     = learning_rate_R;
  learning_rate_RB             = conf["learning_rate_RB"].as<weight_t>();
  original_learning_rate_RB    = learning_rate_RB;
  learning_rate_Shape          = conf["learning_rate_Shape"].as<weight_t>();
  original_learning_rate_Shape = learning_rate_Shape;

  cerr << _p4;
  // output configuration
  cerr << "dtrain_net_interface" << endl << "Parameters:" << endl;
  cerr << setw(25) << "k " << k << endl;
  cerr << setw(25) << "N " << N << endl;
  cerr << setw(25) << "margin " << margin << endl;
  cerr << setw(25) << "decoder conf " << "'"
       << conf["decoder_conf"].as<string>() << "'" << endl;
  cerr << setw(25) << "output " << "'" <<  output_fn << "'" << endl;
  cerr << setw(25) << "debug "  << "'" << debug_fn   << "'" << endl;
  cerr << setw(25) << "learning rates "      << "'"
       << conf["learning_rates"].as<string>() << "'" << endl;
  cerr << setw(25) << "learning rate R "     << learning_rate_R     << endl;
  cerr << setw(25) << "learning rate RB "    << learning_rate_RB    << endl;
  cerr << setw(25) << "learning rate Shape " << learning_rate_Shape << endl;

  // debug
  ostringstream debug_output;

  string done = "done";

  vector<ScoredHyp>* samples;

  size_t i = 0;
  while(true)
  {
    // debug --
    debug_output.str(string());
    debug_output.clear();
    debug_output << "{" << endl; // hack us a nice JSON output
    // -- debug

    bool just_translate = false;

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
      if (boost::starts_with(in, "set_learning_rates")) { // set learning rates
        stringstream ss(in);
        string _,name; weight_t w;
        ss >> _; ss >> name; ss >> w;
        weight_t before = 0;
        ostringstream long_name;
        if (name == "R") {
          before = learning_rate_R;
          learning_rate_R = w;
          long_name << "rule id feature group";
        } else if (name == "RB") {
          before = learning_rate_RB;
          learning_rate_RB = w;
          long_name << "rule bigram feature group";
        } else if (name == "Shape") {
          before = learning_rate_Shape;
          learning_rate_Shape = w;
          long_name << "rule shape feature group";
        } else {
          unsigned fid = FD::Convert(name);
          before = learning_rates[fid];
          learning_rates[fid] = w;
          long_name << "feature '" << name << "'";
        }
        ostringstream o;
        o << "set learning rate for " << long_name.str() << " to " << w
          << " (was: " << before << ")" << endl;
        string s = o.str();
        cerr << "[dtrain] " << s;
        cerr << "[dtrain] done, looping again" << endl;
        sock.send(s.c_str(), s.size()+1, 0);
        continue;
      } else if (boost::starts_with(in, "reset_learning_rates")) {
        cerr << "[dtrain] resetting learning rates" << endl;
        learning_rates = original_learning_rates;
        learning_rate_R = original_learning_rate_R;
        learning_rate_RB = original_learning_rate_RB;
        learning_rate_Shape = original_learning_rate_Shape;
        cerr << "[dtrain] done, looping again" << endl;
        sock.send(done.c_str(), done.size()+1, 0);
        continue;
      } else if (boost::starts_with(in, "set_weights")) { // set learning rates
        stringstream ss(in);
        string _,name; weight_t w;
        ss >> _; ss >> name; ss >> w;
        weight_t before = 0;
        ostringstream o;
        unsigned fid = FD::Convert(name);
        before = lambdas[fid];
        lambdas[fid] = w;
        o << "set weight for feature '" << name << "'"
          << "' to " << w << " (was: " << before << ")" << endl;
        string s = o.str();
        cerr << "[dtrain] " << s;
        cerr << "[dtrain] done, looping again" << endl;
        sock.send(s.c_str(), s.size()+1, 0);
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
        continue;
      } else if (boost::starts_with(in, "get_weight")) { // get weight
        stringstream ss(in);
        string _,name;
        ss >> _; ss >> name;
        cerr << "[dtrain] getting weight for " << name << endl;
        ostringstream o;
        unsigned fid = FD::Convert(name);
        weight_t w = lambdas[fid];
        o << w;
        string s = o.str();
        sock.send(s.c_str(), s.size()+1, 0);
        continue;
      } else if (boost::starts_with(in, "get_rate")) { // get rate
        stringstream ss(in);
        string _,name;
        ss >> _; ss >> name;
        cerr << "[dtrain] getting rate for " << name << endl;
        ostringstream o;
        unsigned fid = FD::Convert(name);
        weight_t  r;
        if (name == "R")
          r = learning_rate_R;
        else if (name == "RB")
          r = learning_rate_RB;
        else if (name == "Shape")
          r = learning_rate_Shape;
        else
          r = learning_rates[fid];
        o << r;
        string s = o.str();
        sock.send(s.c_str(), s.size()+1, 0);
        continue;
      } else { // translate
        vector<string> parts;
        boost::algorithm::split_regex(parts, in, boost::regex(" \\|\\|\\| "));
        if (parts[0] == "act:translate" || parts[0] == "act:translate_learn") {
          if (parts[0] == "act:translate")
            just_translate = true;
          cerr << "[dtrain] translating ..." << endl;
          lambdas.init_vector(&decoder_weights);
          observer->dont_score = true;
          decoder.Decode(parts[1], observer);
          observer->dont_score = false;
          samples = observer->GetSamples();
          if (parts[0] == "act:translate") {
            ostringstream os;
            cerr << "[dtrain] 1best features " << (*samples)[0].f << endl;
            if (output_derivation) {
              os << observer->GetViterbiTreeStr() << endl;
            } else {
              PrintWordIDVec((*samples)[0].w, os);
            }
            if (output_rules) {
              os << observer->GetViterbiRules() << endl;
            }
            sock.send(os.str().c_str(), os.str().size()+1, 0);
            cerr << "[dtrain] done translating, looping again" << endl;
          }
        } //else { // learn
        if (!just_translate) {
          cerr << "[dtrain] learning ..." << endl;
          source = parts[1];
          // debug --
          debug_output << "\"source\":\""
                       << escapeJson(source.substr(source.find_first_of(">")+2, source.find_last_of(">")-6))
                       <<  "\"," << endl;
          debug_output << "\"target\":\"" << escapeJson(parts[2]) <<  "\"," << endl;
          // -- debug
          parts.erase(parts.begin());
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

          for (size_t r = 0; r < samples->size(); r++)
            (*samples)[r].gold = observer->scorer_->Score((*samples)[r].w, refs, rsz);
        //}
        //}
        }
      }
    }

    if (!next)
      break;

    // decode
    lambdas.init_vector(&decoder_weights);

    // debug --)
    ostringstream os;
    PrintWordIDVec((*samples)[0].w, os);
    debug_output << "\"1best\":\"" << escapeJson(os.str());
    debug_output << "\"," << endl;
    debug_output << "\"kbest\":[" << endl;
    size_t h = 0;
    for (auto s: *samples) {
      debug_output << "\"" << s.gold << " ||| "
                           << s.model << " ||| " << s.rank << " ||| ";
      for (auto o: s.f)
        debug_output << escapeJson(FD::Convert(o.first)) << "=" << o.second << " ";
      debug_output << " ||| ";
      ostringstream os;
      PrintWordIDVec(s.w, os);
      debug_output << escapeJson(os.str());
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
    sparseVectorToJson(lambdas, debug_output);
    debug_output << "}," << endl;
    // -- debug
    //

    // get pairs
    SparseVector<weight_t> update;
    size_t num_up = CollectUpdates(samples, update, margin);

    // debug --
    debug_output << "\"1best_features\":{";
    sparseVectorToJson((*samples)[0].f, debug_output);
    debug_output << "}," << endl;
    debug_output << "\"update_raw\":{";
    sparseVectorToJson(update, debug_output);
    debug_output << "}," << endl;
    // -- debug

    // update
    for (auto it: update) {
      string fname = FD::Convert(it.first);
      unsigned k = it.first;
      weight_t v = it.second;
      if (learning_rates.find(it.first) != learning_rates.end()) {
        update[k] = learning_rates[k]*v;
      } else {
        if (boost::starts_with(fname, "R:")) {
          update[k] = learning_rate_R*v;
        } else if (boost::starts_with(fname, "RBS:") ||
                   boost::starts_with(fname, "RBT:")) {
          update[k] = learning_rate_RB*v;
        } else if (boost::starts_with(fname, "Shape_")) {
          update[k] = learning_rate_Shape*v;
        }
      }
    }
    if (!just_translate) {
      lambdas += update;
    } else {
      i++;
    }

    // debug --
    debug_output << "\"update\":{";
    sparseVectorToJson(update, debug_output);
    debug_output << "}," << endl;
    debug_output << "\"num_up\":" << num_up << "," << endl;
    debug_output << "\"updated_features\":" << update.size() << "," << endl;
    debug_output << "\"learning_rate_R\":" << learning_rate_R << "," << endl;
    debug_output << "\"learning_rate_RB\":" << learning_rate_R << "," << endl;
    debug_output << "\"learning_rate_Shape\":" << learning_rate_R << "," << endl;
    debug_output << "\"learning_rates\":{" << endl;
    sparseVectorToJson(learning_rates, debug_output);
    debug_output << "}," << endl;
    debug_output << "\"best_match\":\"";
    ostringstream ps;
    PrintWordIDVec((*samples)[0].w, ps);
    debug_output << escapeJson(ps.str());
    debug_output << "\"," << endl;
    debug_output << "\"best_match_score\":" << (*samples)[0].gold << "," << endl ;
    // -- debug

    // debug --
    debug_output << "\"weights_after\":{" << endl;
    sparseVectorToJson(lambdas, debug_output);
    debug_output << "}" << endl;
    debug_output << "}" << endl;
    // -- debug

    // debug --
    WriteFile f(debug_fn);
    f.get() << debug_output.str();
    f.get() << std::flush;
    // -- debug

    // write current weights
    if (!just_translate) {
      lambdas.init_vector(decoder_weights);
      ostringstream fn;
      fn << output_fn << "." << i << ".gz";
      Weights::WriteToFile(fn.str(), decoder_weights, true);
    }

    if (!just_translate) {
      cerr << "[dtrain] done learning, looping again" << endl;
      sock.send(done.c_str(), done.size()+1, 0);
    }

  } // input loop

  string shutdown = "off";
  sock.send(shutdown.c_str(), shutdown.size()+1, 0);

  cerr << "[dtrain] shutting down, goodbye" << endl;

  return 0;
}

