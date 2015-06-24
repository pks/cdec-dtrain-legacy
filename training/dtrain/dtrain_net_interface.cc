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
  const weight_t margin       = conf["margin"].as<weight_t>();
  const string master_addr    = conf["addr"].as<string>();
  const string output_fn      = conf["output"].as<string>();
  const string debug_fn       = conf["debug_output"].as<string>();

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
  cerr << "dtrain_net_interface" << endl << "Parameters:" << endl;
  cerr << setw(25) << "k " << k << endl;
  cerr << setw(25) << "N " << N << endl;
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

  size_t i = 0;
  while(true)
  {
    // debug --
    debug_output.str(string());
    debug_output.clear();
    debug_output << "{" << endl;
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
      if (in == "shutdown") { // shut down
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
      debug_output << "EgivenFCoherent=" << s.f[FD::Convert("EgivenFCoherent")] << " ";
      debug_output << "SampleCountF=" << s.f[FD::Convert("CountEF")] << " ";
      debug_output << "MaxLexFgivenE=" << s.f[FD::Convert("MaxLexFgivenE")] << " ";
      debug_output << "MaxLexEgivenF=" << s.f[FD::Convert("MaxLexEgivenF")] << " ";
      debug_output << "IsSingletonF=" << s.f[FD::Convert("IsSingletonF")] << " ";
      debug_output << "IsSingletonFE=" << s.f[FD::Convert("IsSingletonFE")] << " ";
      debug_output << "Glue=:" << s.f[FD::Convert("Glue")] << " ";
      debug_output << "WordPenalty=" << s.f[FD::Convert("WordPenalty")] << " ";
      debug_output << "PassThrough=" << s.f[FD::Convert("PassThrough")] << " ";
      debug_output << "LanguageModel=" << s.f[FD::Convert("LanguageModel_OOV")];
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
    debug_output << "\"EgivenFCoherent\":" << lambdas[FD::Convert("EgivenFCoherent")] << "," << endl;
    debug_output << "\"SampleCountF\":" << lambdas[FD::Convert("CountEF")] << "," << endl;
    debug_output << "\"MaxLexFgivenE\":" << lambdas[FD::Convert("MaxLexFgivenE")] << "," << endl;
    debug_output << "\"MaxLexEgivenF\":" << lambdas[FD::Convert("MaxLexEgivenF")] << "," << endl;
    debug_output << "\"IsSingletonF\":" << lambdas[FD::Convert("IsSingletonF")] << "," << endl;
    debug_output << "\"IsSingletonFE\":" << lambdas[FD::Convert("IsSingletonFE")] << "," << endl;
    debug_output << "\"Glue\":" << lambdas[FD::Convert("Glue")] << "," << endl;
    debug_output << "\"WordPenalty\":" << lambdas[FD::Convert("WordPenalty")] << "," << endl;
    debug_output << "\"PassThrough\":" << lambdas[FD::Convert("PassThrough")] << "," << endl;
    debug_output << "\"LanguageModel\":" << lambdas[FD::Convert("LanguageModel_OOV")] << endl;
    debug_output << "}," << endl;
    // -- debug

    // get pairs and update
    SparseVector<weight_t> updates;
    size_t num_up = CollectUpdates(samples, updates, margin);

    // debug --
    debug_output << "\"num_up\":" << num_up << "," << endl;
    debug_output << "\"updated_features\":" << updates.size() << "," << endl;
    debug_output << "\"learning_rate\":" << eta << "," << endl;
    debug_output << "\"best_match\":\"";
    PrintWordIDVec((*samples)[0].w, debug_output);
    debug_output << "\"," << endl;
    debug_output << "\"best_match_score\":" << (*samples)[0].gold << "," << endl ;
    // -- debug

    lambdas.plus_eq_v_times_s(updates, eta);
    i++;

    // debug --
    debug_output << "\"weights_after\":{" << endl;
    debug_output << "\"EgivenFCoherent\":" << lambdas[FD::Convert("EgivenFCoherent")] << "," << endl;
    debug_output << "\"SampleCountF\":" << lambdas[FD::Convert("CountEF")] << "," << endl;
    debug_output << "\"MaxLexFgivenE\":" << lambdas[FD::Convert("MaxLexFgivenE")] << "," << endl;
    debug_output << "\"MaxLexEgivenF\":" << lambdas[FD::Convert("MaxLexEgivenF")] << "," << endl;
    debug_output << "\"IsSingletonF\":" << lambdas[FD::Convert("IsSingletonF")] << "," << endl;
    debug_output << "\"IsSingletonFE\":" << lambdas[FD::Convert("IsSingletonFE")] << "," << endl;
    debug_output << "\"Glue\":" << lambdas[FD::Convert("Glue")] << "," << endl;
    debug_output << "\"WordPenalty\":" << lambdas[FD::Convert("WordPenalty")] << "," << endl;
    debug_output << "\"PassThrough\":" << lambdas[FD::Convert("PassThrough")] << "," << endl;
    debug_output << "\"LanguageModel\":" << lambdas[FD::Convert("LanguageModel_OOV")] << endl;
    debug_output << "}" << endl;
    debug_output << "}" << endl;
    // -- debug

    cerr << "[dtrain] done learning, looping again" << endl;
    string done = "done";
    sock.send(done.c_str(), done.size()+1, 0);

    // debug --
    WriteFile f(debug_fn);
    *f << debug_output.str();
    // -- debug
  } // input loop

  if (output_fn != "") {
    cerr << "[dtrain] writing final weights to '" << output_fn << "'" << endl;
    lambdas.init_vector(decoder_weights);
    Weights::WriteToFile(output_fn, decoder_weights, true);
  }

  string shutdown = "off";
  sock.send(shutdown.c_str(), shutdown.size()+1, 0);

  cerr << "[dtrain] shutting down, goodbye" << endl;

  return 0;
}

