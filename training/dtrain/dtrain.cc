#include "dtrain.h"
#include "score.h"
#include "sample.h"
#include "pairs.h"

using namespace dtrain;


bool
dtrain_init(int argc, char** argv, po::variables_map* conf)
{
  po::options_description ini("Configuration File Options");
  ini.add_options()
    ("bitext,b",          po::value<string>(),                                            "bitext: 'src ||| tgt ||| tgt ||| ...'")
    ("output,o",          po::value<string>()->default_value("-"),                          "output weights file, '-' for STDOUT")
    ("input_weights,w",   po::value<string>(),                                "input weights file (e.g. from previous iteration)")
    ("decoder_config,d",  po::value<string>(),                                                      "configuration file for cdec")
    ("print_weights",     po::value<string>(),                                               "weights to print on each iteration")
    ("stop_after",        po::value<unsigned>()->default_value(0),                                 "stop after X input sentences")
    ("keep",              po::value<bool>()->zero_tokens(),                               "keep weights files for each iteration")
    ("epochs",            po::value<unsigned>()->default_value(10),                               "# of iterations T (per shard)")
    ("k",                 po::value<unsigned>()->default_value(100),                            "how many translations to sample")
    ("filter",            po::value<string>()->default_value("uniq"),                          "filter kbest list: 'not', 'uniq'")
    ("pair_sampling",     po::value<string>()->default_value("XYX"),                 "how to sample pairs: 'all', 'XYX' or 'PRO'")
    ("hi_lo",             po::value<float>()->default_value(0.1),                   "hi and lo (X) for XYX (default 0.1), <= 0.5")
    ("pair_threshold",    po::value<score_t>()->default_value(0.),                         "bleu [0,1] threshold to filter pairs")
    ("N",                 po::value<unsigned>()->default_value(4),                                          "N for Ngrams (BLEU)")
    ("scorer",            po::value<string>()->default_value("stupid_bleu"),      "scoring: bleu, stupid_, smooth_, approx_, lc_")
    ("learning_rate",     po::value<weight_t>()->default_value(1.0),                                              "learning rate")
    ("gamma",             po::value<weight_t>()->default_value(0.),                            "gamma for SVM (0 for perceptron)")
    ("select_weights",    po::value<string>()->default_value("last"),     "output best, last, avg weights ('VOID' to throw away)")
    ("rescale",           po::value<bool>()->zero_tokens(),                     "(re)scale data and weight vector to unit length")
    ("l1_reg",            po::value<string>()->default_value("none"),      "apply l1 regularization with clipping as in 'Tsuroka et al' (2010)")
    ("l1_reg_strength",   po::value<weight_t>(),                                                     "l1 regularization strength")
    ("fselect",           po::value<weight_t>()->default_value(-1), "select top x percent (or by threshold) of features after each epoch NOT IMPLEMENTED") // TODO
    ("loss_margin",       po::value<weight_t>()->default_value(0.),  "update if no error in pref pair but model scores this near")
    ("max_pairs",         po::value<unsigned>()->default_value(std::numeric_limits<unsigned>::max()), "max. # of pairs per Sent.")
    ("pclr",              po::value<string>()->default_value("no"),         "use a (simple|adagrad) per-coordinate learning rate")
    ("batch",             po::value<bool>()->zero_tokens(),                                               "do batch optimization")
    ("repeat",            po::value<unsigned>()->default_value(1),     "repeat optimization over kbest list this number of times")
    ("output_ranking",    po::value<string>()->default_value(""),                                   "output scored kbests to dir")
    ("noup",              po::value<bool>()->zero_tokens(),                                                     "dont't optimize");
  po::options_description cl("Command Line Options");
  cl.add_options()
    ("config,c",         po::value<string>(),              "dtrain config file")
    ("quiet,q",          po::value<bool>()->zero_tokens(),           "be quiet")
    ("verbose,v",        po::value<bool>()->zero_tokens(),         "be verbose");
  cl.add(ini);
  po::store(parse_command_line(argc, argv, cl), *conf);
  if (conf->count("config")) {
    ifstream ini_f((*conf)["config"].as<string>().c_str());
    po::store(po::parse_config_file(ini_f, ini), *conf);
  }
  po::notify(*conf);
  if (!conf->count("decoder_config")) {
    cerr << cl << endl;
    return false;
  }
  if ((*conf)["pair_sampling"].as<string>() != "all" && (*conf)["pair_sampling"].as<string>() != "XYX" &&
        (*conf)["pair_sampling"].as<string>() != "PRO" && (*conf)["pair_sampling"].as<string>() != "output_pairs") {
    cerr << "Wrong 'pair_sampling' param: '" << (*conf)["pair_sampling"].as<string>() << "'." << endl;
    return false;
  }
  if (conf->count("hi_lo") && (*conf)["pair_sampling"].as<string>() != "XYX") {
    cerr << "Warning: hi_lo only works with pair_sampling XYX." << endl;
  }
  if ((*conf)["hi_lo"].as<float>() > 0.5 || (*conf)["hi_lo"].as<float>() < 0.01) {
    cerr << "hi_lo must lie in [0.01, 0.5]" << endl;
    return false;
  }
  if (!conf->count("bitext")) {
    cerr << "No training data given." << endl;
    return false;
  }
  if ((*conf)["pair_threshold"].as<score_t>() < 0) {
    cerr << "The threshold must be >= 0!" << endl;
    return false;
  }
  if ((*conf)["select_weights"].as<string>() != "last" && (*conf)["select_weights"].as<string>() != "best" &&
        (*conf)["select_weights"].as<string>() != "avg" && (*conf)["select_weights"].as<string>() != "VOID") {
    cerr << "Wrong 'select_weights' param: '" << (*conf)["select_weights"].as<string>() << "', use 'last' or 'best'." << endl;
    return false;
  }
  return true;
}

int
main(int argc, char** argv)
{
  // handle most parameters
  po::variables_map conf;
  if (!dtrain_init(argc, argv, &conf)) exit(1); // something is wrong

  bool quiet = false;
  if (conf.count("quiet")) quiet = true;
  bool verbose = false;
  if (conf.count("verbose")) verbose = true;
  bool noup = false;
  if (conf.count("noup")) noup = true;
  bool keep = false;
  if (conf.count("keep")) keep = true;
  bool rescale = false;
  if (conf.count("rescale")) rescale = true;

  const unsigned k = conf["k"].as<unsigned>();
  const unsigned N = conf["N"].as<unsigned>();
  const unsigned T = conf["epochs"].as<unsigned>();
  const unsigned stop_after = conf["stop_after"].as<unsigned>();
  const string pair_sampling = conf["pair_sampling"].as<string>();
  const score_t pair_threshold = conf["pair_threshold"].as<score_t>();
  const string select_weights = conf["select_weights"].as<string>();
  const string output_ranking = conf["output_ranking"].as<string>();
  const float hi_lo = conf["hi_lo"].as<float>();
  const unsigned max_pairs = conf["max_pairs"].as<unsigned>();
  int repeat = conf["repeat"].as<unsigned>();
  weight_t loss_margin = conf["loss_margin"].as<weight_t>();
  bool batch = false;
  if (conf.count("batch")) batch = true;
  if (loss_margin > 9998.) loss_margin = std::numeric_limits<float>::max();
  const string pclr = conf["pclr"].as<string>();
  bool average = false;
  if (select_weights == "avg")
    average = true;
  vector<string> print_weights;
  if (conf.count("print_weights"))
    boost::split(print_weights, conf["print_weights"].as<string>(), boost::is_any_of(" "));

  // setup decoder
  register_feature_functions();
  SetSilent(true);
  ReadFile ini_rf(conf["decoder_config"].as<string>());
  if (!quiet)
    cerr << setw(25) << "cdec conf " << "'" << conf["decoder_config"].as<string>() << "'" << endl;
  Decoder decoder(ini_rf.stream());

  // setup decoder observer
  ScoredKbest* observer = new ScoredKbest(k, new PerSentenceBleuScorer(N));

  // init weights
  vector<weight_t>& decoder_weights = decoder.CurrentWeightVector();

  SparseVector<weight_t> lambdas, cumulative_penalties, w_average, fixed;
  if (conf.count("input_weights"))
    Weights::InitFromFile(conf["input_weights"].as<string>(), &decoder_weights);
  Weights::InitSparseVector(decoder_weights, &lambdas);

  // meta params for perceptron, SVM
  weight_t eta = conf["learning_rate"].as<weight_t>();
  weight_t gamma = conf["gamma"].as<weight_t>();

  // faster perceptron: consider only misranked pairs, see
  bool faster_perceptron = false;
  if (gamma==0 && loss_margin==0) faster_perceptron = true;

  // l1 regularization
  bool l1naive = false;
  bool l1clip = false;
  bool l1cumul = false;
  weight_t l1_reg = 0;
  if (conf["l1_reg"].as<string>() != "none") {
    string s = conf["l1_reg"].as<string>();
    if (s == "naive") l1naive = true;
    else if (s == "clip") l1clip = true;
    else if (s == "cumul") l1cumul = true;
    l1_reg = conf["l1_reg_strength"].as<weight_t>();
  }

  // output
  string output_fn = conf["output"].as<string>();
  // input
  string input_fn;
  ReadFile input(conf["bitext"].as<string>());
  // buffer input for t > 0
  vector<string> src_str_buf;          // source strings (decoder takes only strings)
  vector<vector<vector<WordID> > > refs_as_ids_buf; // references as WordID vecs

  unsigned in_sz = std::numeric_limits<unsigned>::max(); // input index, input size
  vector<pair<score_t, score_t> > all_scores;
  score_t max_score = 0.;
  unsigned best_it = 0;
  float overall_time = 0.;

  // output conf
  if (!quiet) {
    cerr << _p5;
    cerr << endl << "dtrain" << endl << "Parameters:" << endl;
    cerr << setw(25) << "k " << k << endl;
    cerr << setw(25) << "N " << N << endl;
    cerr << setw(25) << "T " << T << endl;
    cerr << setw(25) << "batch " << batch << endl;
    cerr << setw(25) << "learning rate " << eta << endl;
    cerr << setw(25) << "gamma " << gamma << endl;
    cerr << setw(25) << "loss margin " << loss_margin << endl;
    cerr << setw(25) << "faster perceptron " << faster_perceptron << endl;
    cerr << setw(25) << "pairs " << "'" << pair_sampling << "'" << endl;
    if (pair_sampling == "XYX")
      cerr << setw(25) << "hi lo " << hi_lo << endl;
    cerr << setw(25) << "pair threshold " << pair_threshold << endl;
    cerr << setw(25) << "select weights " << "'" << select_weights << "'" << endl;
    if (conf.count("l1_reg"))
      cerr << setw(25) << "l1 reg " << l1_reg << " '" << conf["l1_reg"].as<string>() << "'" << endl;
    if (rescale)
      cerr << setw(25) << "rescale " << rescale << endl;
    cerr << setw(25) << "pclr " << pclr << endl;
    cerr << setw(25) << "max pairs " << max_pairs << endl;
    cerr << setw(25) << "repeat " << repeat << endl;
    cerr << setw(25) << "cdec conf " << "'" << conf["decoder_config"].as<string>() << "'" << endl;
    cerr << setw(25) << "input " << "'" << input_fn << "'" << endl;
    cerr << setw(25) << "output " << "'" << output_fn << "'" << endl;
    if (conf.count("input_weights"))
      cerr << setw(25) << "weights in " << "'" << conf["input_weights"].as<string>() << "'" << endl;
    if (stop_after > 0)
      cerr << setw(25) << "stop_after " << stop_after << endl;
    if (!verbose) cerr << "(a dot represents " << DTRAIN_DOTS << " inputs)" << endl;
  }

  // pclr
  SparseVector<weight_t> learning_rates;
  // batch
  SparseVector<weight_t> batch_updates;
  score_t batch_loss;

  for (unsigned t = 0; t < T; t++) // T epochs
  {

  time_t start, end;
  time(&start);
  score_t score_sum = 0.;
  score_t model_sum(0);
  unsigned ii = 0, rank_errors = 0, margin_violations = 0, npairs = 0, f_count = 0, list_sz = 0, kbest_loss_improve = 0;
  batch_loss = 0.;
  if (!quiet) cerr << "Iteration #" << t+1 << " of " << T << "." << endl;

  while(true)
  {

    string in;
    vector<string> refs;
    bool next = false, stop = false; // next iteration or premature stop
    if (t == 0) {
      if(!getline(*input, in)) next = true;
        boost::algorithm::split_regex(refs, in, boost::regex(" \\|\\|\\| "));
        in = refs[0];
        refs.erase(refs.begin());
    } else {
      if (ii == in_sz) next = true; // stop if we reach the end of our input
    }
    // stop after X sentences (but still go on for those)
    if (stop_after > 0 && stop_after == ii && !next) stop = true;

    // produce some pretty output
    if (!quiet && !verbose) {
      if (ii == 0) cerr << " ";
      if ((ii+1) % (DTRAIN_DOTS) == 0) {
        cerr << ".";
        cerr.flush();
      }
      if ((ii+1) % (20*DTRAIN_DOTS) == 0) {
        cerr << " " << ii+1 << endl;
        if (!next && !stop) cerr << " ";
      }
      if (stop) {
        if (ii % (20*DTRAIN_DOTS) != 0) cerr << " " << ii << endl;
        cerr << "Stopping after " << stop_after << " input sentences." << endl;
      } else {
        if (next) {
          if (ii % (20*DTRAIN_DOTS) != 0) cerr << " " << ii << endl;
        }
      }
    }

    // next iteration
    if (next || stop) break;

    // weights
    lambdas.init_vector(&decoder_weights);

    // getting input
    if (t == 0) {
      vector<vector<WordID> > cur_refs;
      for (auto r: refs) {
        vector<WordID> cur_ref;
        vector<string> tok;
        boost::split(tok, r, boost::is_any_of(" "));
        RegisterAndConvert(tok, cur_ref);
        cur_refs.push_back(cur_ref);
      }
      refs_as_ids_buf.push_back(cur_refs);
      src_str_buf.push_back(in);
    }
    observer->SetReference(refs_as_ids_buf[ii]);
    if (t == 0)
      decoder.Decode(in, observer);
    else
      decoder.Decode(src_str_buf[ii], observer);

    // get (scored) samples
    vector<ScoredHyp>* samples = observer->GetSamples();

    if (output_ranking != "") {
      WriteFile of(output_ranking+"/"+to_string(t)+"."+to_string(ii)+".list"); // works with '-'
      stringstream ss;
      for (auto s: *samples) {
        ss << ii << " ||| ";
        PrintWordIDVec(s.w, ss);
        ss << " ||| " << s.model << " ||| " << s.score << endl;
      }
      of.get() << ss.str();
    }

    if (verbose) {
      cerr << "--- refs for " << ii << ": ";
      for (auto r: refs_as_ids_buf[ii]) {
        PrintWordIDVec(r);
        cerr << endl;
      }
      for (unsigned u = 0; u < samples->size(); u++) {
        cerr << _p2 << _np << "[" << u << ". '";
        PrintWordIDVec((*samples)[u].w);
        cerr << "'" << endl;
        cerr << "SCORE=" << (*samples)[u].score << ",model="<< (*samples)[u].model << endl;
        cerr << "F{" << (*samples)[u].f << "} ]" << endl << endl;
      }
    }

    if (repeat == 1) {
      score_sum += (*samples)[0].score; // stats for 1best
      model_sum += (*samples)[0].model;
    }

    f_count += observer->GetFeatureCount();
    list_sz += observer->GetSize();

    // weight updates
    if (!noup) {
      // get pairs
      vector<pair<ScoredHyp,ScoredHyp> > pairs;
      if (pair_sampling == "all")
        all_pairs(samples, pairs, pair_threshold, max_pairs, faster_perceptron);
      if (pair_sampling == "XYX")
        partXYX(samples, pairs, pair_threshold, max_pairs, faster_perceptron, hi_lo);
      if (pair_sampling == "PRO")
        PROsampling(samples, pairs, pair_threshold, max_pairs);
      if (pair_sampling == "output_pairs")
        all_pairs(samples, pairs, pair_threshold, max_pairs, false);
      int cur_npairs = pairs.size();
      npairs += cur_npairs;

      score_t kbest_loss_first = 0.0, kbest_loss_last = 0.0;

      if (pair_sampling == "output_pairs") {
        for (auto p: pairs) {
          cout << p.first.model << " ||| "  << p.first.score << " ||| " <<  p.first.f  << endl;
          cout << p.second.model << " ||| "  << p.second.score << " ||| " <<  p.second.f  << endl;
          cout << endl;
        }
        continue;
      }

      for (vector<pair<ScoredHyp,ScoredHyp> >::iterator it = pairs.begin();
           it != pairs.end(); it++) {
        if (rescale) {
          it->first.f /= it->first.f.l2norm();
          it->second.f /= it->second.f.l2norm();
        }
        score_t model_diff = it->first.model - it->second.model;
        score_t loss = max(0.0, -1.0 * model_diff);
        kbest_loss_first += loss;
      }

      score_t kbest_loss = 0.0;
      for (int ki=0; ki < repeat; ki++) {

      SparseVector<weight_t> lambdas_copy; // for l1 regularization
      SparseVector<weight_t> sum_up; // for pclr
      if (l1naive||l1clip||l1cumul) lambdas_copy = lambdas;

      for (vector<pair<ScoredHyp,ScoredHyp> >::iterator it = pairs.begin();
           it != pairs.end(); it++) {
        score_t model_diff = it->first.model - it->second.model;
        score_t loss = max(0.0, -1.0 * model_diff);

        if (repeat > 1) {
          model_diff = lambdas.dot(it->first.f) - lambdas.dot(it->second.f);
          kbest_loss += loss;
        }
        bool rank_error = false;
        score_t margin;
        if (faster_perceptron) { // we only have considering misranked pairs
          rank_error = true; // pair sampling already did this for us
          margin = std::numeric_limits<float>::max();
        } else {
          rank_error = model_diff<=0.0;
          margin = fabs(model_diff);
          if (!rank_error && margin < loss_margin) margin_violations++;
        }
        if (rank_error && ki==0) rank_errors++;
        if (rank_error || margin < loss_margin) {
          SparseVector<weight_t> diff_vec = it->first.f - it->second.f;
          if (batch) {
            batch_loss += max(0., -1.0 * model_diff);
            batch_updates += diff_vec;
            continue;
          }
          if (pclr != "no") {
            sum_up += diff_vec;
          } else {
            lambdas.plus_eq_v_times_s(diff_vec, eta);
            if (gamma) lambdas.plus_eq_v_times_s(lambdas, -2*gamma*eta*(1./cur_npairs));
          }
        }
      }

      // per-coordinate learning rate
      if (pclr != "no") {
        SparseVector<weight_t>::iterator it = sum_up.begin();
        for (; it != sum_up.end(); ++it) {
          if (pclr == "simple") {
           lambdas[it->first] += it->second / max(1.0, learning_rates[it->first]);
           learning_rates[it->first]++;
          } else if (pclr == "adagrad") {
            if (learning_rates[it->first] == 0) {
             lambdas[it->first] +=  it->second * eta;
            } else {
             lambdas[it->first] +=  it->second * eta * learning_rates[it->first];
            }
            learning_rates[it->first] += pow(it->second, 2.0);
          }
        }
      }

      // l1 regularization
      // please note that this regularizations happen
      // after a _sentence_ -- not after each example/pair!
      if (l1naive) {
        SparseVector<weight_t>::iterator it = lambdas.begin();
        for (; it != lambdas.end(); ++it) {
          if (!lambdas_copy.get(it->first) || lambdas_copy.get(it->first)!=it->second) {
              it->second *= max(0.0000001, eta/(eta+learning_rates[it->first])); // FIXME
              learning_rates[it->first]++;
            it->second -= sign(it->second) * l1_reg;
          }
        }
      } else if (l1clip) {
        SparseVector<weight_t>::iterator it = lambdas.begin();
        for (; it != lambdas.end(); ++it) {
          if (!lambdas_copy.get(it->first) || lambdas_copy.get(it->first)!=it->second) {
            if (it->second != 0) {
              weight_t v = it->second;
              if (v > 0) {
                it->second = max(0., v - l1_reg);
              } else {
                it->second = min(0., v + l1_reg);
              }
            }
          }
        }
      } else if (l1cumul) {
        weight_t acc_penalty = (ii+1) * l1_reg; // ii is the index of the current input
        SparseVector<weight_t>::iterator it = lambdas.begin();
        for (; it != lambdas.end(); ++it) {
          if (!lambdas_copy.get(it->first) || lambdas_copy.get(it->first)!=it->second) {
            if (it->second != 0) {
              weight_t v = it->second;
              weight_t penalized = 0.;
              if (v > 0) {
                penalized = max(0., v-(acc_penalty + cumulative_penalties.get(it->first)));
              } else {
                penalized = min(0., v+(acc_penalty - cumulative_penalties.get(it->first)));
              }
              it->second = penalized;
              cumulative_penalties.set_value(it->first, cumulative_penalties.get(it->first)+penalized);
            }
          }
        }
      }

      if (ki==repeat-1) { // done
        kbest_loss_last = kbest_loss;
        if (repeat > 1) {
          score_t best_model = -std::numeric_limits<score_t>::max();
          unsigned best_idx = 0;
          for (unsigned i=0; i < samples->size(); i++) {
            score_t s = lambdas.dot((*samples)[i].f);
            if (s > best_model) {
              best_idx = i;
              best_model = s;
            }
          }
          score_sum += (*samples)[best_idx].score;
          model_sum += best_model;
        }
      }
    } // repeat

    if ((kbest_loss_first - kbest_loss_last) >= 0) kbest_loss_improve++;

    } // noup

    if (rescale) lambdas /= lambdas.l2norm();

    ++ii;

  } // input loop

  if (t == 0) in_sz = ii; // remember size of input (# lines)

  if (batch) {
    lambdas.plus_eq_v_times_s(batch_updates, eta);
    if (gamma) lambdas.plus_eq_v_times_s(lambdas, -2*gamma*eta*(1./npairs));
    batch_updates.clear();
  }

  if (average) w_average += lambdas;

  // print some stats
  score_t score_avg = score_sum/(score_t)in_sz;
  score_t model_avg = model_sum/(score_t)in_sz;
  score_t score_diff, model_diff;
  if (t > 0) {
    score_diff = score_avg - all_scores[t-1].first;
    model_diff = model_avg - all_scores[t-1].second;
  } else {
    score_diff = score_avg;
    model_diff = model_avg;
  }

  unsigned nonz = 0;
  if (!quiet) nonz = (unsigned)lambdas.num_nonzero();

  if (!quiet) {
    cerr << _p5 << _p << "WEIGHTS" << endl;
    for (vector<string>::iterator it = print_weights.begin(); it != print_weights.end(); it++) {
      cerr << setw(18) << *it << " = " << lambdas.get(FD::Convert(*it)) << endl;
    }
    cerr << "        ---" << endl;
    cerr << _np << "       1best avg score: " << score_avg;
    cerr << _p << " (" << score_diff << ")" << endl;
    cerr << _np << " 1best avg model score: " << model_avg;
    cerr << _p << " (" << model_diff << ")" << endl;
    cerr << "           avg # pairs: ";
    cerr << _np << npairs/(float)in_sz << endl;
    cerr << "        avg # rank err: ";
    cerr << rank_errors/(float)in_sz;
    if (faster_perceptron) cerr << " (meaningless)";
    cerr << endl;
    cerr << "     avg # margin viol: ";
    cerr << margin_violations/(float)in_sz << endl;
    if (batch) cerr << "            batch loss: " << batch_loss << endl;
    cerr << "       k-best loss imp: " << ((float)kbest_loss_improve/in_sz)*100 << "%" << endl;
    cerr << "    non0 feature count: " <<  nonz << endl;
    cerr << "           avg list sz: " << list_sz/(float)in_sz << endl;
    cerr << "           avg f count: " << f_count/(float)list_sz << endl;
  }

  pair<score_t,score_t> remember;
  remember.first = score_avg;
  remember.second = model_avg;
  all_scores.push_back(remember);
  if (score_avg > max_score) {
    max_score = score_avg;
    best_it = t;
  }
  time (&end);
  float time_diff = difftime(end, start);
  overall_time += time_diff;
  if (!quiet) {
    cerr << _p2 << _np << "(time " << time_diff/60. << " min, ";
    cerr << time_diff/in_sz << " s/S)" << endl;
  }
  if (t+1 != T && !quiet) cerr << endl;

  if (noup) break;

  // write weights to file
  if (select_weights == "best" || keep) {
    lambdas.init_vector(&decoder_weights);
    string w_fn = "weights." + boost::lexical_cast<string>(t) + ".gz";
    Weights::WriteToFile(w_fn, decoder_weights, true);
  }

  } // outer loop

  if (average) w_average /= (weight_t)T;

  if (!noup) {
    if (!quiet) cerr << endl << "Writing weights file to '" << output_fn << "' ..." << endl;
    if (select_weights == "last" || average) { // last, average
      WriteFile of(output_fn);
      ostream& o = *of.stream();
      o.precision(17);
      o << _np;
      if (average) {
        for (SparseVector<weight_t>::iterator it = w_average.begin(); it != w_average.end(); ++it) {
	      if (it->second == 0) continue;
          o << FD::Convert(it->first) << '\t' << it->second << endl;
        }
      } else {
        for (SparseVector<weight_t>::iterator it = lambdas.begin(); it != lambdas.end(); ++it) {
	      if (it->second == 0) continue;
          o << FD::Convert(it->first) << '\t' << it->second << endl;
        }
      }
    } else if (select_weights == "VOID") { // do nothing with the weights
    } else { // best
      if (output_fn != "-") {
        CopyFile("weights."+boost::lexical_cast<string>(best_it)+".gz", output_fn);
      } else {
        ReadFile bestw("weights."+boost::lexical_cast<string>(best_it)+".gz");
        string o;
        cout.precision(17);
        cout << _np;
        while(getline(*bestw, o)) cout << o << endl;
      }
      if (!keep) {
        for (unsigned i = 0; i < T; i++) {
          string s = "weights." + boost::lexical_cast<string>(i) + ".gz";
          unlink(s.c_str());
        }
      }
    }
    if (!quiet) cerr << "done" << endl;
  }

  if (!quiet) {
    cerr << _p5 << _np << endl << "---" << endl << "Best iteration: ";
    cerr << best_it+1 << " [SCORE = " << max_score << "]." << endl;
    cerr << "This took " << overall_time/60. << " min." << endl;
  }
}

