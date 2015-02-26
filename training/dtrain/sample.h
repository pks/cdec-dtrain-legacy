#ifndef _DTRAIN_SAMPLE_H_
#define _DTRAIN_SAMPLE_H_

#include "kbest.h"

namespace dtrain
{


struct ScoredKbest : public DecoderObserver
{
  const size_t k_;
  vector<ScoredHyp> s_;
  size_t src_len_;
  PerSentenceBleuScorer* scorer_;
  vector<vector<WordID> >* refs_;
  vector<Ngrams>* ref_ngs_;
  vector<size_t>* ref_ls_;
  size_t f_count_, sz_;

  ScoredKbest(const size_t k, PerSentenceBleuScorer* scorer) :
    k_(k), scorer_(scorer) {}

  virtual void
  NotifyTranslationForest(const SentenceMetadata& smeta, Hypergraph* hg)
  {
    src_len_ = smeta.GetSourceLength();
    s_.clear(); sz_ = f_count_ = 0;
    KBest::KBestDerivations<vector<WordID>, ESentenceTraversal,
      KBest::FilterUnique, prob_t, EdgeProb> kbest(*hg, k_);
    for (size_t i = 0; i < k_; ++i) {
      const KBest::KBestDerivations<vector<WordID>, ESentenceTraversal, KBest::FilterUnique,
              prob_t, EdgeProb>::Derivation* d =
            kbest.LazyKthBest(hg->nodes_.size() - 1, i);
      if (!d) break;
      ScoredHyp h;
      h.w = d->yield;
      h.f = d->feature_values;
      h.model = log(d->score);
      h.rank = i;
      h.gold = scorer_->Score(h.w, *ref_ngs_, *ref_ls_);
      s_.push_back(h);
      sz_++;
      f_count_ += h.f.size();
    }
  }

  vector<ScoredHyp>* GetSamples() { return &s_; }
  inline void SetReference(vector<Ngrams>& ngs, vector<size_t>& ls)
  {
    ref_ngs_ = &ngs;
    ref_ls_ = &ls;
  }
  inline size_t GetFeatureCount() { return f_count_; }
  inline size_t GetSize() { return sz_; }
};


} // namespace

#endif

