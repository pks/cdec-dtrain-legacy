#ifndef _DTRAIN_SCORE_H_
#define _DTRAIN_SCORE_H_

#include "dtrain.h"

namespace dtrain
{

struct NgramCounts
{
  size_t N_;
  map<size_t, weight_t> clipped_;
  map<size_t, weight_t> sum_;

  NgramCounts(const size_t N) : N_(N) { Zero(); }

  inline void
  operator+=(const NgramCounts& rhs)
  {
    if (rhs.N_ > N_) Resize(rhs.N_);
    for (size_t i = 0; i < N_; i++) {
      this->clipped_[i] += rhs.clipped_.find(i)->second;
      this->sum_[i] += rhs.sum_.find(i)->second;
    }
  }

  inline const NgramCounts
  operator+(const NgramCounts &other) const
  {
    NgramCounts result = *this;
    result += other;

    return result;
  }

  inline void
  Add(const size_t count, const size_t ref_count, const size_t i)
  {
    assert(i < N_);
    if (count > ref_count) {
      clipped_[i] += ref_count;
    } else {
      clipped_[i] += count;
    }
    sum_[i] += count;
  }

  inline void
  Zero()
  {
    for (size_t i = 0; i < N_; i++) {
      clipped_[i] = 0.;
      sum_[i] = 0.;
    }
  }

  inline void
  Resize(size_t N)
  {
    if (N == N_) return;
    else if (N > N_) {
      for (size_t i = N_; i < N; i++) {
        clipped_[i] = 0.;
        sum_[i] = 0.;
      }
    } else { // N < N_
      for (size_t i = N_-1; i > N-1; i--) {
        clipped_.erase(i);
        sum_.erase(i);
      }
    }
    N_ = N;
  }
};

typedef map<vector<WordID>, size_t> Ngrams;

inline Ngrams
MakeNgrams(const vector<WordID>& s, const size_t N)
{
  Ngrams ngrams;
  vector<WordID> ng;
  for (size_t i = 0; i < s.size(); i++) {
    ng.clear();
    for (size_t j = i; j < min(i+N, s.size()); j++) {
      ng.push_back(s[j]);
      ngrams[ng]++;
    }
  }

  return ngrams;
}

inline NgramCounts
MakeNgramCounts(const vector<WordID>& hyp,
                const vector<Ngrams>& ref,
                const size_t N)
{
  Ngrams hyp_ngrams = MakeNgrams(hyp, N);
  NgramCounts counts(N);
  Ngrams::iterator it, ti;
  for (it = hyp_ngrams.begin(); it != hyp_ngrams.end(); it++) {
    size_t max_ref_count = 0;
    for (auto r: ref) {
      ti = r.find(it->first);
      if (ti != r.end())
        max_ref_count = max(max_ref_count, ti->second);
    }
    counts.Add(it->second, min(it->second, max_ref_count), it->first.size()-1);
  }

  return counts;
}

/*
 * per-sentence BLEU
 * as in "Optimizing for Sentence-Level BLEU+1
 *        Yields Short Translations"
 * (Nakov et al. '12)
 *
 * [simply add 1 to reference length for calculation of BP]
 *
 */
struct PerSentenceBleuScorer
{
  const size_t     N_;
  vector<weight_t> w_;

  PerSentenceBleuScorer(size_t n) : N_(n)
  {
    for (size_t i = 1; i <= N_; i++)
      w_.push_back(1.0/N_);
  }

  inline weight_t
  BrevityPenalty(const size_t hl, const size_t rl)
  {
    if (hl > rl)
      return 1;

    return exp(1 - (weight_t)rl/hl);
  }

  inline size_t
  BestMatchLength(const size_t hl,
                  const vector<size_t>& ref_ls)
  {
    size_t m;
    if (ref_ls.size() == 1)  {
      m = ref_ls.front();
    } else {
      size_t i = 0, best_idx = 0;
      size_t best = numeric_limits<size_t>::max();
      for (auto l: ref_ls) {
        size_t d = abs(hl-l);
        if (d < best) { 
          best_idx = i;
          best = d;
        }
        i += 1;
      }
      m = ref_ls[best_idx];
    }

    return m;
  }

  weight_t
  Score(const vector<WordID>& hyp,
        const vector<Ngrams>& ref_ngs,
        const vector<size_t>& ref_ls)
  {
    size_t hl = hyp.size(), rl = 0;
    if (hl == 0) return 0.;
    rl = BestMatchLength(hl, ref_ls);
    if (rl == 0) return 0.;
    NgramCounts counts = MakeNgramCounts(hyp, ref_ngs, N_);
    size_t M = N_;
    vector<weight_t> v = w_;
    if (rl < N_) {
      M = rl;
      for (size_t i = 0; i < M; i++) v[i] = 1/((weight_t)M);
    }
    weight_t sum = 0, add = 0;
    for (size_t i = 0; i < M; i++) {
      if (i == 0 && (counts.sum_[i] == 0 || counts.clipped_[i] == 0)) return 0.;
      if (i > 0) add = 1;
      sum += v[i] * log(((weight_t)counts.clipped_[i] + add)
                        / ((counts.sum_[i] + add)));
    }

    return  BrevityPenalty(hl, rl+1) * exp(sum);
  }
};

} // namespace

#endif

