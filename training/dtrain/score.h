#ifndef _DTRAIN_SCORE_H_
#define _DTRAIN_SCORE_H_

#include "dtrain.h"

namespace dtrain
{

struct NgramCounts
{
  unsigned N_;
  map<unsigned, score_t> clipped_;
  map<unsigned, score_t> sum_;

  NgramCounts(const unsigned N) : N_(N) { Zero(); }

  inline void
  operator+=(const NgramCounts& rhs)
  {
    if (rhs.N_ > N_) Resize(rhs.N_);
    for (unsigned i = 0; i < N_; i++) {
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
  operator*=(const score_t rhs)
  {
    for (unsigned i = 0; i < N_; i++) {
      this->clipped_[i] *= rhs;
      this->sum_[i] *= rhs;
    }
  }

  inline void
  Add(const unsigned count, const unsigned ref_count, const unsigned i)
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
    for (unsigned i = 0; i < N_; i++) {
      clipped_[i] = 0.;
      sum_[i] = 0.;
    }
  }

  inline void
  One()
  {
    for (unsigned i = 0; i < N_; i++) {
      clipped_[i] = 1.;
      sum_[i] = 1.;
    }
  }

  inline void
  Print()
  {
    for (unsigned i = 0; i < N_; i++) {
      cout << i+1 << "grams (clipped):\t" << clipped_[i] << endl;
      cout << i+1 << "grams:\t\t\t" << sum_[i] << endl;
    }
  }

  inline void Resize(unsigned N)
  {
    if (N == N_) return;
    else if (N > N_) {
      for (unsigned i = N_; i < N; i++) {
        clipped_[i] = 0.;
        sum_[i] = 0.;
      }
    } else { // N < N_
      for (unsigned i = N_-1; i > N-1; i--) {
        clipped_.erase(i);
        sum_.erase(i);
      }
    }
    N_ = N;
  }
};

typedef map<vector<WordID>, unsigned> Ngrams;

inline Ngrams
MakeNgrams(const vector<WordID>& s, const unsigned N)
{
  Ngrams ngrams;
  vector<WordID> ng;
  for (size_t i = 0; i < s.size(); i++) {
    ng.clear();
    for (unsigned j = i; j < min(i+N, s.size()); j++) {
      ng.push_back(s[j]);
      ngrams[ng]++;
    }
  }

  return ngrams;
}

inline NgramCounts
MakeNgramCounts(const vector<WordID>& hyp, const vector<vector<WordID> >& refs, const unsigned N)
{
  Ngrams hyp_ngrams = MakeNgrams(hyp, N);
  vector<Ngrams> refs_ngrams;
  for (auto r: refs) {
    Ngrams r_ng = MakeNgrams(r, N);
    refs_ngrams.push_back(r_ng);
  }
  NgramCounts counts(N);
  Ngrams::iterator it, ti;
  for (it = hyp_ngrams.begin(); it != hyp_ngrams.end(); it++) {
    unsigned max_ref_count = 0;
    for (auto ref_ngrams: refs_ngrams) {
      ti = ref_ngrams.find(it->first);
      if (ti != ref_ngrams.end())
        max_ref_count = max(max_ref_count, ti->second);
    }
    counts.Add(it->second, min(it->second, max_ref_count), it->first.size() - 1);
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
  const unsigned N_;
  vector<score_t> w_;

  PerSentenceBleuScorer(unsigned n) : N_(n)
  {
    for (auto i = 1; i <= N_; i++)
      w_.push_back(1.0/N_);
  }

  inline score_t
  BrevityPenalty(const unsigned hyp_len, const unsigned ref_len)
  {
    if (hyp_len > ref_len) return 1;
    return exp(1 - (score_t)ref_len/hyp_len);
  }

  score_t
  Score(const vector<WordID>& hyp, const vector<vector<WordID> >& refs)
  {
    unsigned hyp_len = hyp.size(), ref_len = 0;
    // best match reference length
    if (refs.size() == 1)  {
      ref_len = refs[0].size();
    } else {
      unsigned i = 0, best_idx = 0;
      unsigned best = std::numeric_limits<unsigned>::max();
      for (auto r: refs) {
        unsigned d = abs(hyp_len-r.size());
        if (best > d) best_idx = i;
      }
      ref_len = refs[best_idx].size();
    }
    if (hyp_len == 0 || ref_len == 0) return 0.;
    NgramCounts counts = MakeNgramCounts(hyp, refs, N_);
    unsigned M = N_;
    vector<score_t> v = w_;
    if (ref_len < N_) {
      M = ref_len;
      for (unsigned i = 0; i < M; i++) v[i] = 1/((score_t)M);
    }
    score_t sum = 0, add = 0;
    for (unsigned i = 0; i < M; i++) {
      if (i == 0 && (counts.sum_[i] == 0 || counts.clipped_[i] == 0)) return 0.;
      if (i == 1) add = 1;
      sum += v[i] * log(((score_t)counts.clipped_[i] + add)/((counts.sum_[i] + add)));
    }
    return  BrevityPenalty(hyp_len, ref_len+1) * exp(sum);
  }
};

} // namespace

#endif

