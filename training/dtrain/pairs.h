#ifndef _DTRAIN_PAIRS_H_
#define _DTRAIN_PAIRS_H_

namespace dtrain
{

bool
CmpHypsByScore(ScoredHyp a, ScoredHyp b)
{
  return a.score > b.score;
}

/*
 * multipartite ranking
 *  sort (descending) by bleu
 *  compare top X (hi) to middle Y (med) and low X (lo)
 *  cmp middle Y to low X
 */
inline void
MakePairs(vector<ScoredHyp>* s,
          vector<pair<ScoredHyp,ScoredHyp> >& training,
          bool misranked_only,
          float hi_lo)
{
  unsigned sz = s->size();
  if (sz < 2) return;
  sort(s->begin(), s->end(), CmpHypsByScore);
  unsigned sep = round(sz*hi_lo);
  // hi vs. med vs. low
  unsigned sep_hi = sep;
  if (sz > 4) while (sep_hi < sz && (*s)[sep_hi-1].score == (*s)[sep_hi].score) ++sep_hi;
  else sep_hi = 1;
  for (unsigned i = 0; i < sep_hi; i++) {
    for (unsigned j = sep_hi; j < sz; j++) {
      if (misranked_only && !((*s)[i].model <= (*s)[j].model)) continue;
      if ((*s)[i].score != (*s)[j].score)
        training.push_back(make_pair((*s)[i], (*s)[j]));
    }
  }
  // med vs. low
  unsigned sep_lo = sz-sep;
  while (sep_lo > 0 && (*s)[sep_lo-1].score == (*s)[sep_lo].score) --sep_lo;
  for (unsigned i = sep_hi; i < sep_lo; i++) {
    for (unsigned j = sep_lo; j < sz; j++) {
      if (misranked_only && !((*s)[i].model <= (*s)[j].model)) continue;
      if ((*s)[i].score != (*s)[j].score)
        training.push_back(make_pair((*s)[i], (*s)[j]));
    }
  }
}

} // namespace

#endif

