#ifndef _DTRAIN_UPDATE_H_
#define _DTRAIN_UPDATE_H_

namespace dtrain
{

bool
_cmp(ScoredHyp a, ScoredHyp b)
{
  return a.gold > b.gold;
}

/*
 * multipartite ranking
 *  sort (descending) by bleu
 *  compare top X (hi) to middle Y (med) and low X (lo)
 *  cmp middle Y to low X
 */
inline size_t
CollectUpdates(vector<ScoredHyp>* s,
               SparseVector<weight_t>& updates,
               float margin=0.)
{
  size_t num_up = 0;
  size_t sz = s->size();
  if (sz < 2) return 0;
  sort(s->begin(), s->end(), _cmp);
  size_t sep = round(sz*0.1);
  size_t sep_hi = sep;
  if (sz > 4) {
    while (sep_hi<sz && (*s)[sep_hi-1].gold==(*s)[sep_hi].gold)
      ++sep_hi;
  }
  else sep_hi = 1;
  for (size_t i = 0; i < sep_hi; i++) {
    for (size_t j = sep_hi; j < sz; j++) {
      if (((*s)[i].model-(*s)[j].model) > margin
          || (*s)[i].gold == (*s)[j].gold)
        continue;
      updates += (*s)[i].f-(*s)[j].f;
      num_up++;
    }
  }
  size_t sep_lo = sz-sep;
  while (sep_lo>=sep_hi && (*s)[sep_lo].gold==(*s)[sep_lo+1].gold)
    --sep_lo;
  for (size_t i = sep_hi; i < sep_lo; i++) {
    for (size_t j = sep_lo; j < sz; j++) {
      if (((*s)[i].model-(*s)[j].model) > margin
          || (*s)[i].gold == (*s)[j].gold)
        continue;
      updates += (*s)[i].f-(*s)[j].f;
      num_up++;
    }
  }

  return num_up;
}

} // namespace

#endif

