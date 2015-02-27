#ifndef _DTRAIN_UPDATE_H_
#define _DTRAIN_UPDATE_H_

namespace dtrain
{

bool
_cmp(ScoredHyp a, ScoredHyp b)
{
  return a.gold > b.gold;
}

inline bool
_good(ScoredHyp& a, ScoredHyp& b, weight_t margin)
{
  if ((a.model-b.model)>margin
      || a.gold==b.gold)
    return true;

  return false;
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
               weight_t margin=0.)
{
  size_t num_up = 0;
  size_t sz = s->size();
  sort(s->begin(), s->end(), _cmp);
  size_t sep = round(sz*0.1);
  for (size_t i = 0; i < sep; i++) {
    for (size_t j = sep; j < sz; j++) {
      if (_good((*s)[i], (*s)[j], margin))
        continue;
      updates += (*s)[i].f-(*s)[j].f;
      num_up++;
    }
  }
  size_t sep_lo = sz-sep;
  for (size_t i = sep; i < sep_lo; i++) {
    for (size_t j = sep_lo; j < sz; j++) {
      if (_good((*s)[i], (*s)[j], margin))
        continue;
      updates += (*s)[i].f-(*s)[j].f;
      num_up++;
    }
  }

  return num_up;
}

} // namespace

#endif

