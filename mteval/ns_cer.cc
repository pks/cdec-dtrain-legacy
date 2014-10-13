#include "ns_cer.h"
#include "tdict.h"
#include "levenshtein.h"

static const unsigned kNUMFIELDS = 2;
static const unsigned kEDITDISTANCE = 0;
static const unsigned kCHARCOUNT = 1;

bool CERMetric::IsErrorMetric() const {
  return true;
}

unsigned CERMetric::SufficientStatisticsVectorSize() const {
  return 2;
}

void CERMetric::ComputeSufficientStatistics(const std::vector<WordID>& hyp,
                                            const std::vector<std::vector<WordID> >& refs,
                                            SufficientStats* out) const {
  out->fields.resize(kNUMFIELDS);
  std::string hyp_str(TD::GetString(hyp));
  float best_score = hyp_str.size();
  for (size_t i = 0; i < refs.size(); ++i) {
    std::string ref_str(TD::GetString(refs[i]));
    float score = cdec::LevenshteinDistance(hyp_str, ref_str);
    if (score < best_score) {
      out->fields[kEDITDISTANCE] = score;
      out->fields[kCHARCOUNT] = ref_str.size();
      best_score = score;
    }
  }
}

float CERMetric::ComputeScore(const SufficientStats& stats) const {
  return stats.fields[kEDITDISTANCE] / stats.fields[kCHARCOUNT];
}

