#include "global.hpp"


template<typename ib_confidence_t, typename prefetch_degree_t>
struct RecursivePrefetchLookupTable {
	prefetch_degree_t entries[MAX_PREDICTION_CONFIDENCE + 1];
	RecursivePrefetchLookupTable(){}
};


