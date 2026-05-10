#include <algorithm>
#include <array>
#include <map>
#include <optional>

#include "../bop/bop.h"
#include "cache.h"
#include "msl/lru_table.h"


namespace knob
{
	vector<int32_t> bop_candidates = vector<int32_t>{1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8,9,-9,10,-10,11,-11,12,-12,13,-13,
    14,-14,15,-15,16,-16,18,-18,20,-20,24,-24,30,-30,32,-32,36,-36,40,-40};
	uint32_t bop_max_rounds = 100;
	uint32_t bop_max_score = 31;
	uint32_t bop_top_n = 1;
	bool     bop_enable_pref_buffer = false;
	uint32_t bop_pref_buffer_size = 256;
	uint32_t bop_pref_degree = 4;
	uint32_t bop_rr_size = 256;

}

namespace{
  std::map<CACHE*, shared_ptr<BOPrefetcher>> bops;
    
    
    /**
     * Function reads the lookahead entry and applies a prefetch.
    */
    void advance_lookahead(CACHE* cache, uint64_t addr, vector<int64_t> bestOffsets)
    {
      for(auto& offset : bestOffsets)
        cache->prefetch_line(((int64_t)addr) + offset, (cache->get_mshr_occupancy_ratio() < 0.5), 0);
    }
}


/**
 * Prefetcher's initialization, including the SVM4AP's model.
*/
void CACHE::prefetcher_initialize() {
  ::bops[this] = shared_ptr<BOPrefetcher>((BOPrefetcher*) new BOPrefetcher());
}

/**
 * Prefetcher's operation on each clock cycle. In the case of the GASP, if a prediction has been 
 * performed, it is prefetched from the next cache level.
*/
void CACHE::prefetcher_cycle_operate() { 
  // prefetcher.advance_lookahead(this); 
  }


/**
 * Prefetcher's operation called when a tag is checked in the cache. For the GASP, the IP and the queried address
 * are taken for the prediction and fitting of the SVM4AP model.
*/
uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, bool useful_prefetch, uint8_t type, uint32_t metadata_in)
{
  auto bestOffsets = vector<int64_t>();
  ::bops[this]->invoke_prefetcher(ip, addr, cache_hit, 0, bestOffsets);
  ::advance_lookahead(this, addr, bestOffsets);
  return metadata_in;
}


/**
 * Prefetcher's operation called when a miss is filled in the cache. Here nothing is done.
*/
uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)
{
  ::bops[this]->register_fill(addr);
  return metadata_in;
}


/**
 * Prefetcher's statistic metrics printing.
*/
void CACHE::prefetcher_final_stats() {
  
}
