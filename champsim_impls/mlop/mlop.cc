#include <algorithm>
#include <array>
#include <map>
#include <optional>

#include "../mlop/mlop.h"
#include "cache.h"
#include "msl/lru_table.h"


namespace knob
{
  uint32_t mlop_pref_degree = 16;
	uint32_t mlop_num_updates = 500;
	float mlop_l1d_thresh = 2.0;
	float mlop_l2c_thresh = 0.75;
	float mlop_llc_thresh = 0.3;
	uint32_t mlop_debug_level = 0;

}


namespace{
    std::map<CACHE*, shared_ptr<MLOP>> mlops;
    
    
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
  ::mlops[this] = shared_ptr<MLOP>((MLOP*) new MLOP(this));
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
  ::mlops[this]->invoke_prefetcher(ip, addr, cache_hit, 0, bestOffsets);
  ::advance_lookahead(this, addr, bestOffsets);
  return metadata_in;
}


/**
 * Prefetcher's operation called when a miss is filled in the cache. Here nothing is done.
*/
uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)
{
  ::mlops[this]->register_fill(addr, set, way, prefetch, evicted_addr);
  return metadata_in;
}


/**
 * Prefetcher's statistic metrics printing.
*/
void CACHE::prefetcher_final_stats() {
  
}
