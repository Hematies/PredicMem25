#include once
#include "global.hpp"

template<typename pb_tag_t>
struct PrefetchBufferEntry {
	bool valid;
	pb_tag_t tag;
	PrefetchBufferEntry(){}
};

template<typename pb_tag_t>
struct PrefetchBufferEntriesMatrix {
		PrefetchBufferEntry<pb_tag_t> entries[PB_NUM_SETS];
		PrefetchBufferEntriesMatrix(){}
};

template<typename block_address_t, typename pb_index_t, typename pb_tag_t>
class PrefetchBuffer {
public:
    void operator()(PrefetchBufferEntry<pb_tag_t> entries[PB_NUM_SETS], block_address_t prefetchAddress, bool &issuePrefetch) {
        #pragma HLS INLINE

        #pragma HLS ARRAY_PARTITION variable=entries complete dim=0
        #pragma HLS DEPENDENCE array false variable=entries
        
        pb_index_t index = prefetchAddress % (1 << PB_NUM_SETS_LOG2);
        pb_tag_t tag = prefetchAddress >> PB_NUM_SETS_LOG2;

        if (entries[index].valid && entries[index].tag == tag) {
            issuePrefetch = false; // Already in the buffer, do not prefetch
        } else {
            issuePrefetch = true; // Not in the buffer, prefetch
            entries[index].tag = tag; // Update the buffer with the new tag
            entries[index].valid = true; // Mark the entry as valid
        }

    }
    
};