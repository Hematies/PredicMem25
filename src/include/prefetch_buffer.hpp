#pragma once
#include "global.hpp"

template<typename tag_t>
struct PrefetchBufferEntry {
	bool valid;
	tag_t tag;
	PrefetchBufferEntry(){}
};

template<typename tag_t>
struct PrefetchBufferEntriesMatrix {
		PrefetchBufferEntry<tag_t> entries[PB_NUM_SETS];
		PrefetchBufferEntriesMatrix(){}
};

template<typename block_address_t, typename index_t, typename tag_t>
class PrefetchBuffer {
public:
    void operator()(PrefetchBufferEntry<tag_t> entries[PB_NUM_SETS], block_address_t& prefetchAddress, bool &issuePrefetch) {
        #pragma HLS INLINE

        index_t index = prefetchAddress % (1 << PB_NUM_SETS_LOG2);
        tag_t tag = prefetchAddress >> PB_NUM_SETS_LOG2;

        PrefetchBufferEntry<tag_t> entry = entries[index];

        issuePrefetch = !(entry.valid && entry.tag == tag); // If already in the buffer, do not prefetch
        entry.tag = tag; // Update the buffer with the new tag
		entry.valid = true; // Mark the entry as valid
        entries[index] = entry;

    }
    
};
