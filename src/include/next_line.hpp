#pragma once
#include "global.hpp"

#define NEXT_LINE_TYPES block_address_t

template<typename block_address_t>
class NextLinePrefetcher {
protected:

public:
	void operator()(block_address_t memoryAddress, block_address_t& addressToPrefetch){
#pragma HLS INLINE
		addressToPrefetch = memoryAddress + 1;
	}
};
