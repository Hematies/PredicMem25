#pragma once
#include "global.hpp"

#define STRIDE_TYPES address_t, ib_index_t, ib_way_t, ib_tag_t, block_address_t, ib_confidence_t, ib_lru_t, delta_t

template<typename address_t, typename ib_index_t, typename ib_way_t, typename ib_tag_t, typename block_address_t,
typename ib_confidence_t, typename ib_lru_t, typename delta_t>
class StridePrefetcher {
protected:

public:
void operator()(address_t strideTableAddress, block_address_t memoryAddress, block_address_t& addressToPrefetch){
	static StrideTableEntriesMatrix<ib_tag_t, block_address_t, delta_t, ib_lru_t>
		strideTableEntriesMatrix = initStrideTableEntries<ib_tag_t, block_address_t, delta_t, ib_lru_t>();
#pragma HLS ARRAY_RESHAPE variable=strideTableEntriesMatrix.entries dim=2 complete
#pragma HLS ARRAY_RESHAPE variable=strideTableEntriesMatrix.entries dim=3 complete
#pragma HLS BIND_STORAGE variable=strideTableEntriesMatrix.entries type=RAM_T2P impl=bram latency=1

#pragma HLS DEPENDENCE array false variable=strideTableEntriesMatrix.entries
	static StrideTable<address_t, ib_index_t, ib_way_t, ib_tag_t, block_address_t, delta_t, ib_lru_t> strideTable;
#pragma HLS DEPENDENCE false variable=strideTable


#pragma HLS PIPELINE
	addressToPrefetch = 0;

	ib_index_t index;
	ib_way_t way;
	bool isStrideTableHit;
	StrideTableEntry<ib_tag_t, block_address_t, delta_t, ib_lru_t> strideTableEntryDummy;
	StrideTableEntry<ib_tag_t, block_address_t, delta_t, ib_lru_t> strideTableEntry =
		strideTable(strideTableEntriesMatrix.entries, strideTableAddress, strideTableEntryDummy, true, isStrideTableHit,
		index, way);

	constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
	ib_tag_t tag = strideTableAddress >> numIndexBits;

	delta_t predictedDelta = 0;
	if (strideTableEntry.lastAddress != memoryAddress){
		if(isStrideTableHit) {
			predictedDelta = (delta_t)memoryAddress - (delta_t)strideTableEntry.lastAddress;
			addressToPrefetch = ((delta_t)memoryAddress + predictedDelta);
		}

		strideTableEntry.tag = tag;
		strideTableEntry.valid = true;
		strideTableEntry.lastAddress = memoryAddress;
		strideTableEntry.delta = predictedDelta;
		bool isStrideTableHitDummy;
		strideTable(strideTableEntriesMatrix.entries, strideTableAddress, strideTableEntry, false, isStrideTableHitDummy,
				index, way);
	}
}
};
