#pragma once
#include "global.hpp"

template<typename tag_t, typename block_address_t, typename class_t, typename burst_length_t, typename confidence_t, typename lru_t>
struct BurstInputBufferEntry {
	bool valid;
	tag_t tag;
	block_address_t lastAddress;
	class_t sequence[SEQUENCE_LENGTH];
	class_t burstLengthSequence[SEQUENCE_LENGTH];
	confidence_t confidence;
	block_address_t lastPredictedAddress;
	burst_length_t lastPredictedBurstLength;
	lru_t lruCounter;
	BurstInputBufferEntry(){}
};


template<typename tag_t, typename block_address_t, typename class_t, typename burst_length_t, typename confidence_t, typename lru_t>
struct BurstInputBufferEntriesMatrix {
	BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS];
	BurstInputBufferEntriesMatrix(){}
};


template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename burst_length_t,
	typename confidence_t, typename lru_t>
class BurstInputBuffer {
protected:

	void updateLRU(BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> set[IB_NUM_WAYS], index_t index, way_t way);
	way_t getLeastRecentWay(BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> set[IB_NUM_WAYS], index_t index);
	way_t queryWay(BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> set[IB_NUM_WAYS], index_t index, tag_t tag);

public:
	BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> read(
		BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
		address_t address, bool& isHit);
	// BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t>
	void write(
		BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
		address_t address, BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> entry);

	BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> operator()(
			BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
			address_t address, BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> entry, bool performRead, bool& isHit);

};

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename burst_length_t, typename confidence_t, typename lru_t>
void BurstInputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t>
	::updateLRU(BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> set[IB_NUM_WAYS], index_t index, way_t way)
{
#pragma HLS INLINE
	bool areAllWaysSaturated = true;
	for (int w = 0; w < IB_NUM_WAYS; w++) {
#pragma HLS UNROLL
		areAllWaysSaturated = areAllWaysSaturated && (set[way].lruCounter == IB_MAX_LRU_COUNTER);
	}

	lru_t lruCounters[IB_NUM_WAYS];

	for (int w = 0; w < IB_NUM_WAYS; w++) {
#pragma HLS UNROLL
		bool increment = w == way;
		bool reset = areAllWaysSaturated;
		lru_t lruCounter = set[way].lruCounter;
		bool isSaturated = lruCounter == IB_MAX_LRU_COUNTER;

		if (reset) {
			if (!increment) lruCounter = 0;
			else lruCounter = 1;
		}
		else {
			if (increment && !isSaturated) lruCounter++;
		}
		lruCounters[w] = lruCounter;

	}

	for (int w = 0; w < IB_NUM_WAYS; w++) {
#pragma HLS UNROLL
		set[w].lruCounter = lruCounters[w];
	}
}

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename burst_length_t, typename confidence_t, typename lru_t>
way_t BurstInputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t>
::getLeastRecentWay(BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> set[IB_NUM_WAYS], index_t index)
{
#pragma HLS INLINE
	way_t res = 0;
	lru_t leastRecency = IB_MAX_LRU_COUNTER;
	for (int w = 0; w < IB_NUM_WAYS; w++) {
#pragma HLS UNROLL
		if (set[w].lruCounter < leastRecency) {
			res = w;
			leastRecency = set[w].lruCounter;
		}
	}
	return res;
}

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename burst_length_t, typename confidence_t, typename lru_t>
way_t BurstInputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t>::queryWay(
		BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> set[IB_NUM_WAYS],
		index_t index, tag_t tag)
{
	#pragma HLS INLINE
	way_t res = IB_NUM_WAYS;

	for (int w = 0; w < IB_NUM_WAYS; w++) {
#pragma HLS UNROLL
		if ((set[w].tag == tag)  && (set[w].valid)
				){
			res = w;
			// break;
		}

	}
	return res;
}

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename burst_length_t, typename confidence_t, typename lru_t>
BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> BurstInputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t>::
read(BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS], address_t BurstInputBufferAddress, bool& isHit) {
#pragma HLS INLINE
// #pragma HLS PIPELINE
#pragma HLS ARRAY_RESHAPE variable=entries dim=2 complete
#pragma HLS BIND_STORAGE variable=entries type=RAM_T2P impl=bram latency=1

	BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> res = 
		BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t>();

	constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
	tag_t tag = BurstInputBufferAddress >> numIndexBits;
	index_t index = BurstInputBufferAddress % (1 << numIndexBits);
	isHit = false;


	BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> set[IB_NUM_WAYS];
	for(int w = 0; w < IB_NUM_WAYS; w++){
#pragma HLS UNROLL
		set[w] = entries[index][w];
	}


	way_t way = this->queryWay(set, index, tag);


	if (way != (way_t)IB_NUM_WAYS) {
		res = set[way];
		isHit = true;
	}
	// this->updateLRU(entries, index, way);


	return res;
}


template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename burst_length_t, typename confidence_t, typename lru_t>
// BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t>
void
BurstInputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t>::
write(BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
	address_t BurstInputBufferAddress, BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> entry) {
#pragma HLS INLINE
// #pragma HLS PIPELINE

// #pragma HLS DEPENDENCE array false RAW inter variable=entries

#pragma HLS ARRAY_RESHAPE variable=entries dim=2 complete
#pragma HLS BIND_STORAGE variable=entries type=RAM_T2P impl=bram latency=1

// #pragma HLS ARRAY_PARTITION variable=entries dim=0 complete
// #pragma HLS ARRAY_RESHAPE dim=2 factor=2 object type=block variable=entries

	constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
	tag_t tag = BurstInputBufferAddress >> numIndexBits;
	index_t index = BurstInputBufferAddress % (1 << numIndexBits);

	/*
	BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> set[IB_NUM_WAYS];
	for(int w = 0; w < IB_NUM_WAYS; w++){
	#pragma HLS UNROLL
		set[w] = entries[index][w];
	}
	*/


	way_t leastRecentWay = this->getLeastRecentWay(entries, index);
	way_t way = this->queryWay(entries, index, tag);


	if (way == IB_NUM_WAYS) {
		way = leastRecentWay;
		// entries[index][way].lruCounter = 1;
		// entries[index][way].tag = tag;
		entry.lruCounter = 1;
		entry.tag = tag;
	}

	// entries[index][way].lastAddress = entry.lastAddress;
	// entries[index][way].confidence = entry.confidence;
	// entries[index][way].lastPredictedAddress = entry.lastPredictedAddress;

	// entries[index][way].valid = true;
	entry.valid = true;

	entries[index][way] = entry;
	/*
	for(int i = 0; i < SEQUENCE_LENGTH; i++){
#pragma HLS UNROLL
		entries[index][way].sequence[i] = entry.sequence[i];
	}
	*/
	this->updateLRU(entries, index, way);

}


template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename burst_length_t, typename confidence_t, typename lru_t>
BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t>
BurstInputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t>::
operator()(
			BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
			address_t address, BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> entry, bool performRead, bool& isHit){

#pragma HLS INLINE
// #pragma HLS PIPELINE

// #pragma HLS DEPENDENCE array false RAW inter variable=entries

#pragma HLS ARRAY_RESHAPE variable=entries dim=2 complete
#pragma HLS ARRAY_RESHAPE variable=entries dim=3 complete
// #pragma HLS ARRAY_PARTITION variable=entries dim=0 complete
#pragma HLS BIND_STORAGE variable=entries type=RAM_T2P impl=bram latency=1

	BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> res =
			BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t>();

// #pragma HLS ARRAY_PARTITION variable=entries dim=0 complete
// #pragma HLS ARRAY_RESHAPE dim=2 factor=2 object type=block variable=entries

	constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
	tag_t tag = address >> numIndexBits;
	index_t index = address % (1 << numIndexBits);


	BurstInputBufferEntry<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> set[IB_NUM_WAYS];
#pragma HLS ARRAY_PARTITION variable=set complete dim=0
#pragma HLS DEPENDENCE array false variable=set
	for(int w = 0; w < IB_NUM_WAYS; w++){
	#pragma HLS UNROLL
		set[w] = entries[index][w];
	}



	way_t leastRecentWay = this->getLeastRecentWay(set, index);
	way_t way = this->queryWay(set, index, tag);

	isHit = way != IB_NUM_WAYS;

	if(!performRead){

		if (!isHit) {
			way = leastRecentWay;
			// entries[index][way].lruCounter = 1;
			// entries[index][way].tag = tag;
			entry.lruCounter = 1;
			entry.tag = tag;
		}

		// entries[index][way].lastAddress = entry.lastAddress;
		// entries[index][way].confidence = entry.confidence;
		// entries[index][way].lastPredictedAddress = entry.lastPredictedAddress;

		// entries[index][way].valid = true;
		entry.valid = true;

		// entries[index][way] = entry;
		set[way] = entry;
		res = entry;

		this->updateLRU(set, index, way);

	}
	else{
		if(isHit){
			res = set[way];
		}
	}

	for(int w = 0; w < IB_NUM_WAYS; w++){
		#pragma HLS UNROLL
		entries[index][w] = set[w];
		}

	return res;
}

