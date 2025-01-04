#pragma once
#include "global.hpp"

template<typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
struct InputBufferEntry {
	bool valid;
	tag_t tag;
	block_address_t lastAddress;
	class_t sequence[SEQUENCE_LENGTH];
	confidence_t confidence;
	block_address_t lastPredictedAddress;
	lru_t lruCounter;
	InputBufferEntry(){}
};


template<typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
struct InputBufferEntriesMatrix {
		InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS];
		InputBufferEntriesMatrix(){}
};


template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
class InputBuffer {
protected:

	void updateLRU(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> set[IB_NUM_WAYS], index_t index, way_t way);
	way_t getLeastRecentWay(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> set[IB_NUM_WAYS], index_t index);
	way_t queryWay(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> set[IB_NUM_WAYS], index_t index, tag_t tag);

public:
	InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> read(
		InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
		address_t address, bool& isHit);
	// InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t>
	void write(
		InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
		address_t address, InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entry);

	InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> operator()(
			InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
			address_t address, InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entry, bool performRead, bool& isHit);

};

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
void InputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, confidence_t, lru_t>
	::updateLRU(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> set[IB_NUM_WAYS], index_t index, way_t way)
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

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
way_t InputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, confidence_t, lru_t>
::getLeastRecentWay(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> set[IB_NUM_WAYS], index_t index)
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

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
way_t InputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, confidence_t, lru_t>::queryWay(
		InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> set[IB_NUM_WAYS],
		index_t index, tag_t tag)
{
	#pragma HLS INLINE
	way_t res = IB_NUM_WAYS;

	for (int w = 0; w < IB_NUM_WAYS; w++) {
#pragma HLS UNROLL
		if ((set[w].tag == tag) //  && (entries[index][w].valid)
				){
			res = w;
			// break;
		}

	}
	return res;
}

template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> InputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, confidence_t, lru_t>::
read(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS], address_t inputBufferAddress, bool& isHit) {
#pragma HLS INLINE
// #pragma HLS PIPELINE
#pragma HLS ARRAY_RESHAPE variable=entries dim=2 complete
#pragma HLS BIND_STORAGE variable=entries type=RAM_T2P impl=bram latency=1

	InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> res = 
		InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t>();

	constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
	tag_t tag = inputBufferAddress >> numIndexBits;
	index_t index = inputBufferAddress % (1 << numIndexBits);
	isHit = false;


	InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> set[IB_NUM_WAYS];
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


template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
// InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t>
void
InputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, confidence_t, lru_t>::
write(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
	address_t inputBufferAddress, InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entry) {
#pragma HLS INLINE
// #pragma HLS PIPELINE

// #pragma HLS DEPENDENCE array false RAW inter variable=entries

#pragma HLS ARRAY_RESHAPE variable=entries dim=2 complete
#pragma HLS BIND_STORAGE variable=entries type=RAM_T2P impl=bram latency=1

// #pragma HLS ARRAY_PARTITION variable=entries dim=0 complete
// #pragma HLS ARRAY_RESHAPE dim=2 factor=2 object type=block variable=entries

	constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
	tag_t tag = inputBufferAddress >> numIndexBits;
	index_t index = inputBufferAddress % (1 << numIndexBits);

	/*
	InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> set[IB_NUM_WAYS];
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


template<typename address_t, typename index_t, typename way_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t>
InputBuffer<address_t, index_t, way_t, tag_t, block_address_t, class_t, confidence_t, lru_t>::
operator()(
			InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entries[IB_NUM_SETS][IB_NUM_WAYS],
			address_t address, InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> entry, bool performRead, bool& isHit){

#pragma HLS INLINE
// #pragma HLS PIPELINE

// #pragma HLS DEPENDENCE array false RAW inter variable=entries

#pragma HLS ARRAY_RESHAPE variable=entries dim=2 complete
#pragma HLS ARRAY_RESHAPE variable=entries dim=3 complete
// #pragma HLS ARRAY_PARTITION variable=entries dim=0 complete
#pragma HLS BIND_STORAGE variable=entries type=RAM_T2P impl=bram latency=1

	InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> res =
			InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t>();

// #pragma HLS ARRAY_PARTITION variable=entries dim=0 complete
// #pragma HLS ARRAY_RESHAPE dim=2 factor=2 object type=block variable=entries

	constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
	tag_t tag = address >> numIndexBits;
	index_t index = address % (1 << numIndexBits);


	InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t> set[IB_NUM_WAYS];
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

