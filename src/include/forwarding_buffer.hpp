#pragma once
#include "global.hpp"

template<typename address_t, typename block_address_t, typename class_t, typename confidence_t>
struct ForwardingBufferEntry {
	bool valid;
	address_t inputBufferAddress;
	block_address_t lastAddress;
	class_t sequence[SEQUENCE_LENGTH];
	confidence_t confidence;
	block_address_t lastPredictedAddress;
	ForwardingBufferEntry(){}
};


template<typename address_t, typename block_address_t, typename class_t, typename confidence_t>
struct ForwardingBufferEntriesMatrix {
	ForwardingBufferEntry<address_t, block_address_t, class_t, confidence_t> entries[FORWARDING_DEPTH];
	ForwardingBufferEntriesMatrix(){}
};


template<typename address_t, typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
class ForwardingBuffer {
public:
	ForwardingBufferEntry<address_t, block_address_t, class_t, confidence_t> read(
		ForwardingBufferEntry<address_t, block_address_t, class_t, confidence_t> entries[FORWARDING_DEPTH],
		address_t& address, bool& isHit){
		#pragma HLS INLINE

		ForwardingBufferEntry<address_t, block_address_t, class_t, confidence_t> res;
		isHit = false;

		for(int i = 0; i < FORWARDING_DEPTH; i++){
			#pragma HLS UNROLL
			if(entries[i].valid && entries[i].inputBufferAddress == address){
				isHit = true;
				res = entries[i];
				break;
			}
		}

		return res;
	}

	void write(
		ForwardingBufferEntry<address_t, block_address_t, class_t, confidence_t> entries[FORWARDING_DEPTH],
		InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t>& inputBufferEntry,
		address_t& address, forwarding_index_t& nextSlot{
		#pragma HLS INLINE
		
		forwarding_index_t nextSlot_ = nextSlot; 
		nextSlot = nextSlot == (FORWARDING_DEPTH - 1)? 0 : nextSlot + 1;
		ForwardingBufferEntry<address_t, block_address_t, class_t, confidence_t> newEntry;
		newEntry.valid = true;
		newEntry.inputBufferAddress = address;
		newEntry.lastAddress = inputBufferEntry.lastAddress;
		for(int k = 0; k < SEQUENCE_LENGTH; k++){
			#pragma HLS UNROLL
			newEntry.sequence[k] = inputBufferEntry.sequence[k];
		}
		newEntry.confidence = inputBufferEntry.confidence;
		newEntry.lastPredictedAddress = inputBufferEntry.lastPredictedAddress;
		entries[nextSlot_] = newEntry;
		
	}

};

