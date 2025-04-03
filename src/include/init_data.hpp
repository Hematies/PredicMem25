#pragma once
#include "global.hpp"

template<typename delta_t, typename confidence_t>
constexpr DictionaryEntriesMatrix<delta_t, confidence_t> initDictionaryEntries(){
	DictionaryEntriesMatrix<delta_t, confidence_t> res;
	for(int c = 0; c < NUM_CLASSES; c++){
		res.entries[c] = {false, 0, DICTIONARY_LFU_INITIAL_CONFIDENCE};
	}
	return res;
}

template<typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
constexpr InputBufferEntriesMatrix<tag_t, block_address_t, class_t, confidence_t, lru_t>
	initInputBufferEntries(){
	InputBufferEntriesMatrix<tag_t, block_address_t, class_t, confidence_t, lru_t> res;
	for(int i = 0; i < IB_NUM_SETS; i++){
		for(int j = 0; j < IB_NUM_WAYS; j++){
			res.entries[i][j].valid = false;
			res.entries[i][j].tag = 0;
			res.entries[i][j].lastAddress = 0;
			res.entries[i][j].confidence = 0;
			res.entries[i][j].lastPredictedAddress = 0;
			res.entries[i][j].lruCounter = 0;
			res.entries[i][j].valid = 0;
			for(int k = 0; k < SEQUENCE_LENGTH; k++){
				res.entries[i][j].sequence[k] = NUM_CLASSES;
			}

		}
	}
	return res;
}


template<typename address_t, typename block_address_t, typename class_t, typename confidence_t>
constexpr ForwardingBufferEntriesMatrix<address_t, block_address_t, class_t, confidence_t>
	initForwardingBufferEntries(){
	ForwardingBufferEntriesMatrix<address_t, block_address_t, class_t, confidence_t> res;
	for(int i = 0; i < FORWARDING_DEPTH; i++){
		res.entries[i].valid = false;
		res.entries[i].inputBufferAddress = 0;
		res.entries[i].lastAddress = 0;
		res.entries[i].confidence = 0;
		res.entries[i].lastPredictedAddress = 0;
		for(int k = 0; k < SEQUENCE_LENGTH; k++){
			res.entries[i].sequence[k] = NUM_CLASSES;
		}

	}
	return res;
}


template<typename weight_t>
constexpr SVMWholeMatrix<weight_t> initSVMData(){
	SVMWholeMatrix<weight_t> res;
	for(int c = 0; c < NUM_CLASSES; c++){
		res.intercepts[c] = 0;
		for(int i = 0; i < SEQUENCE_LENGTH; i++){
			for(int j = 0; j < NUM_CLASSES_INCLUDING_NULL; j++){
				res.weightMatrices[c].weights[i][j] = 0;
			}
		}
	}
	return res;
}



