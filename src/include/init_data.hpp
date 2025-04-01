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
				res.entries[i][j].sequence[k] = NUM_CLASSES_INCLUDING_NULL; // NUM_CLASSES
			}

		}
	}
	return res;
}

template<typename tag_t, typename block_address_t, typename class_t, typename burst_length_t, typename confidence_t, typename lru_t>
constexpr BurstInputBufferEntriesMatrix<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t>
	initBurstInputBufferEntries(){
	BurstInputBufferEntriesMatrix<tag_t, block_address_t, class_t, burst_length_t, confidence_t, lru_t> res;
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
				res.entries[i][j].sequence[k] = NUM_CLASSES_INCLUDING_NULL;
			}
			for(int k = 0; k < SEQUENCE_LENGTH; k++){
				res.entries[i][j].burstLengthSequence[k] = NUM_CLASSES_INCLUDING_NULL;
			}
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

template<typename weight_t>
constexpr BurstSVMWholeMatrix<weight_t> initBurstSVMData(){
	BurstSVMWholeMatrix<weight_t> res;
	for(int c = 0; c < NUM_BURST_CLASSES; c++){
		res.intercepts[c] = 0;
		for(int i = 0; i < SEQUENCE_LENGTH; i++){
			for(int j = 0; j < NUM_BURST_CLASSES_INCLUDING_NULL; j++){
				res.weightMatrices[c].weights[i][j] = 0;
			}
		}
	}
	return res;
}



