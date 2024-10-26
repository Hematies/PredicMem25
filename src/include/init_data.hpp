#pragma once
#include "global.hpp"

template<typename delta_t, typename confidence_t>
constexpr void initDictionaryEntries(DictionaryEntry<delta_t, dic_confidence_t> dictionaryEntries[NUM_CLASSES]){
	for(int c = 0; c < NUM_CLASSES; c++){
		dictionaryEntries[c] = {false, 0, DICTIONARY_LFU_INITIAL_CONFIDENCE};
	}
}

template<typename tag_t, typename block_address_t, typename class_t, typename confidence_t, typename lru_t>
constexpr void initInputBufferEntries(InputBufferEntry<tag_t, block_address_t, class_t, confidence_t, lru_t>
	inputBufferEntries[IB_NUM_SETS][IB_NUM_WAYS]){
	for(int i = 0; i < IB_NUM_SETS; i++){
		for(int j = 0; j < IB_NUM_WAYS; j++){
			inputBufferEntries[i][j].valid = false;
			inputBufferEntries[i][j].tag = 0;
			inputBufferEntries[i][j].lastAddress = 0;
			inputBufferEntries[i][j].confidence = 0;
			inputBufferEntries[i][j].lastPredictedAddress = 0;
			inputBufferEntries[i][j].lruCounter = 0;
			inputBufferEntries[i][j].valid = 0;
			for(int k = 0; k < SEQUENCE_LENGTH; k++){
				inputBufferEntries[i][j].sequence[k] = NUM_CLASSES;
			}

		}
	}
}

template<typename weight_t>
constexpr void initSVMWeights(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES]){
	for(int c = 0; c < NUM_CLASSES; c++){
		for(int i = 0; i < SEQUENCE_LENGTH; i++){
			for(int j = 0; j < NUM_CLASSES_INCLUDING_NULL; j++){
				weight_matrices[c].weights[i][j] = 0;
			}
		}
	}
}

template<typename weight_t>
constexpr void initSVMIntercepts(weight_t intercepts[NUM_CLASSES]){
	for(int c = 0; c < NUM_CLASSES; c++){
		intercepts[c] = 0;
	}
}




