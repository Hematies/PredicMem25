#pragma once
#include "global.hpp"


class GASP {
protected:
	InputBuffer<address_t, ib_index_t, ib_way_t, ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> inputBuffer;
	Dictionary<dic_index_t, delta_t, dic_confidence_t> dictionary;
	SVM<svm_weight_t, class_t, svm_distance_t> svm;

	LookUpTable<ib_confidence_t, MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_THRESHOLD + 1> confidenceLookUpTable;
	
	void succesivePrediction(block_address_t baseAddress, class_t baseSequence[SEQUENCE_LENGTH],
		int numPredictions, block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE]);

public:

	GASP(
		InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> inputBufferEntries[IB_NUM_SETS][IB_NUM_WAYS],
		DictionaryEntry<delta_t, dic_confidence_t> dictionaryEntries[NUM_CLASSES],
		svm_weight_t svmWeights[NUM_CLASSES][SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL], svm_weight_t svmIntercepts[NUM_CLASSES]
	) : 
		inputBuffer(InputBuffer<address_t, ib_tag_t, ib_way_t, ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t>(inputBufferEntries)),
		dictionary(Dictionary<dic_index_t, delta_t, dic_confidence_t>(dictionaryEntries)),
		svm(SVM<svm_weight_t, class_t, svm_distance_t>(svmWeights, svmIntercepts)),
		confidenceLookUpTable(
			fillUniformLookUpTable<ib_confidence_t, MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_THRESHOLD + 1>(1, MAX_PREFETCHING_DEGREE)) {}

	int operator()(address_t programCounter, block_address_t memoryAddress, block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE]);
};

int GASP::operator()(address_t inputBufferAddress, block_address_t memoryAddress, block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE]) {
	int prefetchDegree = 0;

	// 1) Input buffer is read:
	InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> emptyEntry = 
		InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t>();
	InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> inputBufferEntry =
		this->inputBuffer(true, inputBufferAddress, emptyEntry);

	constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
	ib_tag_t tag = memoryAddress >> numIndexBits;

	// Skip operation if the previous access is equal to the current:
	if (inputBufferEntry.lastAddress != memoryAddress) {
		class_t predictedClass;
		bool performPrefetch = false;

		// Continue if there has been a hit:
		if (inputBufferEntry.valid) {

			// 2) If the predictedAddress is equal to the current, increment the confidence (decrease otherwise):
			if (inputBufferEntry.lastPredictedAddress == memoryAddress) {
				if (inputBufferEntry.confidence >= (MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_INCREASE))
					inputBufferEntry.confidence = MAX_PREDICTION_CONFIDENCE;
				else
					inputBufferEntry.confidence += PREDICTION_CONFIDENCE_INCREASE;
			}
			else {
				if (inputBufferEntry.confidence <= (PREDICTION_CONFIDENCE_DECREASE))
					inputBufferEntry.confidence = 0;
				else
					inputBufferEntry.confidence += PREDICTION_CONFIDENCE_DECREASE;
			}

			if (inputBufferEntry.confidence >= PREDICTION_CONFIDENCE_THRESHOLD)
				performPrefetch = true;

			// 3) Compute the resulting delta and its class:
			delta_t delta = (delta_t)memoryAddress - (delta_t)inputBufferEntry.lastAddress;
			class_t dictionaryClass;
			DictionaryEntry < delta_t, dic_confidence_t > dictionaryEntry = this->dictionary(false, false, 0, delta, dictionaryClass);

			// 4) Fit-then-predict with the SVM:
			predictedClass = this->svm(false, inputBufferEntry.sequence, dictionaryClass);

			for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
				inputBufferEntry.sequence[i] = inputBufferEntry.sequence[i + 1];
			}
			inputBufferEntry.sequence[SEQUENCE_LENGTH - 1] = predictedClass;

		}
		// If there has been a miss, prepare a blank new input buffer entry:
		else {
			inputBufferEntry.tag = tag;
			inputBufferEntry.valid = true;
			inputBufferEntry.lastAddress = memoryAddress;
			inputBufferEntry.lastPredictedAddress = 0;
			inputBufferEntry.lruCounter = 1;
			inputBufferEntry.confidence = 0;
			for (int i = 0; i < SEQUENCE_LENGTH; i++) {
				inputBufferEntry.sequence[i] = NUM_CLASSES;
			}

			// 4) Predict with the SVM:
			predictedClass = this->svm(true, inputBufferEntry.sequence, 0);

		}

		// 5) Get the finally predicted address:
		dic_index_t dummyIndex;
		delta_t predictedDelta = this->dictionary(true, true, predictedClass, 0, dummyIndex).delta;
		block_address_t predictedAddress = ((delta_t)memoryAddress + predictedDelta);

		if (performPrefetch) {
			addressesToPrefetch[prefetchDegree] = predictedAddress;
			
			// 6) Apply recursive/successive prefetching on the calculated prefetching degree (>= 1):
			prefetchDegree = this->confidenceLookUpTable.table[inputBufferEntry.confidence - PREDICTION_CONFIDENCE_THRESHOLD];
			this->succesivePrediction(predictedAddress, inputBufferEntry.sequence, prefetchDegree, addressesToPrefetch);
		}

		// 7) Update the input buffer with the entry:
		inputBufferEntry.lastAddress = memoryAddress;
		inputBufferEntry.lastPredictedAddress = predictedAddress;
		inputBuffer(false, inputBufferAddress, inputBufferEntry);
	}

	return prefetchDegree;
}


void GASP::succesivePrediction(block_address_t baseAddress, class_t baseSequence[SEQUENCE_LENGTH], 
	int numPredictions, block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE]) {
	class_t sequence[SEQUENCE_LENGTH];
	block_address_t memoryAddress = baseAddress;

	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
		sequence[i] = baseSequence[i];
	}
	for (int k = 1; k < numPredictions; k++) {
		class_t predictedClass = this->svm(true, sequence, 0);

		dic_index_t dummyIndex;
		delta_t predictedDelta = this->dictionary(true, true, predictedClass, 0, dummyIndex).delta;
		memoryAddress = ((delta_t)memoryAddress + predictedDelta);

		addressesToPrefetch[k] = memoryAddress;

		for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
			sequence[i] = sequence[i + 1];
		}
		sequence[SEQUENCE_LENGTH - 1] = predictedClass;

	}
}