#include <iostream>
#include "../include/global.hpp"

DictionaryEntry<delta_t, dic_confidence_t> testDictionary(dic_index_t index){
// #pragma HLS TOP
// #pragma HLS TOP

	static DictionaryEntry<delta_t, dic_confidence_t> dictionaryEntries[NUM_CLASSES];
#pragma HLS ARRAY_PARTITION variable=dictionaryEntries dim=1 complete

	static Dictionary<dic_index_t, delta_t, dic_confidence_t> dictionary;

	dic_index_t resultingIndex = 0;
	DictionaryEntry<delta_t, dic_confidence_t> res = dictionary.write(dictionaryEntries, -2, resultingIndex);
	res = dictionary.read(dictionaryEntries, true, 0, 0, resultIndex, true);
	res = dictionary.read(dictionaryEntries, false, 0, -5, resultIndex, true);
	return res;
}

InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> testInputBuffer(address_t addr){
// #pragma HLS TOP

	static InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t>
		entries[IB_NUM_SETS][IB_NUM_WAYS];
#pragma HLS ARRAY_PARTITION variable=entries dim=0 factor=2 block

	static InputBuffer<address_t, ib_index_t, ib_way_t, ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t>
		inputBuffer;

	InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> res = {
			true, 0, 1, {1,1,1,1,1,1,1,1}, 0, 2, 0
	};
	InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> res1 = {
			true, 20, 1, {1,1,1,1,1,1,1,1}, 0, 2, 0
	};
	inputBuffer.write(entries, 0, res);
	inputBuffer.write(entries, 1 << 30, res1);
	inputBuffer.write(entries, 2, res);
	inputBuffer.write(entries, 3, res);
	res = inputBuffer.read(entries, 0);
	res = inputBuffer.read(entries, 1);

	return res;
}


void testSVM(class_t input[SEQUENCE_LENGTH], class_t target, class_t output[MAX_PREFETCHING_DEGREE]){
// #pragma HLS TOP
// #pragma PIPELINE
	// static class_t input[SEQUENCE_LENGTH];
	#pragma HLS ARRAY_PARTITION variable=input complete

	static weigth_matrix_t<svm_weight_t> weight_matrices[NUM_CLASSES];
	// #pragma HLS SHARED variable=weight_matrices->weights
	#pragma HLS ARRAY_PARTITION variable=weight_matrices complete

	static svm_weight_t intercepts[NUM_CLASSES];
	// #pragma HLS SHARED variable=intercepts
	#pragma HLS ARRAY_PARTITION variable=intercepts complete

	class_t newInput[SEQUENCE_LENGTH];
	#pragma HLS ARRAY_PARTITION variable=newInput complete


	static SVM<svm_weight_t, class_t, svm_distance_t> svm;

// #pragma HLS DATAFLOW
	/*
	class_t res = svm.fitAndPredict(weight_matrices, intercepts, input, 0);
	for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
	#pragma HLS UNROLL
		newInput[i] = input[i + 1];
	}
	newInput[SEQUENCE_LENGTH - 1] = res;

	class_t res1 = svm.predict(weight_matrices, intercepts, newInput);
	return res1;
	*/
	svm.fitAndRecursivelyPredict(weight_matrices, intercepts, input, target, output, 1);
	// class_t res = svm.fitAndPredict(weight_matrices, intercepts, input, target);
	// output[0] = res;
}


int testGASP(address_t inputBufferAddress, block_address_t memoryAddress, block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE]){
// #pragma HLS INTERFACE ap_ctrl_chain port=return
#pragma HLS TOP

	static LookUpTable<ib_confidence_t, MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_THRESHOLD + 1>
		confidenceLookUpTable = fillUniformLookUpTable<ib_confidence_t, MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_THRESHOLD + 1>(1, MAX_PREFETCHING_DEGREE);

	static DictionaryEntry<delta_t, dic_confidence_t> dictionaryEntries[NUM_CLASSES];
#pragma HLS ARRAY_PARTITION variable=dictionaryEntries complete
// #pragma HLS DEPENDENCE array false inter variable=dictionaryEntries

	static InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t>
			inputBufferEntries[IB_NUM_SETS][IB_NUM_WAYS];
// #pragma HLS SHARED variable=inputBufferEntries
	#pragma HLS ARRAY_PARTITION variable=inputBufferEntries dim=0 factor=2 block
	#pragma HLS DEPENDENCE array false inter variable=inputBufferEntries

	static weigth_matrix_t<svm_weight_t> weight_matrices[NUM_CLASSES];
// #pragma HLS DEPENDENCE array false inter variable=weight_matrices
#pragma HLS ARRAY_PARTITION variable=weight_matrices complete
#pragma HLS ARRAY_PARTITION variable=weight_matrices->weights complete

	static svm_weight_t intercepts[NUM_CLASSES];
// #pragma HLS DEPENDENCE array false inter variable=intercepts
#pragma HLS ARRAY_PARTITION variable=intercepts complete

	static InputBuffer<address_t, ib_index_t, ib_way_t, ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> inputBuffer;
#pragma HLS DEPENDENCE false variable=inputBuffer

	static Dictionary<dic_index_t, delta_t, dic_confidence_t> dictionary;
#pragma HLS DEPENDENCE false variable=dictionary

	static SVM<svm_weight_t, class_t, svm_distance_t> svm;
#pragma HLS DEPENDENCE false variable=svm

	/*
	static GASP gasp = GASP();
	int res = gasp(inputBufferEntries, dictionaryEntries, weight_matrices, intercepts, inputBufferAddress, memoryAddress, addressesToPrefetch);
	return res;
	*/


#pragma HLS PIPELINE
	int prefetchDegree = 0;

	// 1) Input buffer is read:
	InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> inputBufferEntry =
		inputBuffer.read(inputBufferEntries, inputBufferAddress);
#pragma HLS AGGREGATE variable=inputBufferEntry

	constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
	ib_tag_t tag = memoryAddress >> numIndexBits;

	// Skip operation if the previous access is equal to the current:
	// if (inputBufferEntry.lastAddress != memoryAddress) {
	if(true) {
		class_t predictedClass;
// #pragma HLS DEPENDENCE false inter variable=predictedClass

		/*
		DictionaryEntry<delta_t, dic_confidence_t> updatedDictionaryEntries[NUM_CLASSES];
#pragma HLS ARRAY_PARTITION variable=updatedDictionaryEntries complete
*/
// #pragma HLS AGGREGATE variable=updatedDictionaryEntries
// #pragma HLS DEPENDENCE false inter variable=updatedDictionaryEntries


		bool performPrefetch = false;

		// Continue if there has been a hit:
		// if (inputBufferEntry.valid) {
		if(true) {

			// 2) If the predictedAddress is equal to the current, increment the confidence (decrease otherwise):
			if (inputBufferEntry.lastPredictedAddress == memoryAddress) {
				if (inputBufferEntry.confidence >= (MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_INCREASE))
					inputBufferEntry.confidence = MAX_PREDICTION_CONFIDENCE;
				else
					inputBufferEntry.confidence += PREDICTION_CONFIDENCE_INCREASE;
			}
			else {
				if (inputBufferEntry.confidence <= (-PREDICTION_CONFIDENCE_DECREASE))
					inputBufferEntry.confidence = 0;
				else
					inputBufferEntry.confidence += PREDICTION_CONFIDENCE_DECREASE;
			}

			if (inputBufferEntry.confidence >= PREDICTION_CONFIDENCE_THRESHOLD)
				performPrefetch = true;

			// 3) Compute the resulting delta and its class:
			delta_t delta = (delta_t)memoryAddress - (delta_t)inputBufferEntry.lastAddress;
			class_t dictionaryClass;
// #pragma HLS DEPENDENCE false inter variable=dictionaryClass
			DictionaryEntry < delta_t, dic_confidence_t > dictionaryEntry = dictionary.write(dictionaryEntries, delta, dictionaryClass);

			// 4) Fit-then-predict with the SVM:
			predictedClass = svm.fitAndPredict(weight_matrices, intercepts, inputBufferEntry.sequence, dictionaryClass);


			for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
#pragma HLS UNROLL
				inputBufferEntry.sequence[i] = inputBufferEntry.sequence[i + 1];
			}
			inputBufferEntry.sequence[SEQUENCE_LENGTH - 1] = dictionaryClass;

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
#pragma HLS UNROLL
				inputBufferEntry.sequence[i] = NUM_CLASSES;
			}

			// 4) Predict with the SVM:
			predictedClass = svm.predict(weight_matrices, intercepts, inputBufferEntry.sequence);

		}

		// 5) Get the finally predicted address:
		dic_index_t dummyIndex;
		delta_t predictedDelta;
// #pragma HLS DEPENDENCE false inter variable=predictedDelta

		/*
		for(int c = 0; c < NUM_CLASSES; c++){
		#pragma HLS UNROLL
			// updatedDictionaryEntries[c] = dictionaryEntries[c];
			updatedDictionaryEntries[c].confidence = dictionaryEntries[c].confidence;
			updatedDictionaryEntries[c].delta = dictionaryEntries[c].delta;
			updatedDictionaryEntries[c].valid = dictionaryEntries[c].valid;
		}
		*/


		predictedDelta = dictionary.read(
				// updatedDictionaryEntries,
				dictionaryEntries,
				true, predictedClass, 0, dummyIndex, false).delta;
		block_address_t predictedAddress = ((delta_t)memoryAddress + predictedDelta);

		if (performPrefetch) {
			addressesToPrefetch[0] = predictedAddress;

			// 6) Apply recursive/successive prefetching on the calculated prefetching degree (>= 1):
			prefetchDegree = confidenceLookUpTable.table[inputBufferEntry.confidence - PREDICTION_CONFIDENCE_THRESHOLD];
			/*
			succesivePrediction(inputBufferEntries, dictionaryEntries, weight_matrices, intercepts,
					predictedAddress, inputBufferEntry.sequence, prefetchDegree, addressesToPrefetch);
			*/
		}

		// 7) Update the input buffer with the entry:
		inputBufferEntry.lastAddress = memoryAddress;
		inputBufferEntry.lastPredictedAddress = predictedAddress;
		inputBuffer.write(inputBufferEntries, inputBufferAddress, inputBufferEntry);
	}

	return prefetchDegree;
}


