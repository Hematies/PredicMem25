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
	res = dictionary.read(dictionaryEntries, true, 0, 0, resultIndex);
	res = dictionary.read(dictionaryEntries, false, 0, -5, resultIndex);
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


void testSVM(class_t output[MAX_PREFETCHING_DEGREE]){
#pragma HLS TOP
// #pragma PIPELINE
	static class_t input[SEQUENCE_LENGTH];
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
	svm.fitAndRecursivelyPredict(weight_matrices, intercepts, input, 0, output, 1);
}


int testGASP(address_t inputBufferAddress, block_address_t memoryAddress, block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE]){
// #pragma HLS INTERFACE ap_ctrl_chain port=return
#pragma HLS TOP

	static DictionaryEntry<delta_t, dic_confidence_t> dictionaryEntries[NUM_CLASSES];
#pragma HLS ARRAY_PARTITION variable=dictionaryEntries complete

	static InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t>
			inputBufferEntries[IB_NUM_SETS][IB_NUM_WAYS];
// #pragma HLS SHARED variable=inputBufferEntries
	#pragma HLS ARRAY_PARTITION variable=inputBufferEntries dim=0 factor=2 block

	static weigth_matrix_t<svm_weight_t> weight_matrices[NUM_CLASSES];
#pragma HLS ARRAY_PARTITION variable=weight_matrices complete
	static svm_weight_t intercepts[NUM_CLASSES];
#pragma HLS ARRAY_PARTITION variable=intercepts complete

	static GASP gasp = GASP();
	int res = gasp(inputBufferEntries, dictionaryEntries, weight_matrices, intercepts, inputBufferAddress, memoryAddress, addressesToPrefetch);
	return res;
}


