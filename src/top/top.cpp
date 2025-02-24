#include <iostream>
#include "../include/global.hpp"
#include <hls_stream.h>
#include <ap_axi_sdata.h>

DictionaryEntry<delta_t, dic_confidence_t> operateDictionary(dic_index_t index, delta_t delta, bool performRead, dic_index_t &resultIndex, bool &isHit){
	#pragma HLS PIPELINE

	static DictionaryEntriesMatrix<delta_t, dic_confidence_t> dictionaryEntriesMatrix
				= initDictionaryEntries<delta_t, dic_confidence_t>();
		#pragma HLS ARRAY_PARTITION variable=dictionaryEntriesMatrix.entries complete

	static Dictionary<dic_index_t, delta_t, dic_confidence_t> dictionary;
		#pragma HLS DEPENDENCE false variable=dictionary

	// resultIndex = 0;
	DictionaryEntry<delta_t, dic_confidence_t> res;
	if(performRead){
		res = dictionary.read(dictionaryEntriesMatrix.entries, index != NUM_CLASSES, index, delta, resultIndex, true, isHit);
	}
	else{
		res = dictionary.write(dictionaryEntriesMatrix.entries, delta, resultIndex, isHit);
	}
	return res;
}

InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> operateInputBuffer(address_t addr, InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> entry,
		bool performRead, bool& isHit){
#pragma HLS PIPELINE


	static InputBufferEntriesMatrix<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t>
			inputBufferEntriesMatrix = initInputBufferEntries<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t>();
	static InputBufferEntriesMatrix<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t>
				inputBufferEntriesMatrixCopy = initInputBufferEntries<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t>();
// #pragma HLS BIND_STORAGE variable=inputBufferEntriesMatrix.entries type=ram_t2p impl=lutram latency=1
// #pragma HLS ARRAY_PARTITION variable=inputBufferEntriesMatrix.entries dim=2 factor=2 block
// #pragma HLS ARRAY_PARTITION variable=inputBufferEntriesMatrix.entries dim=0 complete
// #pragma HLS ARRAY_RESHAPE dim=2 factor=2 object type=block variable=inputBufferEntriesMatrix.entries
#pragma HLS DEPENDENCE array false variable=inputBufferEntriesMatrix.entries

	static InputBuffer<address_t, ib_index_t, ib_way_t, ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> inputBuffer;
	// #pragma HLS DEPENDENCE false variable=inputBuffer


	InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> res;
	/*
	if(performRead){
		res = inputBuffer.read(inputBufferEntriesMatrixCopy.entries, addr, isHit);
	}
	else{
		inputBuffer.write(inputBufferEntriesMatrix.entries, inputBufferEntriesMatrixCopy.entries, addr, entry);
		res = entry;
	}
	*/
	res = inputBuffer(inputBufferEntriesMatrix.entries, addr, entry, performRead, isHit);
	return res;
}


void operateSVM(class_t input[SEQUENCE_LENGTH], class_t target, class_t output[MAX_PREFETCHING_DEGREE]){

#pragma HLS PIPELINE
	#pragma HLS INTERFACE ap_fifo port=output

	#pragma HLS ARRAY_PARTITION variable=input complete

	static WeightMatrix<svm_weight_t> weight_matrices[NUM_CLASSES];
	static WeightMatrix<svm_weight_t> weight_matrices_copy[NUM_CLASSES];
	// #pragma HLS SHARED variable=weight_matrices->weights
	// #pragma HLS ARRAY_PARTITION variable=weight_matrices complete
	// #pragma HLS ARRAY_PARTITION variable=weight_matrices->weights complete

	static svm_weight_t intercepts[NUM_CLASSES];
	static svm_weight_t intercepts_copy[NUM_CLASSES];
	// #pragma HLS ARRAY_PARTITION variable=intercepts complete

	static SVM<svm_weight_t, class_t, svm_distance_t> svm;

	svm.recursivelyPredictAndFit(weight_matrices, weight_matrices_copy, intercepts, intercepts_copy, input, target, output, MAX_PREFETCHING_DEGREE);
	// svm.fit(weight_matrices, weight_matrices_copy, intercepts, intercepts_copy, input, target);
	// output[0] = svm.predictAndFit(weight_matrices, weight_matrices_copy, intercepts, intercepts_copy, input, target);
	// output[0] = svm.predict(weight_matrices, intercepts, input);
}



void prefetchWithGASP(address_t instructionPointer, block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE],
		bool nop
		){
#pragma HLS INTERFACE ap_fifo port=addressesToPrefetch
#pragma HLS PIPELINE
	GASP<GASP_TYPES> gasp = GASP<GASP_TYPES>();
	gasp(instructionPointer, memoryAddress, addressesToPrefetch);
}

void prefetchWithSGASP(block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE],
		bool nop
		){
#pragma HLS INTERFACE ap_fifo port=addressesToPrefetch
#pragma HLS PIPELINE
	GASP<SGASP_TYPES> gasp = GASP<SGASP_TYPES>();
	gasp(memoryAddress >> REGION_BLOCK_SIZE_LOG2, memoryAddress, addressesToPrefetch);
}

void prefetchWithSGASPWithAXI(axi_data_t *inputAddress,
		axi_data_t *outputAddress, axi_data_t prefetchedData[MAX_PREFETCHING_DEGREE]
		){
#pragma HLS INTERFACE m_axi port=outputAddress depth=4 offset=direct
#pragma HLS INTERFACE mode=s_axilite port=inputAddress depth=4
#pragma HLS PIPELINE
	GASP<SGASP_TYPES> gasp = GASP<SGASP_TYPES>();

	block_address_t memoryBlockAddress, blockAddressesToPrefetch[MAX_PREFETCHING_DEGREE];
	// memoryBlockAddress = ((address_t)inputAddress);//  >> BLOCK_SIZE_LOG2;
	address_t memoryBlockAddress_ = ((address_t)inputAddress);//  >> BLOCK_SIZE_LOG2;
	memoryBlockAddress_ = memoryBlockAddress_ >> BLOCK_SIZE_LOG2;
	// memoryBlockAddress = memoryBlockAddress_.range(NUM_BLOCK_ADDRESS_BITS - 1, 0);
	// memoryBlockAddress = memoryBlockAddress_ | 0x0;

	*inputAddress = outputAddress[(address_t)inputAddress];
	gasp(memoryBlockAddress_ >> (REGION_BLOCK_SIZE_LOG2), memoryBlockAddress_, blockAddressesToPrefetch);

	for(int i = 0; i < MAX_PREFETCHING_DEGREE; i++){
		prefetchedData[i] = outputAddress[blockAddressesToPrefetch[i] << BLOCK_SIZE_LOG2];
	}
}


void prefetchWithGASPWithNop(address_t instructionPointer, block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE],
		bool nop
		){
#pragma HLS INTERFACE ap_fifo port=addressesToPrefetch
#pragma HLS PIPELINE
	GASP<GASP_TYPES> gasp = GASP<GASP_TYPES>();
	if(!nop) gasp(instructionPointer, memoryAddress, addressesToPrefetch);
}

void prefetchWithSGASPWithNop(block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE],
		bool nop
		){
#pragma HLS INTERFACE ap_fifo port=addressesToPrefetch
#pragma HLS PIPELINE
	GASP<SGASP_TYPES> gasp = GASP<SGASP_TYPES>();
	if(!nop) gasp(memoryAddress >> REGION_BLOCK_SIZE_LOG2, memoryAddress, addressesToPrefetch);
}



