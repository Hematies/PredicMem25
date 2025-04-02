#include <iostream>
#include "top.hpp"
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

	static WeightMatrix<svm_weight_t, NUM_CLASSES_INCLUDING_NULL> weight_matrices[NUM_CLASSES];
	static WeightMatrix<svm_weight_t, NUM_CLASSES_INCLUDING_NULL> weight_matrices_copy[NUM_CLASSES];
	// #pragma HLS SHARED variable=weight_matrices->weights
	// #pragma HLS ARRAY_PARTITION variable=weight_matrices complete
	// #pragma HLS ARRAY_PARTITION variable=weight_matrices->weights complete

	static svm_weight_t intercepts[NUM_CLASSES];
	static svm_weight_t intercepts_copy[NUM_CLASSES];
	// #pragma HLS ARRAY_PARTITION variable=intercepts complete

	static SVM<svm_weight_t, class_t, svm_distance_t, NUM_CLASSES, NUM_CLASSES_INCLUDING_NULL> svm;

	svm.recursivelyPredictAndFit(weight_matrices, weight_matrices_copy, intercepts, intercepts_copy, input, target, output, MAX_PREFETCHING_DEGREE);
	// svm.fit(weight_matrices, weight_matrices_copy, intercepts, intercepts_copy, input, target);
	// output[0] = svm.predictAndFit(weight_matrices, weight_matrices_copy, intercepts, intercepts_copy, input, target);
	// output[0] = svm.predict(weight_matrices, intercepts, input);
}



void prefetchWithGASP(address_t instructionPointer, block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE]
		){
#pragma HLS INTERFACE ap_fifo port=addressesToPrefetch
#pragma HLS PIPELINE
	GASP<GASP_TYPES> gasp = GASP<GASP_TYPES>();
	gasp(instructionPointer, memoryAddress, addressesToPrefetch);
}

void prefetchWithSGASP(block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE]
		){
#pragma HLS INTERFACE ap_fifo port=addressesToPrefetch
#pragma HLS PIPELINE
	GASP<SGASP_TYPES> gasp = GASP<SGASP_TYPES>();
	gasp(memoryAddress >> REGION_BLOCK_SIZE_LOG2, memoryAddress, addressesToPrefetch);
}

void prefetchWithSGASPWithAXI(address_t inputAddress,
		axi_data_t *readPort,
		axi_data_t prefetchedData[MAX_PREFETCHING_DEGREE]
		){
// #pragma HLS TOP name=prefetchWithSGASPWithAXI
#pragma HLS INTERFACE mode=m_axi depth=32 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=32 num_write_outstanding=32 port=readPort offset=direct
#pragma HLS PIPELINE
	GASP<SGASP_TYPES> gasp = GASP<SGASP_TYPES>();

	block_address_t memoryBlockAddress, blockAddressesToPrefetch[MAX_PREFETCHING_DEGREE];
	address_t memoryBlockAddress_ = inputAddress >> BLOCK_SIZE_LOG2;

	if(((address_t)inputAddress >= START_CACHEABLE_MEM_REGION) && ((address_t)inputAddress < END_CACHEABLE_MEM_REGION)){
		gasp(memoryBlockAddress_ >> (REGION_BLOCK_SIZE_LOG2), memoryBlockAddress_, blockAddressesToPrefetch);

		for(int i = 0; i < MAX_PREFETCHING_DEGREE; i++){
			prefetchedData[i] = readPort[((address_t)blockAddressesToPrefetch[i]) << BLOCK_SIZE_LOG2];
		}
	}

}


void computeBurst(block_address_t prefetchBlockAddress, prefetch_block_burst_length_t prefetchBurstLength,
		address_t& prefetchAddress, burst_length_in_words_t& totalBurstLength){
#pragma HLS PIPELINE

	prefetchAddress = ((address_t)prefetchBlockAddress) << BLOCK_SIZE_LOG2;

	bool performPrefetch = prefetchAddress != 0;
	totalBurstLength =
			((((burst_length_in_words_t)prefetchBurstLength) + 1) << BLOCK_SIZE_LOG2) >> AXI_DATA_SIZE_BYTES_LOG2;

	if(!performPrefetch) totalBurstLength = 0;
}


void prefetchWithAXIBurst(hls::burst_maxi<axi_data_t>& readPort,
		hls::stream<axi_data_t>& prefetchedData,
	const address_t prefetchAddress, const burst_length_in_words_t totalBurstLength
){

	#pragma HLS PIPELINE

	readPort.read_request(prefetchAddress, totalBurstLength);
		
	for(burst_length_in_words_t i = 0; i < totalBurstLength; i++){
#pragma HLS LOOP_TRIPCOUNT min=0 max=64
#pragma HLS DEPENDENCE dependent=false type=inter variable=prefetchedData
#pragma HLS DEPENDENCE dependent=false type=intra variable=prefetchedData
	#pragma HLS PIPELINE
		axi_data_t data = readPort.read();
		prefetchedData.write(data);
	}

}

void prefetchWithBSGASPWithAXI(address_t inputAddress,
		burst_size_t burstSize,
		burst_length_t burstLength,
		hls::burst_maxi<axi_data_t> readPort,
		hls::stream<axi_data_t>& prefetchedData
		){
// #pragma HLS TOP name=prefetchWithBSGASPWithAXI
#pragma HLS INTERFACE mode=ap_ctrl_chain port=return
#pragma HLS INTERFACE mode=m_axi depth=256 latency=5 max_widen_bitwidth=512 num_read_outstanding=32 port=readPort offset=direct
#pragma HLS DATAFLOW

	BGASP<BSGASP_TYPES> bgasp = BGASP<BSGASP_TYPES>();
	prefetch_block_burst_length_t prefetchBurstLength = 0;
	bool performPrefetch = false;

	axi_data_t buffer[1 << ((NUM_CLASSES - 1) + BLOCK_SIZE_LOG2 - AXI_DATA_SIZE_BYTES_LOG2)];

	block_address_t prefetchAddress_, memoryBlockAddress = inputAddress >> BLOCK_SIZE_LOG2;
	address_t prefetchAddress = inputAddress >> BLOCK_SIZE_LOG2;
	region_address_t regionAddress = memoryBlockAddress >> (REGION_BLOCK_SIZE_LOG2);
	block_burst_length_t blockBurstLength_ = (((burst_size_and_length_t)(burstLength + 1)) << burstSize) >> BLOCK_SIZE_LOG2;
	prefetch_block_burst_length_t blockBurstLength =
			(blockBurstLength_ >> AXI_MAX_BURST_BLOCK_LOG2) != 0?
					(((prefetch_block_burst_length_t)1) << AXI_MAX_BURST_BLOCK_LOG2) :
					(prefetch_block_burst_length_t)blockBurstLength_;
	burst_length_in_words_t totalBurstLength;


	bgasp(regionAddress, memoryBlockAddress, blockBurstLength,
			prefetchAddress_, prefetchBurstLength);

	computeBurst(prefetchAddress_, prefetchBurstLength,
			prefetchAddress, totalBurstLength);
	prefetchWithAXIBurst(readPort, prefetchedData, prefetchAddress, totalBurstLength);

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
#pragma HLS TOP name=prefetchWithSGASPWithNop
#pragma HLS INTERFACE ap_fifo port=addressesToPrefetch
#pragma HLS PIPELINE
	GASP<SGASP_TYPES> gasp = GASP<SGASP_TYPES>();
	if(!nop) gasp(memoryAddress >> REGION_BLOCK_SIZE_LOG2,
			memoryAddress, addressesToPrefetch);
}



