#pragma once
#include "global.hpp"

#define GASP_TYPES address_t, ib_index_t, ib_way_t, ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t, \
	 dic_index_t, delta_t,  dic_confidence_t, \
	 svm_weight_t, svm_distance_t

#define SGASP_TYPES region_address_t, ib_index_t, ib_way_t, ib_region_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t, \
	 dic_index_t, delta_t, dic_confidence_t, \
	 svm_weight_t, svm_distance_t

template<typename address_t, typename ib_index_t, typename ib_way_t, typename ib_tag_t, typename block_address_t, typename class_t, typename ib_confidence_t, typename ib_lru_t,
	typename dic_index_t, typename delta_t, typename dic_confidence_t,
	typename svm_weight_t, typename svm_distance_t>
class GASP {
protected:

public:

	void operator()(address_t inputBufferAddress, block_address_t memoryAddress,
			block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE]){
	#pragma HLS ARRAY_PARTITION variable=addressesToPrefetch complete

		static LookUpTable<ib_confidence_t, MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_THRESHOLD + 1>
			confidenceLookUpTable = fillUniformLookUpTable<ib_confidence_t, MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_THRESHOLD + 1>(1, MAX_PREFETCHING_DEGREE);

		static DictionaryEntriesMatrix<delta_t, dic_confidence_t> dictionaryEntriesMatrix
			= initDictionaryEntries<delta_t, dic_confidence_t>();
	#pragma HLS ARRAY_PARTITION variable=dictionaryEntriesMatrix.entries complete

		static InputBufferEntriesMatrix<ib_tag_t, block_address_t, class_t, ib_lru_t>
				inputBufferEntriesMatrix = initInputBufferEntries<ib_tag_t, block_address_t, class_t, ib_lru_t>();
	#pragma HLS ARRAY_RESHAPE variable=inputBufferEntriesMatrix.entries dim=2 complete
	#pragma HLS ARRAY_RESHAPE variable=inputBufferEntriesMatrix.entries dim=3 complete
	// #pragma HLS ARRAY_PARTITION variable=inputBufferEntriesMatrix.entries dim=0 complete
	#pragma HLS BIND_STORAGE variable=inputBufferEntriesMatrix.entries type=RAM_T2P impl=bram latency=1

	#pragma HLS DEPENDENCE array false variable=inputBufferEntriesMatrix.entries

		static ConfidenceBufferEntriesMatrix<ib_confidence_t, block_address_t>
			confidenceBufferEntriesMatrix = initConfidenceBufferEntries<ib_confidence_t, block_address_t>();
	#pragma HLS ARRAY_RESHAPE variable=confidenceBufferEntriesMatrix.entries dim=2 complete
	#pragma HLS ARRAY_RESHAPE variable=confidenceBufferEntriesMatrix.entries dim=3 complete
	// #pragma HLS ARRAY_PARTITION variable=confidenceBufferEntriesMatrix.entries dim=0 complete
	#pragma HLS BIND_STORAGE variable=confidenceBufferEntriesMatrix.entries type=RAM_T2P impl=bram latency=1

	#pragma HLS DEPENDENCE array false variable=confidenceBufferEntriesMatrix.entries

		static SVMWholeMatrix<svm_weight_t> svmMatrix = initSVMData<svm_weight_t>();
		static SVMWholeMatrix<svm_weight_t> svmMatrixCopy = initSVMData<svm_weight_t>();
	#pragma HLS ARRAY_PARTITION variable=svmMatrix.weightMatrices complete
	#pragma HLS ARRAY_PARTITION variable=svmMatrixCopy.weightMatrices complete
	#pragma HLS ARRAY_PARTITION variable=svmMatrix.intercepts complete

		static InputBuffer<address_t, ib_index_t, ib_way_t, ib_tag_t, block_address_t, class_t, ib_lru_t> inputBuffer;
	#pragma HLS DEPENDENCE false variable=inputBuffer

		static ConfidenceBuffer<ib_confidence_t, block_address_t> confidenceBuffer;
	#pragma HLS DEPENDENCE false variable=confidenceBuffer

		static Dictionary<dic_index_t, delta_t, dic_confidence_t> dictionary;
	#pragma HLS DEPENDENCE false variable=dictionary

		static SVM<svm_weight_t, class_t, svm_distance_t> svm;
	#pragma HLS DEPENDENCE false variable=svm

	#pragma HLS PIPELINE
		int prefetchDegree = 0;

		// 1) Input buffer is read:
		bool isInputBufferHit;
		ib_index_t index;
		ib_way_t way;
		InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_lru_t> inputBufferEntryDummy;
		InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_lru_t> inputBufferEntry =
			inputBuffer(inputBufferEntriesMatrix.entries, inputBufferAddress, inputBufferEntryDummy, true, isInputBufferHit,
			index, way);

		ConfidenceBufferEntry<ib_confidence_t, block_address_t> confidenceBufferEntry = 
			confidenceBuffer.read(confidenceBufferEntriesMatrix.entries, index, way);
	// #pragma HLS AGGREGATE variable=inputBufferEntry

		constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
		ib_tag_t tag = inputBufferAddress >> numIndexBits;

		// Skip operation if the previous access is equal to the current:
		if (inputBufferEntry.lastAddress != memoryAddress) {
			class_t predictedClasses[MAX_PREFETCHING_DEGREE];


			bool performPrefetch = false;
			
			class_t sequence[SEQUENCE_LENGTH];
			class_t dictionaryClass;

			// Continue if there has been a hit:
			if (isInputBufferHit) {

				// 3) Compute the resulting delta and its class:
				delta_t delta = (delta_t)memoryAddress - (delta_t)inputBufferEntry.lastAddress;
				bool dummyIsHit;
				DictionaryEntry < delta_t, dic_confidence_t > dictionaryEntry = dictionary.write(
						dictionaryEntriesMatrix.entries, delta, dictionaryClass, dummyIsHit);

				for (int i = 0; i < SEQUENCE_LENGTH; i++) {
				#pragma HLS UNROLL
					sequence[i] = inputBufferEntry.sequence[i];
				}
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
				inputBufferEntry.lruCounter = 1;
				for (int i = 0; i < SEQUENCE_LENGTH; i++) {
	#pragma HLS UNROLL
					inputBufferEntry.sequence[i] = NUM_CLASSES;
					sequence[i] = NUM_CLASSES;
				}
			}
			// 2) Update the input buffer with the entry:
			inputBufferEntry.lastAddress = memoryAddress;
			bool isInputBufferHitDummy;
			inputBuffer(inputBufferEntriesMatrix.entries, inputBufferAddress, inputBufferEntry, false, isInputBufferHitDummy, index, way);


			if(isInputBufferHit){
				// 3) If the predictedAddress is equal to the current, increment the confidence (decrease otherwise):
				if (confidenceBufferEntry.lastPredictedAddress == memoryAddress) {
					if (confidenceBufferEntry.confidence >= (MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_INCREASE))
						confidenceBufferEntry.confidence = MAX_PREDICTION_CONFIDENCE;
					else
						confidenceBufferEntry.confidence += PREDICTION_CONFIDENCE_INCREASE;
				}
				else {
					if (confidenceBufferEntry.confidence <= (-PREDICTION_CONFIDENCE_DECREASE))
						confidenceBufferEntry.confidence = 0;
					else
						confidenceBufferEntry.confidence += PREDICTION_CONFIDENCE_DECREASE;
				}

				if (confidenceBufferEntry.confidence >= PREDICTION_CONFIDENCE_THRESHOLD)
					performPrefetch = true;


				// 4) Predict-then-fit with the SVM applying recursive/successive prefetching
				// on the calculated prefetching degree (>= 1):
				prefetchDegree = confidenceLookUpTable.table[confidenceBufferEntry.confidence - PREDICTION_CONFIDENCE_THRESHOLD];
				svm.recursivelyPredictAndFit(svmMatrix.weightMatrices, svmMatrixCopy.weightMatrices, svmMatrix.intercepts, svmMatrixCopy.intercepts, sequence, dictionaryClass, predictedClasses,
						prefetchDegree == 0? 1 : prefetchDegree);
			}
			else{
				// 3) Reset the confidence:
				confidenceBufferEntry.confidence = 0;

				// 4) Predict with the SVM:
				predictedClasses[0] = svm.predict(svmMatrixCopy.weightMatrices, svmMatrixCopy.intercepts, inputBufferEntry.sequence);
			}


			// 5) Get the finally predicted address:
			dic_index_t dummyIndex;
			delta_t predictedDelta;

			predictedDelta = dictionaryEntriesMatrix.entries[(int)predictedClasses[0]].delta;
			block_address_t predictedAddress = ((delta_t)memoryAddress + predictedDelta);

			if(performPrefetch){
				block_address_t addressesToPrefetch_[MAX_PREFETCHING_DEGREE];
				block_address_t prevAddress = predictedAddress;
				addressesToPrefetch_[0] = predictedAddress;
				for(int i = 1; i < MAX_PREFETCHING_DEGREE; i++){
	#pragma HLS UNROLL
					if(i < prefetchDegree){
						delta_t predictedDelta_ = dictionaryEntriesMatrix.entries[(int)predictedClasses[i]].delta;
						block_address_t addr = (delta_t)prevAddress + predictedDelta_;
						addressesToPrefetch_[i] = addr;
						prevAddress = addr;
					}
				}
				for(int i = 0; i < MAX_PREFETCHING_DEGREE; i++){
					if(i < prefetchDegree){
						addressesToPrefetch[i] = addressesToPrefetch_[i];
					}
				}
			}
			else{
				for(int i = 0; i < MAX_PREFETCHING_DEGREE; i++){
					#pragma HLS UNROLL
					addressesToPrefetch[i] = 0;
				}
			}

			// 6) Update the confidence buffer with the entry:
			confidenceBufferEntry.lastPredictedAddress = predictedAddress;
			confidenceBuffer.write(confidenceBufferEntriesMatrix.entries, index, way, confidenceBufferEntry);
		}
	}
};
