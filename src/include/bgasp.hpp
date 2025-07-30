#pragma once
#include "global.hpp"

#define BGASP_TYPES address_t, ib_index_t, ib_way_t, ib_tag_t, block_address_t, class_t, prefetch_block_burst_length_t, ib_confidence_t, ib_lru_t, \
	 dic_index_t, delta_t,  dic_confidence_t, \
	 svm_weight_t, svm_distance_t

#define BSGASP_TYPES region_address_t, ib_index_t, ib_way_t, ib_region_tag_t, block_address_t, class_t, prefetch_block_burst_length_t, ib_confidence_t, ib_lru_t, \
	 dic_index_t, delta_t, dic_confidence_t, \
	 svm_weight_t, svm_distance_t

template<typename address_t, typename ib_index_t, typename ib_way_t, typename ib_tag_t, typename block_address_t, typename class_t, typename burst_length_t,
	typename ib_confidence_t, typename ib_lru_t,
	typename dic_index_t, typename delta_t, typename dic_confidence_t,
	typename svm_weight_t, typename svm_distance_t>
class BGASP {
protected:

public:

	void phase1(address_t inputBufferAddress, block_address_t memoryAddress, burst_length_t burstLength,
		block_address_t& predictedAddress, burst_length_t& predictedBurstLength,
		ib_index_t& index, ib_way_t& way, bool& nop, bool& isInputBufferHit) { 

		static DictionaryEntriesMatrix<delta_t, dic_confidence_t> dictionaryEntriesMatrix
			= initDictionaryEntries<delta_t, dic_confidence_t>();
		#pragma HLS ARRAY_PARTITION variable=dictionaryEntriesMatrix.entries complete

			static BurstInputBufferEntriesMatrix<ib_tag_t, block_address_t, class_t, burst_length_t, ib_lru_t>
					inputBufferEntriesMatrix = initBurstInputBufferEntries<ib_tag_t, block_address_t, class_t, burst_length_t, ib_lru_t>();
		#pragma HLS ARRAY_RESHAPE variable=inputBufferEntriesMatrix.entries dim=2 complete
		#pragma HLS ARRAY_RESHAPE variable=inputBufferEntriesMatrix.entries dim=3 complete
		#pragma HLS BIND_STORAGE variable=inputBufferEntriesMatrix.entries type=RAM_T2P impl=bram latency=1

		#pragma HLS DEPENDENCE array false variable=inputBufferEntriesMatrix.entries

		static BurstForwardingBufferEntriesMatrix<address_t, block_address_t, class_t, burst_length_t>
			forwardingBufferEntriesMatrix = initBurstForwardingBufferEntries<address_t, block_address_t, class_t, burst_length_t>();
		#pragma HLS ARRAY_RESHAPE variable=forwardingBufferEntriesMatrix.entries complete
		static forwarding_index_t forwardingBufferNextSlot = 0;
		forwarding_index_t forwardingBufferCurrentSlot = 0;

		static SVMWholeMatrix<svm_weight_t> svmMatrix = initSVMData<svm_weight_t>();
		static SVMWholeMatrix<svm_weight_t> svmMatrixCopy = initSVMData<svm_weight_t>();
		#pragma HLS ARRAY_PARTITION variable=svmMatrix.weightMatrices complete
		#pragma HLS ARRAY_PARTITION variable=svmMatrixCopy.weightMatrices complete
		#pragma HLS ARRAY_PARTITION variable=svmMatrix.intercepts complete
		#pragma HLS ARRAY_PARTITION variable=svmMatrixCopy.intercepts complete

		static BurstSVMWholeMatrix<svm_weight_t> burstSvmMatrix = initBurstSVMData<svm_weight_t>();
		static BurstSVMWholeMatrix<svm_weight_t> burstSvmMatrixCopy = initBurstSVMData<svm_weight_t>();
		#pragma HLS ARRAY_PARTITION variable=burstSvmMatrix.weightMatrices complete
		#pragma HLS ARRAY_PARTITION variable=burstSvmMatrixCopy.weightMatrices complete
		#pragma HLS ARRAY_PARTITION variable=burstSvmMatrix.intercepts complete

		static BurstInputBuffer<address_t, ib_index_t, ib_way_t, ib_tag_t, block_address_t, class_t, burst_length_t, ib_lru_t> inputBuffer;
		#pragma HLS DEPENDENCE false variable=inputBuffer

		static BurstForwardingBuffer<address_t, ib_tag_t, block_address_t, class_t, burst_length_t, ib_lru_t> forwardingBuffer;
		#pragma HLS DEPENDENCE false variable=forwardingBuffer

		static Dictionary<dic_index_t, delta_t, dic_confidence_t> dictionary;
		#pragma HLS DEPENDENCE false variable=dictionary

		static SVM<svm_weight_t, class_t, svm_distance_t, NUM_CLASSES, NUM_CLASSES_INCLUDING_NULL> svm;
		#pragma HLS DEPENDENCE false variable=svm

		static SVM<svm_weight_t, burst_length_t, svm_distance_t, NUM_BURST_CLASSES, NUM_BURST_CLASSES_INCLUDING_NULL> burstSvm;
		#pragma HLS DEPENDENCE false variable=svm

		#pragma HLS PIPELINE
		
		int prefetchDegree = 1;
		
		if(!nop){
			// 1) Input buffer is read:
			BurstInputBufferEntry<ib_tag_t, block_address_t, class_t, burst_length_t, ib_lru_t> inputBufferEntryDummy;
			BurstInputBufferEntry<ib_tag_t, block_address_t, class_t, burst_length_t, ib_lru_t> inputBufferEntry =
				inputBuffer(inputBufferEntriesMatrix.entries, inputBufferAddress, inputBufferEntryDummy, true, isInputBufferHit,
				index, way);

		// #pragma HLS AGGREGATE variable=inputBufferEntry

			constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
			ib_tag_t tag = inputBufferAddress >> numIndexBits;

			// 1.5) Forwarding buffer is read:
			bool isForwardingBufferHit;
			BurstForwardingBufferEntry<address_t, block_address_t, class_t, burst_length_t> forwardingBufferEntry =
				forwardingBuffer.read(forwardingBufferEntriesMatrix.entries, inputBufferAddress, 
					forwardingBufferCurrentSlot, isForwardingBufferHit);

			block_address_t lastAddress;
			class_t updatedSequence[SEQUENCE_LENGTH];
			class_t sequence[SEQUENCE_LENGTH];
			burst_length_t burstUpdatedSequence[SEQUENCE_LENGTH];
			burst_length_t burstSequence[SEQUENCE_LENGTH];
			bool resetLruCounter = false;

			// If the forwarding buffer is hit, we obtain the data from it. Else, we allocate a new entry and take the data from the input buffer:
			if(isForwardingBufferHit){
				lastAddress = forwardingBufferEntry.lastAddress;
				for(int k = 0; k < SEQUENCE_LENGTH; k++){
					#pragma HLS UNROLL
					sequence[k] = forwardingBufferEntry.sequence[k];
					burstSequence[k] = forwardingBufferEntry.burstLengthSequence[k];
				}
			}

			else{

				lastAddress = inputBufferEntry.lastAddress;
				for(int k = 0; k < SEQUENCE_LENGTH; k++){
					#pragma HLS UNROLL
					sequence[k] = inputBufferEntry.sequence[k];
					burstSequence[k] = inputBufferEntry.burstLengthSequence[k];
				}

			}
			

			// Skip operation if the previous access is equal to the current:
			nop = inputBufferEntry.lastAddress == memoryAddress;

			if (!nop) {
				class_t predictedClasses[MAX_PREFETCHING_DEGREE];
				burst_length_t predictedBurstLengths[MAX_PREFETCHING_DEGREE];

				bool performPrefetch = false;
				bool performBurstPrefetch = false;
				
				if(isInputBufferHit) {
					// 3) Compute the resulting delta and its class:
					delta_t delta = (delta_t)memoryAddress - (delta_t)lastAddress;
					class_t dictionaryClass;
					bool dummyIsHit;
					DictionaryEntry < delta_t, dic_confidence_t > dictionaryEntry = dictionary.write(
							dictionaryEntriesMatrix.entries, delta, dictionaryClass, dummyIsHit);
					
					for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
						#pragma HLS UNROLL
						updatedSequence[i] = sequence[i + 1];
						burstUpdatedSequence[i] = burstSequence[i + 1];
					}
					updatedSequence[SEQUENCE_LENGTH - 1] = dictionaryClass;
					burstUpdatedSequence[SEQUENCE_LENGTH - 1] = burstLength;


					// 3.5 Update the forwarding buffer to include the new sequence and lastAddress: 

					if(!isForwardingBufferHit){
						forwardingBuffer.write(forwardingBufferEntriesMatrix.entries, memoryAddress, updatedSequence, burstSequence, inputBufferAddress,
							forwardingBufferCurrentSlot, forwardingBufferNextSlot); 
					}
					else{
						forwardingBufferEntriesMatrix.entries[forwardingBufferCurrentSlot].lastAddress = memoryAddress;
						for(int k = 0; k < SEQUENCE_LENGTH; k++){
							#pragma HLS UNROLL
							forwardingBufferEntriesMatrix.entries[forwardingBufferCurrentSlot].sequence[k] = updatedSequence[k];
							forwardingBufferEntriesMatrix.entries[forwardingBufferCurrentSlot].burstLengthSequence[k] = burstUpdatedSequence[k];
						}
					}

					// 6) Update the input buffer with the entry:
					for (int i = 0; i < SEQUENCE_LENGTH; i++) {
						#pragma HLS UNROLL
						inputBufferEntry.sequence[i] = updatedSequence[i];
						inputBufferEntry.burstLengthSequence[i] = burstUpdatedSequence[i];
					}

					inputBufferEntry.tag = tag;
					inputBufferEntry.valid = true;
					inputBufferEntry.lastAddress = memoryAddress;
					// inputBufferEntry.lastPredictedAddress = predictedAddress;
					// inputBuffer.write(inputBufferEntriesMatrix.entries, inputBufferEntriesMatrixCopy.entries,inputBufferAddress, inputBufferEntry);
					bool isInputBufferHitDummy;
					inputBuffer(inputBufferEntriesMatrix.entries, inputBufferAddress, inputBufferEntry, false, isInputBufferHitDummy, index, way);


					// 4) Predict-then-fit with the SVM applying recursive/successive prefetching
					// on the calculated prefetching degree (>= 1):
					svm.recursivelyPredictAndFit(svmMatrix.weightMatrices, svmMatrixCopy.weightMatrices, svmMatrix.intercepts, svmMatrixCopy.intercepts, sequence, dictionaryClass, predictedClasses,
							1);
					burstSvm.recursivelyPredictAndFit(burstSvmMatrix.weightMatrices, burstSvmMatrixCopy.weightMatrices, burstSvmMatrix.intercepts, 
						burstSvmMatrixCopy.intercepts, burstSequence, burstLength,
						predictedBurstLengths,
						1);
				}
				// If there has been a miss, prepare a blank new input buffer entry:
				else {
					inputBufferEntry.lruCounter = 1;
					for (int i = 0; i < SEQUENCE_LENGTH; i++) {
					#pragma HLS UNROLL
						sequence[i] = NUM_CLASSES_INCLUDING_NULL;
						updatedSequence[i] = NUM_CLASSES_INCLUDING_NULL;
						burstSequence[i] = NUM_BURST_CLASSES_INCLUDING_NULL;
						burstUpdatedSequence[i] = NUM_BURST_CLASSES_INCLUDING_NULL;
					}

					/* HABRIA QUE PONER ESTO AQUI
						* forwardingBuffer.write(forwardingBufferEntriesMatrix.entries, memoryAddress, updatedSequence, inputBufferAddress,
							forwardingBufferCurrentSlot, forwardingBufferNextSlot);
						*/

					// 6) Update the input buffer with the entry:
					for (int i = 0; i < SEQUENCE_LENGTH; i++) {
						#pragma HLS UNROLL
						inputBufferEntry.sequence[i] = updatedSequence[i];
						inputBufferEntry.burstLengthSequence[i] = updatedSequence[i];
					}

					inputBufferEntry.tag = tag;
					inputBufferEntry.valid = true;
					inputBufferEntry.lastAddress = memoryAddress;
					bool isInputBufferHitDummy;
					inputBuffer(inputBufferEntriesMatrix.entries, inputBufferAddress, inputBufferEntry, false, isInputBufferHitDummy, index, way);

					// 4) Predict with the SVM:
					predictedClasses[0] = svm.predict(svmMatrixCopy.weightMatrices, svmMatrixCopy.intercepts, sequence);
					predictedBurstLengths[0] = burstSvm.predict(burstSvmMatrixCopy.weightMatrices, burstSvmMatrixCopy.intercepts,
						burstSequence);

				}


				// 5) Get the finally predicted address:
				dic_index_t dummyIndex;
				delta_t predictedDelta;

				predictedDelta = dictionaryEntriesMatrix.entries[(int)predictedClasses[0]].delta;
				predictedAddress = ((delta_t)memoryAddress + predictedDelta);
				predictedBurstLength = predictedBurstLengths[0];
			}
		}		
	}

	void phase2(address_t inputBufferAddress, block_address_t memoryAddress, burst_length_t burstLength,
		block_address_t predictedAddress, burst_length_t predictedBurstLength, 
		block_address_t& prefetchAddress, burst_length_t& prefetchBurstLength,
		ib_index_t index, ib_way_t way, bool nop, bool isInputBufferHit){
		
		static LookUpTable<ib_confidence_t, MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_THRESHOLD + 1>
		confidenceLookUpTable = fillUniformLookUpTable<ib_confidence_t, MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_THRESHOLD + 1>(1, MAX_PREFETCHING_DEGREE);

		
		static BurstConfidenceForwardingBufferEntriesMatrix<address_t, ib_confidence_t, block_address_t, burst_length_t>
				confidenceForwardingBufferEntriesMatrix = initBurstConfidenceForwardingBufferEntries<address_t, ib_confidence_t, block_address_t, burst_length_t>();
			#pragma HLS ARRAY_RESHAPE variable=confidenceForwardingBufferEntriesMatrix.entries complete
			static conf_forwarding_index_t confidenceForwardingBufferNextSlot = 0;
			conf_forwarding_index_t confidenceForwardingBufferCurrentSlot = 0;

		// #pragma HLS DEPENDENCE array false variable=forwardingBufferEntriesMatrix.entries
			static BurstConfidenceBufferEntriesMatrix<ib_confidence_t, block_address_t, burst_length_t>
				confidenceBufferEntriesMatrix = initBurstConfidenceBufferEntries<ib_confidence_t, block_address_t, burst_length_t>();
		#pragma HLS ARRAY_RESHAPE variable=confidenceBufferEntriesMatrix.entries dim=2 complete
		#pragma HLS ARRAY_RESHAPE variable=confidenceBufferEntriesMatrix.entries dim=3 complete
		// #pragma HLS ARRAY_PARTITION variable=confidenceBufferEntriesMatrix.entries dim=0 complete
		#pragma HLS BIND_STORAGE variable=confidenceBufferEntriesMatrix.entries type=RAM_T2P impl=bram latency=1

		#pragma HLS DEPENDENCE array false variable=confidenceBufferEntriesMatrix.entries

		
		static BurstConfidenceBuffer<ib_confidence_t, block_address_t, burst_length_t> confidenceBuffer;
		#pragma HLS DEPENDENCE false variable=confidenceBuffer

		static BurstConfidenceForwardingBuffer<address_t, ib_confidence_t, block_address_t, burst_length_t> confidenceForwardingBuffer;
		#pragma HLS DEPENDENCE false variable=confidenceForwardingBuffer

		
		#pragma HLS PIPELINE
		int prefetchDegree = 1;

		if(!nop) {
			// 2) If the predictedAddress is equal to the current, increment the confidence (decrease otherwise):
			BurstConfidenceBufferEntry<ib_confidence_t, block_address_t, burst_length_t> confidenceBufferEntry;
			confidenceBufferEntry =
					confidenceBuffer.read(confidenceBufferEntriesMatrix.entries, index, way);

			// 2.5) Confidence forwarding buffer is read:
			bool isConfidenceForwardingBufferHit;
			BurstConfidenceForwardingBufferEntry<address_t, ib_confidence_t, block_address_t, burst_length_t> confidenceForwardingBufferEntry;

			ib_confidence_t confidence;
			block_address_t lastPredictedAddress;
			ib_confidence_t burstConfidence;
			burst_length_t lastPredictedBurst;


			if(isInputBufferHit){ // Dummy evaluation just to force the scheduling to delay the confidence update

				confidenceForwardingBufferEntry = confidenceForwardingBuffer.read(confidenceForwardingBufferEntriesMatrix.entries, inputBufferAddress,
						confidenceForwardingBufferCurrentSlot, isConfidenceForwardingBufferHit);
				if(isConfidenceForwardingBufferHit){
					confidence = confidenceForwardingBufferEntry.confidence;
					lastPredictedAddress = confidenceForwardingBufferEntry.lastPredictedAddress;
					burstConfidence = confidenceForwardingBufferEntry.burstConfidence;
					lastPredictedBurst = confidenceForwardingBufferEntry.lastPredictedBurstLength;
				}
				else{
					confidence = confidenceBufferEntry.confidence;
					lastPredictedAddress = confidenceBufferEntry.lastPredictedAddress;
					burstConfidence = confidenceBufferEntry.burstConfidence;
					lastPredictedBurst = confidenceBufferEntry.lastPredictedBurstLength;
				}

				bool addressHit = lastPredictedAddress == memoryAddress;
				bool burstHit = lastPredictedBurst == burstLength;

				// 3) If the predictedAddress is equal to the current, increment the confidence (decrease otherwise):
				if (addressHit) {

					if (confidence >= (MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_INCREASE))
						confidence = MAX_PREDICTION_CONFIDENCE;
					else
						confidence += PREDICTION_CONFIDENCE_INCREASE;

				}
				else {

					if (confidence <= (-PREDICTION_CONFIDENCE_DECREASE))
						confidence = 0;
					else
						confidence += PREDICTION_CONFIDENCE_DECREASE;

				}

				if (burstHit) {

					if (burstConfidence >= (MAX_PREDICTION_CONFIDENCE - BURST_PREDICTION_CONFIDENCE_INCREASE))
						burstConfidence = MAX_PREDICTION_CONFIDENCE;
					else
						burstConfidence += BURST_PREDICTION_CONFIDENCE_INCREASE;

				}
				else {

					if (burstConfidence <= (-BURST_PREDICTION_CONFIDENCE_DECREASE))
						burstConfidence = 0;
					else
						burstConfidence += BURST_PREDICTION_CONFIDENCE_DECREASE;

				}
				

				if(!isConfidenceForwardingBufferHit){
					confidenceForwardingBuffer.write(confidenceForwardingBufferEntriesMatrix.entries, confidence, predictedAddress,
							burstConfidence, lastPredictedBurst, 
							inputBufferAddress,
							confidenceForwardingBufferCurrentSlot, confidenceForwardingBufferNextSlot);
				}
				else{
					confidenceForwardingBufferEntriesMatrix.entries[confidenceForwardingBufferCurrentSlot].confidence = confidence;
					confidenceForwardingBufferEntriesMatrix.entries[confidenceForwardingBufferCurrentSlot].lastPredictedAddress = predictedAddress;
					confidenceForwardingBufferEntriesMatrix.entries[confidenceForwardingBufferCurrentSlot].confidence = burstConfidence;
					confidenceForwardingBufferEntriesMatrix.entries[confidenceForwardingBufferCurrentSlot].lastPredictedAddress = predictedBurstLength;
				}

			}
			else{
				// 3) Reset the confidence:
				confidence = 0;
			}


			// 6) Update the confidence buffer with the entry:

			confidenceBufferEntry.confidence = confidence;
			confidenceBufferEntry.lastPredictedAddress = predictedAddress;
			confidenceBufferEntry.burstConfidence = burstConfidence;
			confidenceBufferEntry.lastPredictedBurstLength = lastPredictedBurst;

			confidenceBuffer.write(confidenceBufferEntriesMatrix.entries, index, way, confidenceBufferEntry);

			bool performPrefetch = isInputBufferHit && confidence >= PREDICTION_CONFIDENCE_THRESHOLD;
			bool performBurstPrefetch = isInputBufferHit && burstConfidence >= BURST_PREDICTION_CONFIDENCE_THRESHOLD;

			// 7) Select the predicted address to prefetch:
			if(performPrefetch){
				prefetchAddress = predictedAddress;
				if(performBurstPrefetch)
					prefetchBurstLength = predictedBurstLength;
				else
					prefetchBurstLength = 1;
			}
			else{
				prefetchAddress = 0;
				prefetchBurstLength = 0;

			}
		}
	}
	
	void operator()(address_t inputBufferAddress, block_address_t memoryAddress, burst_length_t burstLength,
			block_address_t& outputAddress, burst_length_t& outputBurstLength
			){
		static LookUpTable<ib_confidence_t, MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_THRESHOLD + 1>
			confidenceLookUpTable = fillUniformLookUpTable<ib_confidence_t, MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_THRESHOLD + 1>(1, MAX_PREFETCHING_DEGREE);

		static DictionaryEntriesMatrix<delta_t, dic_confidence_t> dictionaryEntriesMatrix
			= initDictionaryEntries<delta_t, dic_confidence_t>();
		#pragma HLS ARRAY_PARTITION variable=dictionaryEntriesMatrix.entries complete

			static BurstInputBufferEntriesMatrix<ib_tag_t, block_address_t, class_t, burst_length_t, ib_lru_t>
					inputBufferEntriesMatrix = initBurstInputBufferEntries<ib_tag_t, block_address_t, class_t, burst_length_t, ib_lru_t>();
		#pragma HLS ARRAY_RESHAPE variable=inputBufferEntriesMatrix.entries dim=2 complete
		#pragma HLS ARRAY_RESHAPE variable=inputBufferEntriesMatrix.entries dim=3 complete
		#pragma HLS BIND_STORAGE variable=inputBufferEntriesMatrix.entries type=RAM_T2P impl=bram latency=1

		#pragma HLS DEPENDENCE array false variable=inputBufferEntriesMatrix.entries

		static BurstForwardingBufferEntriesMatrix<address_t, block_address_t, class_t, burst_length_t>
			forwardingBufferEntriesMatrix = initBurstForwardingBufferEntries<address_t, block_address_t, class_t, burst_length_t>();
		#pragma HLS ARRAY_RESHAPE variable=forwardingBufferEntriesMatrix.entries complete
		static forwarding_index_t forwardingBufferNextSlot = 0;
		forwarding_index_t forwardingBufferCurrentSlot = 0;

		static BurstConfidenceForwardingBufferEntriesMatrix<address_t, ib_confidence_t, block_address_t, burst_length_t>
				confidenceForwardingBufferEntriesMatrix = initBurstConfidenceForwardingBufferEntries<address_t, ib_confidence_t, block_address_t, burst_length_t>();
			#pragma HLS ARRAY_RESHAPE variable=confidenceForwardingBufferEntriesMatrix.entries complete
			static conf_forwarding_index_t confidenceForwardingBufferNextSlot = 0;
			conf_forwarding_index_t confidenceForwardingBufferCurrentSlot = 0;

		// #pragma HLS DEPENDENCE array false variable=forwardingBufferEntriesMatrix.entries
			static BurstConfidenceBufferEntriesMatrix<ib_confidence_t, block_address_t, burst_length_t>
				confidenceBufferEntriesMatrix = initBurstConfidenceBufferEntries<ib_confidence_t, block_address_t, burst_length_t>();
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
		#pragma HLS ARRAY_PARTITION variable=svmMatrixCopy.intercepts complete

			static BurstSVMWholeMatrix<svm_weight_t> burstSvmMatrix = initBurstSVMData<svm_weight_t>();
			static BurstSVMWholeMatrix<svm_weight_t> burstSvmMatrixCopy = initBurstSVMData<svm_weight_t>();
		#pragma HLS ARRAY_PARTITION variable=burstSvmMatrix.weightMatrices complete
		#pragma HLS ARRAY_PARTITION variable=burstSvmMatrixCopy.weightMatrices complete
		#pragma HLS ARRAY_PARTITION variable=burstSvmMatrix.intercepts complete

			static BurstInputBuffer<address_t, ib_index_t, ib_way_t, ib_tag_t, block_address_t, class_t, burst_length_t, ib_lru_t> inputBuffer;
		#pragma HLS DEPENDENCE false variable=inputBuffer

		static BurstConfidenceBuffer<ib_confidence_t, block_address_t, burst_length_t> confidenceBuffer;
		#pragma HLS DEPENDENCE false variable=confidenceBuffer

		static BurstForwardingBuffer<address_t, ib_tag_t, block_address_t, class_t, burst_length_t, ib_lru_t> forwardingBuffer;
	#pragma HLS DEPENDENCE false variable=forwardingBuffer

		static BurstConfidenceForwardingBuffer<address_t, ib_confidence_t, block_address_t, burst_length_t> confidenceForwardingBuffer;
	#pragma HLS DEPENDENCE false variable=confidenceForwardingBuffer

			static Dictionary<dic_index_t, delta_t, dic_confidence_t> dictionary;
		#pragma HLS DEPENDENCE false variable=dictionary

			static SVM<svm_weight_t, class_t, svm_distance_t, NUM_CLASSES, NUM_CLASSES_INCLUDING_NULL> svm;
		#pragma HLS DEPENDENCE false variable=svm

			static SVM<svm_weight_t, burst_length_t, svm_distance_t, NUM_BURST_CLASSES, NUM_BURST_CLASSES_INCLUDING_NULL> burstSvm;
		#pragma HLS DEPENDENCE false variable=svm

		#pragma HLS PIPELINE
		
		int prefetchDegree = 1;

		// 1) Input buffer is read:
		bool isInputBufferHit;
		ib_index_t index;
		ib_way_t way;
		BurstInputBufferEntry<ib_tag_t, block_address_t, class_t, burst_length_t, ib_lru_t> inputBufferEntryDummy;
		BurstInputBufferEntry<ib_tag_t, block_address_t, class_t, burst_length_t, ib_lru_t> inputBufferEntry =
			inputBuffer(inputBufferEntriesMatrix.entries, inputBufferAddress, inputBufferEntryDummy, true, isInputBufferHit,
			index, way);

	// #pragma HLS AGGREGATE variable=inputBufferEntry

		constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
		ib_tag_t tag = inputBufferAddress >> numIndexBits;

		// 1.5) Forwarding buffer is read:
		bool isForwardingBufferHit;
		BurstForwardingBufferEntry<address_t, block_address_t, class_t, burst_length_t> forwardingBufferEntry =
			forwardingBuffer.read(forwardingBufferEntriesMatrix.entries, inputBufferAddress, 
				forwardingBufferCurrentSlot, isForwardingBufferHit);

		block_address_t lastAddress;
		class_t updatedSequence[SEQUENCE_LENGTH];
		class_t sequence[SEQUENCE_LENGTH];
		burst_length_t burstUpdatedSequence[SEQUENCE_LENGTH];
		burst_length_t burstSequence[SEQUENCE_LENGTH];
		bool resetLruCounter = false;

		// If the forwarding buffer is hit, we obtain the data from it. Else, we allocate a new entry and take the data from the input buffer:
		if(isForwardingBufferHit){
			lastAddress = forwardingBufferEntry.lastAddress;
			for(int k = 0; k < SEQUENCE_LENGTH; k++){
				#pragma HLS UNROLL
				sequence[k] = forwardingBufferEntry.sequence[k];
				burstSequence[k] = forwardingBufferEntry.burstLengthSequence[k];
			}
		}

		else{

			lastAddress = inputBufferEntry.lastAddress;
			for(int k = 0; k < SEQUENCE_LENGTH; k++){
				#pragma HLS UNROLL
				sequence[k] = inputBufferEntry.sequence[k];
				burstSequence[k] = inputBufferEntry.burstLengthSequence[k];
			}

		}
		

		// Skip operation if the previous access is equal to the current:
		if (inputBufferEntry.lastAddress != memoryAddress) {
			class_t predictedClasses[MAX_PREFETCHING_DEGREE];
			burst_length_t predictedBurstLengths[MAX_PREFETCHING_DEGREE];

			bool performPrefetch = false;
			bool performBurstPrefetch = false;
			
			if(isInputBufferHit) {
				// 3) Compute the resulting delta and its class:
				delta_t delta = (delta_t)memoryAddress - (delta_t)lastAddress;
				class_t dictionaryClass;
				bool dummyIsHit;
				DictionaryEntry < delta_t, dic_confidence_t > dictionaryEntry = dictionary.write(
						dictionaryEntriesMatrix.entries, delta, dictionaryClass, dummyIsHit);
				
				for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
					#pragma HLS UNROLL
					updatedSequence[i] = sequence[i + 1];
					burstUpdatedSequence[i] = burstSequence[i + 1];
				}
				updatedSequence[SEQUENCE_LENGTH - 1] = dictionaryClass;
				burstUpdatedSequence[SEQUENCE_LENGTH - 1] = burstLength;


				// 3.5 Update the forwarding buffer to include the new sequence and lastAddress: 

				if(!isForwardingBufferHit){
					forwardingBuffer.write(forwardingBufferEntriesMatrix.entries, memoryAddress, updatedSequence, burstSequence, inputBufferAddress,
						forwardingBufferCurrentSlot, forwardingBufferNextSlot); 
				}
				else{
					forwardingBufferEntriesMatrix.entries[forwardingBufferCurrentSlot].lastAddress = memoryAddress;
					for(int k = 0; k < SEQUENCE_LENGTH; k++){
						#pragma HLS UNROLL
						forwardingBufferEntriesMatrix.entries[forwardingBufferCurrentSlot].sequence[k] = updatedSequence[k];
						forwardingBufferEntriesMatrix.entries[forwardingBufferCurrentSlot].burstLengthSequence[k] = burstUpdatedSequence[k];
					}
				}

				// 6) Update the input buffer with the entry:
				for (int i = 0; i < SEQUENCE_LENGTH; i++) {
					#pragma HLS UNROLL
					inputBufferEntry.sequence[i] = updatedSequence[i];
					inputBufferEntry.burstLengthSequence[i] = burstUpdatedSequence[i];
				}

				inputBufferEntry.tag = tag;
				inputBufferEntry.valid = true;
				inputBufferEntry.lastAddress = memoryAddress;
				// inputBufferEntry.lastPredictedAddress = predictedAddress;
				// inputBuffer.write(inputBufferEntriesMatrix.entries, inputBufferEntriesMatrixCopy.entries,inputBufferAddress, inputBufferEntry);
				bool isInputBufferHitDummy;
				inputBuffer(inputBufferEntriesMatrix.entries, inputBufferAddress, inputBufferEntry, false, isInputBufferHitDummy, index, way);


				// 4) Predict-then-fit with the SVM applying recursive/successive prefetching
				// on the calculated prefetching degree (>= 1):
				svm.recursivelyPredictAndFit(svmMatrix.weightMatrices, svmMatrixCopy.weightMatrices, svmMatrix.intercepts, svmMatrixCopy.intercepts, sequence, dictionaryClass, predictedClasses,
						1);
				burstSvm.recursivelyPredictAndFit(burstSvmMatrix.weightMatrices, burstSvmMatrixCopy.weightMatrices, burstSvmMatrix.intercepts, 
					burstSvmMatrixCopy.intercepts, burstSequence, burstLength,
					predictedBurstLengths,
					1);
			}
			// If there has been a miss, prepare a blank new input buffer entry:
			else {
				inputBufferEntry.lruCounter = 1;
				for (int i = 0; i < SEQUENCE_LENGTH; i++) {
				#pragma HLS UNROLL
					sequence[i] = NUM_CLASSES_INCLUDING_NULL;
					updatedSequence[i] = NUM_CLASSES_INCLUDING_NULL;
					burstSequence[i] = NUM_BURST_CLASSES_INCLUDING_NULL;
					burstUpdatedSequence[i] = NUM_BURST_CLASSES_INCLUDING_NULL;
				}

				/* HABRIA QUE PONER ESTO AQUI
					* forwardingBuffer.write(forwardingBufferEntriesMatrix.entries, memoryAddress, updatedSequence, inputBufferAddress,
						forwardingBufferCurrentSlot, forwardingBufferNextSlot);
					*/

				// 6) Update the input buffer with the entry:
				for (int i = 0; i < SEQUENCE_LENGTH; i++) {
					#pragma HLS UNROLL
					inputBufferEntry.sequence[i] = updatedSequence[i];
					inputBufferEntry.burstLengthSequence[i] = updatedSequence[i];
				}

				inputBufferEntry.tag = tag;
				inputBufferEntry.valid = true;
				inputBufferEntry.lastAddress = memoryAddress;
				bool isInputBufferHitDummy;
				inputBuffer(inputBufferEntriesMatrix.entries, inputBufferAddress, inputBufferEntry, false, isInputBufferHitDummy, index, way);

				// 4) Predict with the SVM:
				predictedClasses[0] = svm.predict(svmMatrixCopy.weightMatrices, svmMatrixCopy.intercepts, sequence);
				predictedBurstLengths[0] = burstSvm.predict(burstSvmMatrixCopy.weightMatrices, burstSvmMatrixCopy.intercepts,
					burstSequence);

			}


			// 5) Get the finally predicted address:
			dic_index_t dummyIndex;
			delta_t predictedDelta;

			predictedDelta = dictionaryEntriesMatrix.entries[(int)predictedClasses[0]].delta;
			block_address_t predictedAddress = ((delta_t)memoryAddress + predictedDelta);

			// 2) If the predictedAddress is equal to the current, increment the confidence (decrease otherwise):
			BurstConfidenceBufferEntry<ib_confidence_t, block_address_t, burst_length_t> confidenceBufferEntry;
			confidenceBufferEntry =
					confidenceBuffer.read(confidenceBufferEntriesMatrix.entries, index, way);

			// 2.5) Confidence forwarding buffer is read:
			bool isConfidenceForwardingBufferHit;
			BurstConfidenceForwardingBufferEntry<address_t, ib_confidence_t, block_address_t, burst_length_t> confidenceForwardingBufferEntry;

			ib_confidence_t confidence;
			block_address_t lastPredictedAddress;
			ib_confidence_t burstConfidence;
			block_address_t lastPredictedBurst;


			if(isInputBufferHit
					&& predictedAddress){ // Dummy evaluation just to force the scheduling to delay the confidence update

				confidenceForwardingBufferEntry = confidenceForwardingBuffer.read(confidenceForwardingBufferEntriesMatrix.entries, inputBufferAddress,
						confidenceForwardingBufferCurrentSlot, isConfidenceForwardingBufferHit);
				if(isConfidenceForwardingBufferHit){
					confidence = confidenceForwardingBufferEntry.confidence;
					lastPredictedAddress = confidenceForwardingBufferEntry.lastPredictedAddress;
					burstConfidence = confidenceForwardingBufferEntry.burstConfidence;
					lastPredictedBurst = confidenceForwardingBufferEntry.lastPredictedBurstLength;
				}
				else{
					confidence = confidenceBufferEntry.confidence;
					lastPredictedAddress = confidenceBufferEntry.lastPredictedAddress;
					burstConfidence = confidenceBufferEntry.burstConfidence;
					lastPredictedBurst = confidenceBufferEntry.lastPredictedBurstLength;
				}

				bool addressHit = lastPredictedAddress == memoryAddress;
				bool burstHit = lastPredictedBurst == burstLength;

				// 3) If the predictedAddress is equal to the current, increment the confidence (decrease otherwise):
				if (addressHit) {

					if (confidence >= (MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_INCREASE))
						confidence = MAX_PREDICTION_CONFIDENCE;
					else
						confidence += PREDICTION_CONFIDENCE_INCREASE;

				}
				else {

					if (confidence <= (-PREDICTION_CONFIDENCE_DECREASE))
						confidence = 0;
					else
						confidence += PREDICTION_CONFIDENCE_DECREASE;

				}

				if (burstHit) {

					if (burstConfidence >= (MAX_PREDICTION_CONFIDENCE - BURST_PREDICTION_CONFIDENCE_INCREASE))
						burstConfidence = MAX_PREDICTION_CONFIDENCE;
					else
						burstConfidence += BURST_PREDICTION_CONFIDENCE_INCREASE;

				}
				else {

					if (burstConfidence <= (-BURST_PREDICTION_CONFIDENCE_DECREASE))
						burstConfidence = 0;
					else
						burstConfidence += BURST_PREDICTION_CONFIDENCE_DECREASE;

				}
				

				if(!isConfidenceForwardingBufferHit){
					confidenceForwardingBuffer.write(confidenceForwardingBufferEntriesMatrix.entries, confidence, predictedAddress,
							burstConfidence, lastPredictedBurst, 
							inputBufferAddress,
							confidenceForwardingBufferCurrentSlot, confidenceForwardingBufferNextSlot);
				}
				else{
					confidenceForwardingBufferEntriesMatrix.entries[confidenceForwardingBufferCurrentSlot].confidence = confidence;
					confidenceForwardingBufferEntriesMatrix.entries[confidenceForwardingBufferCurrentSlot].lastPredictedAddress = predictedAddress;
					confidenceForwardingBufferEntriesMatrix.entries[confidenceForwardingBufferCurrentSlot].confidence = burstConfidence;
					confidenceForwardingBufferEntriesMatrix.entries[confidenceForwardingBufferCurrentSlot].lastPredictedAddress = predictedBurstLengths[0];
				}

			}
			else{
				// 3) Reset the confidence:
				confidence = 0;
			}


			// 6) Update the confidence buffer with the entry:

			confidenceBufferEntry.confidence = confidence;
			confidenceBufferEntry.lastPredictedAddress = predictedAddress;
			confidenceBufferEntry.burstConfidence = burstConfidence;
			confidenceBufferEntry.lastPredictedBurstLength = lastPredictedBurst;

			confidenceBuffer.write(confidenceBufferEntriesMatrix.entries, index, way, confidenceBufferEntry);

			performPrefetch = isInputBufferHit && confidence >= PREDICTION_CONFIDENCE_THRESHOLD;
			performBurstPrefetch = isInputBufferHit && burstConfidence >= BURST_PREDICTION_CONFIDENCE_THRESHOLD;

			// 7) Select the predicted address to prefetch:
			if(performPrefetch){
				outputAddress = predictedAddress;
				if(performBurstPrefetch)
					outputBurstLength = predictedBurstLengths[0];
				else
					outputBurstLength = 1;
			}
			else{
				outputAddress = 0;
				outputBurstLength = 0;

			}

		}
	}
};
