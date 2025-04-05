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

	void phase1(address_t inputBufferAddress, block_address_t memoryAddress,
		block_address_t& predictedAddress, 
		ib_index_t& index, ib_way_t& way, bool& nop, bool& isInputBufferHit){

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

		static ForwardingBufferEntriesMatrix<address_t, block_address_t, class_t, ib_confidence_t>
		forwardingBufferEntriesMatrix = initForwardingBufferEntries<address_t, block_address_t, class_t, ib_confidence_t>();
		#pragma HLS ARRAY_RESHAPE variable=forwardingBufferEntriesMatrix.entries complete
		static forwarding_index_t forwardingBufferNextSlot = 0;
		forwarding_index_t forwardingBufferCurrentSlot = 0;

		static SVMWholeMatrix<svm_weight_t> svmMatrix = initSVMData<svm_weight_t>();
			static SVMWholeMatrix<svm_weight_t> svmMatrixCopy = initSVMData<svm_weight_t>();
			#pragma HLS ARRAY_PARTITION variable=svmMatrix.weightMatrices complete
			#pragma HLS ARRAY_PARTITION variable=svmMatrixCopy.weightMatrices complete
			#pragma HLS ARRAY_PARTITION variable=svmMatrix.intercepts complete
			#pragma HLS ARRAY_PARTITION variable=svmMatrixCopy.intercepts complete

		static InputBuffer<address_t, ib_index_t, ib_way_t, ib_tag_t, block_address_t, class_t, ib_lru_t> inputBuffer;
		#pragma HLS DEPENDENCE false variable=inputBuffer

		static ForwardingBuffer<address_t, ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> forwardingBuffer;
		#pragma HLS DEPENDENCE false variable=forwardingBuffer

		static Dictionary<dic_index_t, delta_t, dic_confidence_t> dictionary;
		#pragma HLS DEPENDENCE false variable=dictionary

		static SVM<svm_weight_t, class_t, svm_distance_t> svm;
		#pragma HLS DEPENDENCE false variable=svm

		#pragma HLS PIPELINE

		if(!nop){
			// 1) Input buffer is read:
			InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_lru_t> inputBufferEntryDummy;
			InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_lru_t> inputBufferEntry =
				inputBuffer(inputBufferEntriesMatrix.entries, inputBufferAddress, inputBufferEntryDummy, true, isInputBufferHit,
				index, way);

			// #pragma HLS AGGREGATE variable=inputBufferEntry

			constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
			ib_tag_t tag = inputBufferAddress >> numIndexBits;

			// 1.5) Forwarding buffer is read:
			bool isForwardingBufferHit;
			ForwardingBufferEntry<address_t, block_address_t, class_t, ib_confidence_t> forwardingBufferEntry =
				forwardingBuffer.read(forwardingBufferEntriesMatrix.entries, inputBufferAddress, 
					forwardingBufferCurrentSlot, isForwardingBufferHit);

			block_address_t lastAddress;
			bool resetLruCounter = false;

			// If the forwarding buffer is hit, we obtain the data from it. Else, we allocate a new entry and take the data from the input buffer:
			if(isForwardingBufferHit){
				lastAddress = forwardingBufferEntry.lastAddress;
				for(int k = 0; k < SEQUENCE_LENGTH; k++){
					#pragma HLS UNROLL
					sequence[k] = forwardingBufferEntry.sequence[k];
				}
			}

			else{

				lastAddress = inputBufferEntry.lastAddress;
				for(int k = 0; k < SEQUENCE_LENGTH; k++){
					#pragma HLS UNROLL
					sequence[k] = inputBufferEntry.sequence[k];
				}

			}


			// Skip operation if the previous access is equal to the current:
			nop = inputBufferEntry.lastAddress == memoryAddress;
			if (!nop) {
				class_t predictedClasses[MAX_PREFETCHING_DEGREE];

				bool performPrefetch = false;
				
				if(isInputBufferHit) {
					/*
					if(true){
						lastAddress = inputBufferEntry.lastAddress;
						for(int k = 0; k < SEQUENCE_LENGTH; k++){
							#pragma HLS UNROLL
							sequence[k] = inputBufferEntry.sequence[k];
						}
					}
					*/

					// 3) Compute the resulting delta and its class:
					delta_t delta = (delta_t)memoryAddress - (delta_t)lastAddress;
					class_t dictionaryClass;
					bool dummyIsHit;
					DictionaryEntry < delta_t, dic_confidence_t > dictionaryEntry = dictionary.write(
							dictionaryEntriesMatrix.entries, delta, dictionaryClass, dummyIsHit);
					for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
						#pragma HLS UNROLL
						updatedSequence[i] = sequence[i + 1];
					}
					updatedSequence[SEQUENCE_LENGTH - 1] = dictionaryClass;


					// 3.5 Update the forwarding buffer to include the new sequence and lastAddress: 

					if(!isForwardingBufferHit){
						forwardingBuffer.write(forwardingBufferEntriesMatrix.entries, memoryAddress, updatedSequence, inputBufferAddress,
							forwardingBufferCurrentSlot, forwardingBufferNextSlot); 
					}
					else{
						forwardingBufferEntriesMatrix.entries[forwardingBufferCurrentSlot].lastAddress = memoryAddress;
						for(int k = 0; k < SEQUENCE_LENGTH; k++){
							#pragma HLS UNROLL
							forwardingBufferEntriesMatrix.entries[forwardingBufferCurrentSlot].sequence[k] = updatedSequence[k];
						}
					}

					// 6) Update the input buffer with the entry:
					for (int i = 0; i < SEQUENCE_LENGTH; i++) {
						#pragma HLS UNROLL
						inputBufferEntry.sequence[i] = updatedSequence[i];
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
					svm.recursivelyPredictAndFit(svmMatrix.weightMatrices, svmMatrixCopy.weightMatrices, svmMatrix.intercepts, svmMatrixCopy.intercepts, 
						sequence, updatedSequence[SEQUENCE_LENGTH - 1], predictedClasses,
						1);

				}
				// If there has been a miss, prepare a blank new input buffer entry:
				else {
					inputBufferEntry.lruCounter = 1;
					for (int i = 0; i < SEQUENCE_LENGTH; i++) {
					#pragma HLS UNROLL
						sequence[i] = NUM_CLASSES;
						updatedSequence[i] = NUM_CLASSES;
					}

					/* HABRIA QUE PONER ESTO AQUI
					* forwardingBuffer.write(forwardingBufferEntriesMatrix.entries, memoryAddress, updatedSequence, inputBufferAddress,
							forwardingBufferCurrentSlot, forwardingBufferNextSlot);
					*/

					// 6) Update the input buffer with the entry:
					for (int i = 0; i < SEQUENCE_LENGTH; i++) {
						#pragma HLS UNROLL
						inputBufferEntry.sequence[i] = updatedSequence[i];
					}

					inputBufferEntry.tag = tag;
					inputBufferEntry.valid = true;
					inputBufferEntry.lastAddress = memoryAddress;
					// inputBufferEntry.lastPredictedAddress = predictedAddress;
					// inputBuffer.write(inputBufferEntriesMatrix.entries, inputBufferEntriesMatrixCopy.entries,inputBufferAddress, inputBufferEntry);
					bool isInputBufferHitDummy;
					inputBuffer(inputBufferEntriesMatrix.entries, inputBufferAddress, inputBufferEntry, false, isInputBufferHitDummy, index, way);

					// 4) Predict with the SVM:
					predictedClasses[0] = svm.predict(svmMatrixCopy.weightMatrices, svmMatrixCopy.intercepts, sequence);
				}
				
				class_t predictedClass = predictedClasses[0];

				// 5) Get the finally predicted address:
				dic_index_t dummyIndex;
				delta_t predictedDelta;

				predictedDelta = dictionaryEntriesMatrix.entries[predictedClass].delta;
				predictedAddress = ((delta_t)memoryAddress + predictedDelta);
			}
		}
		
	}

	phase2(address_t inputBufferAddress, block_address_t memoryAddress,
		block_address_t predictedAddress, block_address_t& prefetchAddress,
		ib_index_t index, ib_way_t way, bool nop, bool isInputBufferHit){
		
		static LookUpTable<ib_confidence_t, MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_THRESHOLD + 1>
		confidenceLookUpTable = fillUniformLookUpTable<ib_confidence_t, MAX_PREDICTION_CONFIDENCE - PREDICTION_CONFIDENCE_THRESHOLD + 1>(1, MAX_PREFETCHING_DEGREE);


		static ConfidenceForwardingBufferEntriesMatrix<address_t, ib_confidence_t, block_address_t>
			confidenceForwardingBufferEntriesMatrix = initConfidenceForwardingBufferEntries<address_t, ib_confidence_t, block_address_t>();
		#pragma HLS ARRAY_RESHAPE variable=confidenceForwardingBufferEntriesMatrix.entries complete
		static conf_forwarding_index_t confidenceForwardingBufferNextSlot = 0;
		conf_forwarding_index_t confidenceForwardingBufferCurrentSlot = 0;

		// #pragma HLS DEPENDENCE array false variable=forwardingBufferEntriesMatrix.entries
		static ConfidenceBufferEntriesMatrix<ib_confidence_t, block_address_t>
			confidenceBufferEntriesMatrix = initConfidenceBufferEntries<ib_confidence_t, block_address_t>();
		#pragma HLS ARRAY_RESHAPE variable=confidenceBufferEntriesMatrix.entries dim=2 complete
		#pragma HLS ARRAY_RESHAPE variable=confidenceBufferEntriesMatrix.entries dim=3 complete
		// #pragma HLS ARRAY_PARTITION variable=confidenceBufferEntriesMatrix.entries dim=0 complete
		#pragma HLS BIND_STORAGE variable=confidenceBufferEntriesMatrix.entries type=RAM_T2P impl=bram latency=1

		#pragma HLS DEPENDENCE array false variable=confidenceBufferEntriesMatrix.entries

		static ConfidenceBuffer<ib_confidence_t, block_address_t> confidenceBuffer;
		#pragma HLS DEPENDENCE false variable=confidenceBuffer

		static ConfidenceForwardingBuffer<address_t, ib_confidence_t, block_address_t> confidenceForwardingBuffer;
		#pragma HLS DEPENDENCE false variable=confidenceForwardingBuffer

		#pragma HLS PIPELINE
		int prefetchDegree = 1;
		
		if(!nop) {
			// 2) If the predictedAddress is equal to the current, increment the confidence (decrease otherwise):
			ConfidenceBufferEntry<ib_confidence_t, block_address_t> confidenceBufferEntry;
			confidenceBufferEntry =
					confidenceBuffer.read(confidenceBufferEntriesMatrix.entries, index, way);

			// 2.5) Confidence forwarding buffer is read:
			bool isConfidenceForwardingBufferHit;
			ConfidenceForwardingBufferEntry<address_t, ib_confidence_t, block_address_t> confidenceForwardingBufferEntry;

			ib_confidence_t confidence;
			block_address_t lastPredictedAddress;

			if(isInputBufferHit){ // Dummy evaluation just to force the scheduling to delay the confidence update

				confidenceForwardingBufferEntry = confidenceForwardingBuffer.read(confidenceForwardingBufferEntriesMatrix.entries, inputBufferAddress,
						confidenceForwardingBufferCurrentSlot, isConfidenceForwardingBufferHit);
				if(isConfidenceForwardingBufferHit){
					confidence = confidenceForwardingBufferEntry.confidence;
					lastPredictedAddress = confidenceForwardingBufferEntry.lastPredictedAddress;
				}
				else{
					confidence = confidenceBufferEntry.confidence;
					lastPredictedAddress = confidenceBufferEntry.lastPredictedAddress;
				}

				bool addressHit = lastPredictedAddress == memoryAddress;

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

				if(!isConfidenceForwardingBufferHit){
					confidenceForwardingBuffer.write(confidenceForwardingBufferEntriesMatrix.entries, confidence, predictedAddress, inputBufferAddress,
							confidenceForwardingBufferCurrentSlot, confidenceForwardingBufferNextSlot);
				}
				else{
					confidenceForwardingBufferEntriesMatrix.entries[confidenceForwardingBufferCurrentSlot].confidence = confidence;
					confidenceForwardingBufferEntriesMatrix.entries[confidenceForwardingBufferCurrentSlot].lastPredictedAddress = predictedAddress;
				}

			}
			else{
				// 3) Reset the confidence:
				confidence = 0;
			}


			// 6) Update the confidence buffer with the entry:

			confidenceBufferEntry.confidence = confidence;
			confidenceBufferEntry.lastPredictedAddress = predictedAddress;
			confidenceBuffer.write(confidenceBufferEntriesMatrix.entries, index, way, confidenceBufferEntry);

			bool performPrefetch = isInputBufferHit && confidence >= PREDICTION_CONFIDENCE_THRESHOLD;

			// 7) Select the predicted address to prefetch:
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

			prefetchAddress = addressesToPrefetch[0];
		}
		else{
			prefetchAddress = 0;
		}
	
	}

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

	static ForwardingBufferEntriesMatrix<address_t, block_address_t, class_t, ib_confidence_t>
		forwardingBufferEntriesMatrix = initForwardingBufferEntries<address_t, block_address_t, class_t, ib_confidence_t>();
	#pragma HLS ARRAY_RESHAPE variable=forwardingBufferEntriesMatrix.entries complete
	static forwarding_index_t forwardingBufferNextSlot = 0;
	forwarding_index_t forwardingBufferCurrentSlot = 0;

	static ConfidenceForwardingBufferEntriesMatrix<address_t, ib_confidence_t, block_address_t>
			confidenceForwardingBufferEntriesMatrix = initConfidenceForwardingBufferEntries<address_t, ib_confidence_t, block_address_t>();
		#pragma HLS ARRAY_RESHAPE variable=confidenceForwardingBufferEntriesMatrix.entries complete
		static conf_forwarding_index_t confidenceForwardingBufferNextSlot = 0;
		conf_forwarding_index_t confidenceForwardingBufferCurrentSlot = 0;

	// #pragma HLS DEPENDENCE array false variable=forwardingBufferEntriesMatrix.entries
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
	#pragma HLS ARRAY_PARTITION variable=svmMatrixCopy.intercepts complete

		static InputBuffer<address_t, ib_index_t, ib_way_t, ib_tag_t, block_address_t, class_t, ib_lru_t> inputBuffer;
	#pragma HLS DEPENDENCE false variable=inputBuffer

		static ForwardingBuffer<address_t, ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> forwardingBuffer;
	#pragma HLS DEPENDENCE false variable=forwardingBuffer

		static ConfidenceBuffer<ib_confidence_t, block_address_t> confidenceBuffer;
	#pragma HLS DEPENDENCE false variable=confidenceBuffer

		static ConfidenceForwardingBuffer<address_t, ib_confidence_t, block_address_t> confidenceForwardingBuffer;
	#pragma HLS DEPENDENCE false variable=confidenceForwardingBuffer

		static Dictionary<dic_index_t, delta_t, dic_confidence_t> dictionary;
	#pragma HLS DEPENDENCE false variable=dictionary

		static SVM<svm_weight_t, class_t, svm_distance_t> svm;
	#pragma HLS DEPENDENCE false variable=svm

	#pragma HLS PIPELINE
		int prefetchDegree = 1;

		// 1) Input buffer is read:
		bool isInputBufferHit;
		ib_index_t index;
		ib_way_t way;
		InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_lru_t> inputBufferEntryDummy;
		InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_lru_t> inputBufferEntry =
			inputBuffer(inputBufferEntriesMatrix.entries, inputBufferAddress, inputBufferEntryDummy, true, isInputBufferHit,
			index, way);

	// #pragma HLS AGGREGATE variable=inputBufferEntry

		constexpr auto numIndexBits = NUM_ADDRESS_BITS - IB_NUM_TAG_BITS;
		ib_tag_t tag = inputBufferAddress >> numIndexBits;

		// 1.5) Forwarding buffer is read:
		bool isForwardingBufferHit;
		ForwardingBufferEntry<address_t, block_address_t, class_t, ib_confidence_t> forwardingBufferEntry =
			forwardingBuffer.read(forwardingBufferEntriesMatrix.entries, inputBufferAddress, 
				forwardingBufferCurrentSlot, isForwardingBufferHit);

		block_address_t lastAddress;
		class_t updatedSequence[SEQUENCE_LENGTH];
		class_t sequence[SEQUENCE_LENGTH];
		bool resetLruCounter = false;

		// If the forwarding buffer is hit, we obtain the data from it. Else, we allocate a new entry and take the data from the input buffer:
		if(isForwardingBufferHit){
			lastAddress = forwardingBufferEntry.lastAddress;
			for(int k = 0; k < SEQUENCE_LENGTH; k++){
				#pragma HLS UNROLL
				sequence[k] = forwardingBufferEntry.sequence[k];
			}
		}

		else{

			lastAddress = inputBufferEntry.lastAddress;
			for(int k = 0; k < SEQUENCE_LENGTH; k++){
				#pragma HLS UNROLL
				sequence[k] = inputBufferEntry.sequence[k];
			}

		}
		

		// Skip operation if the previous access is equal to the current:
		if (inputBufferEntry.lastAddress != memoryAddress) {
			class_t predictedClasses[MAX_PREFETCHING_DEGREE];

			bool performPrefetch = false;
			
			if(isInputBufferHit) {
				/*
				if(true){
					lastAddress = inputBufferEntry.lastAddress;
					for(int k = 0; k < SEQUENCE_LENGTH; k++){
						#pragma HLS UNROLL
						sequence[k] = inputBufferEntry.sequence[k];
					}
				}
				*/

				// 3) Compute the resulting delta and its class:
				delta_t delta = (delta_t)memoryAddress - (delta_t)lastAddress;
				class_t dictionaryClass;
				bool dummyIsHit;
				DictionaryEntry < delta_t, dic_confidence_t > dictionaryEntry = dictionary.write(
						dictionaryEntriesMatrix.entries, delta, dictionaryClass, dummyIsHit);
				for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
					#pragma HLS UNROLL
					updatedSequence[i] = sequence[i + 1];
				}
				updatedSequence[SEQUENCE_LENGTH - 1] = dictionaryClass;


				// 3.5 Update the forwarding buffer to include the new sequence and lastAddress: 

				if(!isForwardingBufferHit){
					forwardingBuffer.write(forwardingBufferEntriesMatrix.entries, memoryAddress, updatedSequence, inputBufferAddress,
						forwardingBufferCurrentSlot, forwardingBufferNextSlot); 
				}
				else{
					forwardingBufferEntriesMatrix.entries[forwardingBufferCurrentSlot].lastAddress = memoryAddress;
					for(int k = 0; k < SEQUENCE_LENGTH; k++){
						#pragma HLS UNROLL
						forwardingBufferEntriesMatrix.entries[forwardingBufferCurrentSlot].sequence[k] = updatedSequence[k];
					}
				}

				// 6) Update the input buffer with the entry:
				for (int i = 0; i < SEQUENCE_LENGTH; i++) {
					#pragma HLS UNROLL
					inputBufferEntry.sequence[i] = updatedSequence[i];
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
			}
			// If there has been a miss, prepare a blank new input buffer entry:
			else {
				inputBufferEntry.lruCounter = 1;
				for (int i = 0; i < SEQUENCE_LENGTH; i++) {
				#pragma HLS UNROLL
					sequence[i] = NUM_CLASSES;
					updatedSequence[i] = NUM_CLASSES;
				}

				/* HABRIA QUE PONER ESTO AQUI
				 * forwardingBuffer.write(forwardingBufferEntriesMatrix.entries, memoryAddress, updatedSequence, inputBufferAddress,
						forwardingBufferCurrentSlot, forwardingBufferNextSlot);
				 */

				// 6) Update the input buffer with the entry:
				for (int i = 0; i < SEQUENCE_LENGTH; i++) {
					#pragma HLS UNROLL
					inputBufferEntry.sequence[i] = updatedSequence[i];
				}

				inputBufferEntry.tag = tag;
				inputBufferEntry.valid = true;
				inputBufferEntry.lastAddress = memoryAddress;
				// inputBufferEntry.lastPredictedAddress = predictedAddress;
				// inputBuffer.write(inputBufferEntriesMatrix.entries, inputBufferEntriesMatrixCopy.entries,inputBufferAddress, inputBufferEntry);
				bool isInputBufferHitDummy;
				inputBuffer(inputBufferEntriesMatrix.entries, inputBufferAddress, inputBufferEntry, false, isInputBufferHitDummy, index, way);

				// 4) Predict with the SVM:
				predictedClasses[0] = svm.predict(svmMatrixCopy.weightMatrices, svmMatrixCopy.intercepts, sequence);

			}


			// 5) Get the finally predicted address:
			dic_index_t dummyIndex;
			delta_t predictedDelta;

			predictedDelta = dictionaryEntriesMatrix.entries[(int)predictedClasses[0]].delta;
			block_address_t predictedAddress = ((delta_t)memoryAddress + predictedDelta);

			// 2) If the predictedAddress is equal to the current, increment the confidence (decrease otherwise):
			ConfidenceBufferEntry<ib_confidence_t, block_address_t> confidenceBufferEntry;
			confidenceBufferEntry =
					confidenceBuffer.read(confidenceBufferEntriesMatrix.entries, index, way);

			// 2.5) Confidence forwarding buffer is read:
			bool isConfidenceForwardingBufferHit;
			ConfidenceForwardingBufferEntry<address_t, ib_confidence_t, block_address_t> confidenceForwardingBufferEntry;

			ib_confidence_t confidence;
			block_address_t lastPredictedAddress;


			if(isInputBufferHit
					&& predictedAddress){ // Dummy evaluation just to force the scheduling to delay the confidence update

				confidenceForwardingBufferEntry = confidenceForwardingBuffer.read(confidenceForwardingBufferEntriesMatrix.entries, inputBufferAddress,
						confidenceForwardingBufferCurrentSlot, isConfidenceForwardingBufferHit);
				if(isConfidenceForwardingBufferHit){
					confidence = confidenceForwardingBufferEntry.confidence;
					lastPredictedAddress = confidenceForwardingBufferEntry.lastPredictedAddress;
				}
				else{
					confidence = confidenceBufferEntry.confidence;
					lastPredictedAddress = confidenceBufferEntry.lastPredictedAddress;
				}

				bool addressHit = lastPredictedAddress == memoryAddress;

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

				if(!isConfidenceForwardingBufferHit){
					confidenceForwardingBuffer.write(confidenceForwardingBufferEntriesMatrix.entries, confidence, predictedAddress, inputBufferAddress,
							confidenceForwardingBufferCurrentSlot, confidenceForwardingBufferNextSlot);
				}
				else{
					confidenceForwardingBufferEntriesMatrix.entries[confidenceForwardingBufferCurrentSlot].confidence = confidence;
					confidenceForwardingBufferEntriesMatrix.entries[confidenceForwardingBufferCurrentSlot].lastPredictedAddress = predictedAddress;
				}

			}
			else{
				// 3) Reset the confidence:
				confidence = 0;
			}


			// 6) Update the confidence buffer with the entry:

			confidenceBufferEntry.confidence = confidence;
			confidenceBufferEntry.lastPredictedAddress = predictedAddress;
			confidenceBuffer.write(confidenceBufferEntriesMatrix.entries, index, way, confidenceBufferEntry);

			performPrefetch = isInputBufferHit && confidence >= PREDICTION_CONFIDENCE_THRESHOLD;

			// 7) Select the predicted address to prefetch:
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

		}
	}

};
