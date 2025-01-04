#pragma once

template<typename weight_t>
struct WeightMatrix{
	weight_t weights[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL];
	WeightMatrix(){}
};

template<typename class_t>
struct InputSequence{
	class_t sequence[SEQUENCE_LENGTH];
	InputSequence(){}
};

template<typename weight_t>
struct SVMWholeMatrix{
	WeightMatrix<weight_t> weightMatrices[NUM_CLASSES];
	weight_t intercepts[NUM_CLASSES];
	SVMWholeMatrix(){}
};


template<typename weight_t, typename class_t, typename distance_t>
class SVM {
protected:

	void fitPredictedHyperplane(WeightMatrix<weight_t>& weight_matrix, weight_t& intercept, class_t oneHotSequence[SEQUENCE_LENGTH]);// [NUM_CLASSES_INCLUDING_NULL]);
	void fitTargetHyperplane(WeightMatrix<weight_t>& weight_matrix, weight_t& intercept, class_t oneHotSequence[SEQUENCE_LENGTH]);// [NUM_CLASSES_INCLUDING_NULL]);
	void encodeInOneHot(class_t classSequence[SEQUENCE_LENGTH], bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL]);
	class_t predict_(WeightMatrix<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t input[SEQUENCE_LENGTH]);
	
	distance_t distanceToHyperplane(WeightMatrix<weight_t>& weight_matrix, weight_t intercept, class_t sequence[SEQUENCE_LENGTH]);
public:
	void fit(WeightMatrix<weight_t> weight_matrices[NUM_CLASSES], WeightMatrix<weight_t> weight_matrices_copy[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], weight_t intercepts_copy[NUM_CLASSES], class_t input[SEQUENCE_LENGTH], const class_t target);
	class_t predict(WeightMatrix<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t input[SEQUENCE_LENGTH]);
	class_t predictAndFit(WeightMatrix<weight_t> weight_matrices[NUM_CLASSES], WeightMatrix<weight_t> weight_matrices_copy[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], weight_t intercepts_copy[NUM_CLASSES], class_t input[SEQUENCE_LENGTH], class_t target);
	void recursivelyPredictAndFit(WeightMatrix<weight_t> weight_matrices[NUM_CLASSES], WeightMatrix<weight_t> weight_matrices_copy[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], weight_t intercepts_copy[NUM_CLASSES], class_t input[SEQUENCE_LENGTH], class_t target,
			class_t outputs[MAX_PREFETCHING_DEGREE], int numPredictions);

};

template<typename weight_t, typename class_t, typename distance_t>
void SVM<weight_t, class_t, distance_t>::encodeInOneHot(class_t classSequence[SEQUENCE_LENGTH], bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL]) {
// #pragma HLS INLINE
#pragma HLS PIPELINE
	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
	#pragma HLS UNROLL
			for (class_t j = 0; j < SEQUENCE_LENGTH; j++) {
				#pragma HLS UNROLL
				oneHotSequence[i][j] = classSequence[i] == j;
			}
	}
}

template<typename weight_t, typename class_t>
weight_t selectWeight(weight_t weights[NUM_CLASSES_INCLUDING_NULL], class_t correspondingClass){
#pragma HLS INLINE
	return weights[correspondingClass];
}

template<typename weight_t, typename class_t, typename distance_t>
distance_t SVM<weight_t, class_t, distance_t>::distanceToHyperplane(WeightMatrix<weight_t>& weight_matrix, weight_t intercept, class_t sequence[SEQUENCE_LENGTH]) {
#pragma HLS BIND_STORAGE variable=weight_matrix type=RAM_T2P impl=bram latency=1
#pragma HLS ARRAY_PARTITION variable=weight_matrix.weights dim=1 complete

#pragma HLS DEPENDENCE array false inter variable=weight_matrix.weights

#pragma HLS INLINE
// #pragma HLS PIPELINE
	distance_t res = -intercept;

	distance_t selectedWeights[SEQUENCE_LENGTH];
#pragma HLS ARRAY_PARTITION variable=selectedWeights dim=1 complete

	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
	#pragma HLS UNROLL
			class_t correspondingClass = sequence[i];
			// selectedWeights[i] = selectWeight<weight_t, class_t>(weight_matrix.weights[i], correspondingClass);
			selectedWeights[i] = weight_matrix.weights[i][correspondingClass];
		}

	/*
	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
#pragma HLS UNROLL
		res += selectedWeights[i];
	}
	*/

	distance_t weights_[SEQUENCE_LENGTH / 2];

	for(int i = 0; i < SEQUENCE_LENGTH; i+=2){
#pragma HLS UNROLL

		if(i != (SEQUENCE_LENGTH - 1)){
			weights_[i >> 1] = selectedWeights[i] + selectedWeights[i+1];
		}
	}

	for(int i = 0; i < SEQUENCE_LENGTH / 2; i++){
#pragma HLS UNROLL

		res += weights_[i];
	}
	if(SEQUENCE_LENGTH % 2 == 1){
		res += selectedWeights[SEQUENCE_LENGTH - 1];
	}


	return res;
}

template<typename weight_t, typename class_t, typename distance_t>
class_t SVM<weight_t, class_t, distance_t>::predict_(WeightMatrix<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t sequence [SEQUENCE_LENGTH]) {
#pragma HLS ARRAY_PARTITION variable=weight_matrices complete
#pragma HLS ARRAY_PARTITION variable=weight_matrices->weights dim=1 complete
#pragma HLS BIND_STORAGE variable=weight_matrices->weights type=RAM_T2P impl=bram latency=1
#pragma HLS ARRAY_PARTITION variable=intercepts complete
#pragma HLS ARRAY_PARTITION variable=sequence complete

	#pragma HLS INLINE
// #pragma HLS PIPELINE

	class_t res = 0;

	distance_t distances[NUM_CLASSES];
#pragma HLS ARRAY_PARTITION variable=distances complete

	for (int c = 0; c < NUM_CLASSES; c++) {
#pragma HLS UNROLL
		distances[c] = distanceToHyperplane(weight_matrices[c], intercepts[c], sequence);

	}


	class_t indexes[NUM_CLASSES / 2];
#pragma HLS ARRAY_PARTITION variable=indexes complete
	distance_t minDistances[NUM_CLASSES / 2];
#pragma HLS ARRAY_PARTITION variable=minDistances complete
	for (int c = 0; c < NUM_CLASSES; c+=2) {
		#pragma HLS UNROLL
		if(c != (NUM_CLASSES - 1)){
			bool lower = distances[c] < distances[c+1];
			indexes[c >> 1] = lower? c : c+1;
			minDistances[c >> 1] = lower? distances[c] : distances[c+1];
		}

	}
	res = indexes[0];
	distance_t minDistance = minDistances[0];
	for (int c = 0; c < NUM_CLASSES / 2; c++) {
		#pragma HLS UNROLL
		if (minDistances[c] < minDistance) {
			res = indexes[c];
			minDistance = minDistances[c];
		}
	}
	if(NUM_CLASSES % 2 == 1){
		if (distances[NUM_CLASSES - 1] < minDistance) {
			res = NUM_CLASSES - 1;
		}
	}



	return res;
}


template<typename weight_t, typename class_t, typename distance_t>
class_t SVM<weight_t, class_t, distance_t>::predict(WeightMatrix<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t sequence[SEQUENCE_LENGTH]) {
	// #pragma HLS INLINE
	#pragma HLS PIPELINE
	return predict_(weight_matrices, intercepts, sequence);
}

template<typename weight_t, typename class_t, typename distance_t>
void SVM<weight_t, class_t, distance_t>::fitPredictedHyperplane(WeightMatrix<weight_t>& weight_matrix, weight_t& intercept, class_t sequence[SEQUENCE_LENGTH]){// [NUM_CLASSES_INCLUDING_NULL]){
 // #pragma HLS INLINE
	bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL];
	#pragma HLS ARRAY_PARTITION variable=oneHotSequence complete
		encodeInOneHot(sequence, oneHotSequence);
	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
			#pragma HLS UNROLL
				for(int j = 0; j < NUM_CLASSES_INCLUDING_NULL; j++){
	#pragma HLS UNROLL
					weight_matrix.weights[i][j] += (weight_t)
							oneHotSequence[i][j];
				}
			}


			intercept--;
}
template<typename weight_t, typename class_t, typename distance_t>
void SVM<weight_t, class_t, distance_t>::fitTargetHyperplane(WeightMatrix<weight_t>& weight_matrix, weight_t& intercept, class_t sequence[SEQUENCE_LENGTH]){// [NUM_CLASSES_INCLUDING_NULL]){
// #pragma HLS INLINE
	bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL];
	#pragma HLS ARRAY_PARTITION variable=oneHotSequence complete
		encodeInOneHot(sequence, oneHotSequence);
	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
			#pragma HLS UNROLL
				for(int j = 0; j < NUM_CLASSES_INCLUDING_NULL; j++){
	#pragma HLS UNROLL
					weight_matrix.weights[i][j] -= (weight_t)
							oneHotSequence[i][j];
				}
			}


			intercept++;
}

template<typename weight_t, typename class_t, typename distance_t>
void SVM<weight_t, class_t, distance_t>::fit(WeightMatrix<weight_t> weight_matrices[NUM_CLASSES], WeightMatrix<weight_t> weight_matrices_copy[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], weight_t intercepts_copy[NUM_CLASSES], class_t sequence[SEQUENCE_LENGTH], const class_t target) {
#pragma HLS ARRAY_PARTITION variable=weight_matrices complete
// #pragma HLS ARRAY_PARTITION variable=weight_matrices->weights dim=0 complete
#pragma HLS BIND_STORAGE variable=weight_matrices->weights type=RAM_T2P impl=bram latency=1

#pragma HLS ARRAY_PARTITION variable=intercepts complete

#pragma HLS ARRAY_PARTITION variable=weight_matrices_copy complete
// #pragma HLS ARRAY_PARTITION variable=weight_matrices->weights dim=0 complete
#pragma HLS BIND_STORAGE variable=weight_matrices_copy->weights type=RAM_T2P impl=bram latency=1

#pragma HLS ARRAY_PARTITION variable=intercepts_copy complete

#pragma HLS ARRAY_PARTITION variable=sequence complete
#pragma HLS DEPENDENCE dependent=false type=inter variable=weight_matrices
#pragma HLS DEPENDENCE dependent=false type=inter variable=weight_matrices->weights
#pragma HLS DEPENDENCE dependent=false type=inter variable=intercepts
#pragma HLS DEPENDENCE dependent=false type=inter variable=weight_matrices_copy
#pragma HLS DEPENDENCE dependent=false type=inter variable=weight_matrices_copy->weights
#pragma HLS DEPENDENCE dependent=false type=inter variable=intercepts_copy

	#pragma HLS INLINE
// #pragma HLS PIPELINE

	/*
	bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL];
#pragma HLS ARRAY_PARTITION variable=oneHotSequence complete
	encodeInOneHot(sequence, oneHotSequence);
	*/

	WeightMatrix<weight_t> weight_matrices_[NUM_CLASSES];
	weight_t intercepts_[NUM_CLASSES];

	/*
	for(int k = 0; k < NUM_CLASSES; k++){
#pragma HLS UNROLL
		weight_matrices_[k] = weight_matrices[k];
		intercepts_[k] = intercepts[k];
	}
	*/

	class_t predictedClass = predict_(weight_matrices, intercepts, sequence);

	/*
	WeightMatrix<weight_t> weight_matrix_predicted_class = weight_matrices_[(int)predictedClass];
	weight_t intercept_predicted_class = intercepts_[(int)predictedClass];
	WeightMatrix<weight_t> weight_matrix_target_class = weight_matrices_[(int)target];
	weight_t intercept_target_class = intercepts_[(int)target];
	*/

	// WeightMatrix<ap_int<2>> weight_matrices_delta[NUM_CLASSES];
	// ap_int<2> intercepts_delta[NUM_CLASSES];

	/*
	for(int k = 0; k < NUM_CLASSES; k++){
		bool classIsTargetOrPredicted = k == target || k == predictedClass;
		#pragma HLS UNROLL
		for (int i = 0; i < SEQUENCE_LENGTH; i++) {
		#pragma HLS UNROLL
			for(int j = 0; j < NUM_CLASSES_INCLUDING_NULL; j++){
			#pragma HLS UNROLL
				weight_matrices_delta[k].weights[i][j] = k == target? -oneHotSequence[i][j] : 0;
				// weight_matrices_delta[k].weights[i][j] = k == predictedClass? (ap_int<2>) +oneHotSequence[i][j] : weight_matrices_delta[k].weights[i][j];
			}
		}
		intercepts_delta[k] = k == target? +1 : 0;
		// intercepts_delta[k] = k == predictedClass? (ap_int<2>)-1 : intercepts_delta[k];

	}
	*/


	if(target != predictedClass){
		/*
		for(int k = 0; k < NUM_CLASSES; k++){
				#pragma HLS UNROLL
				for (int i = 0; i < SEQUENCE_LENGTH; i++) {
					#pragma HLS UNROLL
						for(int j = 0; j < NUM_CLASSES_INCLUDING_NULL; j++){
						#pragma HLS UNROLL
							weight_matrices[k].weights[i][j] += weight_matrices_delta[k].weights[i][j];
					}
				}
				intercepts[k] += intercepts_delta[k];
			}
		*/

		// fitPredictedHyperplane(weight_matrix_predicted_class, intercept_predicted_class, sequence);
		// fitTargetHyperplane(weight_matrix_target_class, intercept_target_class, sequence);

		/*
			for (int i = 0; i < SEQUENCE_LENGTH; i++) {
			#pragma HLS UNROLL
				weight_matrices[predictedClass].weights[i][sequence[i]]++;
			}


		intercepts[predictedClass]--;



			for (int i = 0; i < SEQUENCE_LENGTH; i++) {
			#pragma HLS UNROLL
				weight_matrices[target].weights[i][sequence[i]]--;
			}


		intercepts[target]++;
	*/
		for(int k = 0; k < NUM_CLASSES; k++){
			#pragma HLS UNROLL
			for (int i = 0; i < SEQUENCE_LENGTH; i++) {
				#pragma HLS UNROLL
					if(k == predictedClass){
						weight_matrices[k].weights[i][sequence[i]]++;
						weight_matrices_copy[k].weights[i][sequence[i]] = weight_matrices[k].weights[i][sequence[i]];
					}
					else if(k == target){
						weight_matrices[k].weights[i][sequence[i]]--;
						weight_matrices_copy[k].weights[i][sequence[i]] = weight_matrices[k].weights[i][sequence[i]];
				}
			}
			if(k == predictedClass){
				intercepts[k]--;
				intercepts_copy[k] = intercepts[k];
			}
			else if(k == target){
				intercepts[k]++;
				intercepts_copy[k] = intercepts[k];
			}
		}

	}

	/*
	weight_matrices[predictedClass] = weight_matrix_predicted_class;
	intercepts[predictedClass] = intercept_predicted_class;
	weight_matrices[target] = weight_matrix_target_class;
	intercepts[target] = intercept_target_class;
	*/


}


template<typename weight_t, typename class_t, typename distance_t>
class_t SVM<weight_t, class_t, distance_t>::predictAndFit(WeightMatrix<weight_t> weight_matrices[NUM_CLASSES], WeightMatrix<weight_t> weight_matrices_copy[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], weight_t intercepts_copy[NUM_CLASSES], class_t input[SEQUENCE_LENGTH], class_t target) {
	// #pragma HLS INLINE
// #pragma HLS PIPELINE
// #pragma HLS ARRAY_PARTITION variable=weight_matrices->weights dim=0 complete
#pragma HLS ARRAY_PARTITION variable=weight_matrices complete
#pragma HLS BIND_STORAGE variable=weight_matrices->weights type=RAM_T2P impl=bram latency=1
#pragma HLS ARRAY_PARTITION variable=intercepts complete
#pragma HLS ARRAY_PARTITION variable=input complete

	class_t res;

	class_t newInput[SEQUENCE_LENGTH];
	for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
#pragma HLS UNROLL
		newInput[i] = input[i + 1];
	}
	newInput[SEQUENCE_LENGTH - 1] = target;

	res = predict_(weight_matrices_copy, intercepts_copy, newInput);
	fit(weight_matrices, weight_matrices_copy, intercepts, intercepts_copy, input, target);


	return res;
}

template<typename weight_t, typename class_t, typename distance_t>
void SVM<weight_t, class_t, distance_t>::recursivelyPredictAndFit(WeightMatrix<weight_t> weight_matrices[NUM_CLASSES], WeightMatrix<weight_t> weight_matrices_copy[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], weight_t intercepts_copy[NUM_CLASSES], class_t input[SEQUENCE_LENGTH], class_t target,
		class_t outputs[MAX_PREFETCHING_DEGREE], int numPredictions) {
	#pragma HLS INLINE
// #pragma HLS ARRAY_PARTITION variable=weight_matrices complete
#pragma HLS ARRAY_PARTITION variable=weight_matrices complete
// #pragma HLS ARRAY_PARTITION variable=weight_matrices->weights dim=0 complete
#pragma HLS ARRAY_PARTITION variable=intercepts complete
#pragma HLS ARRAY_PARTITION variable=input complete
#pragma HLS ARRAY_PARTITION variable=outputs complete
#pragma HLS DEPENDENCE dependent=false type=inter variable=weight_matrices
#pragma HLS DEPENDENCE dependent=false type=inter variable=weight_matrices->weights
#pragma HLS DEPENDENCE dependent=false type=inter variable=intercepts
// #pragma HLS PIPELINE

	/*
	WeightMatrix<weight_t> weight_matrices_[NUM_CLASSES];
		weight_t intercepts_[NUM_CLASSES];
#pragma HLS DEPENDENCE dependent=false type=inter variable=weight_matrices_
#pragma HLS DEPENDENCE dependent=false type=inter variable=weight_matrices_->weights
#pragma HLS DEPENDENCE dependent=false type=inter variable=intercepts_
		for(int k = 0; k < NUM_CLASSES; k++){
	#pragma HLS UNROLL
			weight_matrices_[k] = weight_matrices[k];
			intercepts_[k] = intercepts[k];
		}
		*/

	class_t newInput[SEQUENCE_LENGTH];
	for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
#pragma HLS UNROLL
		newInput[i] = input[i + 1];
	}
	newInput[SEQUENCE_LENGTH - 1] = target;

	for(int k = 0; k < MAX_PREFETCHING_DEGREE; k++){
#pragma HLS UNROLL
		if(k < numPredictions){
			class_t res;
			res = predict_(weight_matrices_copy, intercepts_copy, newInput);
			outputs[k] = res;
			for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
			#pragma HLS UNROLL
					newInput[i] = input[i + 1];
				}
			newInput[SEQUENCE_LENGTH - 1] = res;

		}

	}

	fit(weight_matrices, weight_matrices_copy, intercepts, intercepts_copy, input, target);



}


