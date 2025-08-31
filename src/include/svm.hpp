#pragma once

template<typename weight_t, uint8_t NUM_CLASSES_INCLUDING_NULL__>
struct WeightMatrix{
	weight_t weights[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL__];
	WeightMatrix(){}
};

template<typename class_t>
struct InputSequence{
	class_t sequence[SEQUENCE_LENGTH];
	InputSequence(){}
};

template<typename weight_t>
struct SVMWholeMatrix{
	WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL> weightMatrices[NUM_CLASSES];
	weight_t intercepts[NUM_CLASSES];
	SVMWholeMatrix(){}
};

template<typename weight_t>
struct BurstSVMWholeMatrix{
	WeightMatrix<weight_t, NUM_BURST_CLASSES_INCLUDING_NULL> weightMatrices[NUM_BURST_CLASSES];
	weight_t intercepts[NUM_BURST_CLASSES];
	BurstSVMWholeMatrix(){}
};


template<typename weight_t, typename class_t, typename distance_t, uint8_t NUM_CLASSES__, uint8_t NUM_CLASSES_INCLUDING_NULL__>
class SVM {
protected:

	void fitPredictedHyperplane(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__>& weight_matrix, weight_t& intercept, class_t oneHotSequence[SEQUENCE_LENGTH]);
	void fitTargetHyperplane(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__>& weight_matrix, weight_t& intercept, class_t oneHotSequence[SEQUENCE_LENGTH]);
	void encodeInOneHot(class_t classSequence[SEQUENCE_LENGTH], bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL__]);
	class_t predict_(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices[NUM_CLASSES__], weight_t intercepts[NUM_CLASSES__], class_t input[SEQUENCE_LENGTH]);
	
	distance_t distanceToHyperplane(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__>& weight_matrix, weight_t intercept, class_t sequence[SEQUENCE_LENGTH]);
public:
	void fit(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices[NUM_CLASSES__], WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices_copy[NUM_CLASSES__], weight_t intercepts[NUM_CLASSES__], weight_t intercepts_copy[NUM_CLASSES__], class_t input[SEQUENCE_LENGTH], const class_t target);
	class_t predict(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices[NUM_CLASSES__], weight_t intercepts[NUM_CLASSES__], class_t input[SEQUENCE_LENGTH]);
	class_t predictAndFit(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices[NUM_CLASSES__], WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices_copy[NUM_CLASSES__], weight_t intercepts[NUM_CLASSES__], weight_t intercepts_copy[NUM_CLASSES__], class_t input[SEQUENCE_LENGTH], class_t target);
	void recursivelyPredictAndFit(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices[NUM_CLASSES__], WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices_copy[NUM_CLASSES__], weight_t intercepts[NUM_CLASSES__], weight_t intercepts_copy[NUM_CLASSES__], class_t input[SEQUENCE_LENGTH], class_t target,
			class_t outputs[MAX_PREFETCHING_DEGREE], int numPredictions);

};

template<typename weight_t, typename class_t, typename distance_t, uint8_t NUM_CLASSES__, uint8_t NUM_CLASSES_INCLUDING_NULL__>
void SVM<weight_t, class_t, distance_t, NUM_CLASSES__, NUM_CLASSES_INCLUDING_NULL__>::
encodeInOneHot(class_t classSequence[SEQUENCE_LENGTH], bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL__]) {
#pragma HLS PIPELINE
	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
	#pragma HLS UNROLL
			for (class_t j = 0; j < SEQUENCE_LENGTH; j++) {
				#pragma HLS UNROLL
				oneHotSequence[i][j] = classSequence[i] == j;
			}
	}
}

template<typename weight_t, typename class_t, uint8_t NUM_CLASSES_INCLUDING_NULL__>
weight_t selectWeight(weight_t weights[NUM_CLASSES_INCLUDING_NULL__], class_t correspondingClass){
#pragma HLS INLINE
	return weights[correspondingClass];
}

template<typename weight_t, typename class_t, typename distance_t, uint8_t NUM_CLASSES__, uint8_t NUM_CLASSES_INCLUDING_NULL__>
distance_t SVM<weight_t, class_t, distance_t, NUM_CLASSES__, NUM_CLASSES_INCLUDING_NULL__>::
distanceToHyperplane(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__>& weight_matrix, weight_t intercept, class_t sequence[SEQUENCE_LENGTH]) {
#pragma HLS BIND_STORAGE variable=weight_matrix type=RAM_2P impl=lutram latency=1
#pragma HLS ARRAY_PARTITION variable=weight_matrix.weights dim=1 complete

#pragma HLS DEPENDENCE array false inter variable=weight_matrix.weights

#pragma HLS INLINE
	distance_t res = -intercept;

	distance_t selectedWeights[SEQUENCE_LENGTH];
#pragma HLS ARRAY_PARTITION variable=selectedWeights dim=1 complete

	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
	#pragma HLS UNROLL
		class_t correspondingClass = sequence[i];
		selectedWeights[i] = weight_matrix.weights[i][correspondingClass];
	}

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

template<typename weight_t, typename class_t, typename distance_t, uint8_t NUM_CLASSES__, uint8_t NUM_CLASSES_INCLUDING_NULL__>
class_t SVM<weight_t, class_t, distance_t, NUM_CLASSES__, NUM_CLASSES_INCLUDING_NULL__>::
predict_(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices[NUM_CLASSES__], weight_t intercepts[NUM_CLASSES__], class_t sequence [SEQUENCE_LENGTH]) {
#pragma HLS ARRAY_PARTITION variable=weight_matrices complete
#pragma HLS ARRAY_PARTITION variable=weight_matrices->weights dim=1 complete
#pragma HLS BIND_STORAGE variable=weight_matrices->weights type=RAM_2P impl=lutram latency=1
#pragma HLS ARRAY_PARTITION variable=intercepts complete
#pragma HLS ARRAY_PARTITION variable=sequence complete

	#pragma HLS INLINE

	class_t res = 0;

	distance_t distances[NUM_CLASSES__];
#pragma HLS ARRAY_PARTITION variable=distances complete

	for (int c = 0; c < NUM_CLASSES__; c++) {
#pragma HLS UNROLL
		distances[c] = distanceToHyperplane(weight_matrices[c], intercepts[c], sequence);

	}


	class_t indexes[NUM_CLASSES__ / 2];
#pragma HLS ARRAY_PARTITION variable=indexes complete
	distance_t minDistances[NUM_CLASSES__ / 2];
#pragma HLS ARRAY_PARTITION variable=minDistances complete
	for (int c = 0; c < NUM_CLASSES__; c+=2) {
		#pragma HLS UNROLL
		if(c != (NUM_CLASSES__ - 1)){
			bool lower = distances[c] < distances[c+1];
			indexes[c >> 1] = lower? c : c+1;
			minDistances[c >> 1] = lower? distances[c] : distances[c+1];
		}

	}
	res = indexes[0];
	distance_t minDistance = minDistances[0];
	for (int c = 0; c < NUM_CLASSES__ / 2; c++) {
		#pragma HLS UNROLL
		if (minDistances[c] < minDistance) {
			res = indexes[c];
			minDistance = minDistances[c];
		}
	}
	if(NUM_CLASSES__ % 2 == 1){
		if (distances[NUM_CLASSES__ - 1] < minDistance) {
			res = NUM_CLASSES__ - 1;
		}
	}



	return res;
}


template<typename weight_t, typename class_t, typename distance_t, uint8_t NUM_CLASSES__, uint8_t NUM_CLASSES_INCLUDING_NULL__>
class_t SVM<weight_t, class_t, distance_t, NUM_CLASSES__, NUM_CLASSES_INCLUDING_NULL__>::predict(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices[NUM_CLASSES__], weight_t intercepts[NUM_CLASSES__], class_t sequence[SEQUENCE_LENGTH]) {
	#pragma HLS INLINE
	return predict_(weight_matrices, intercepts, sequence);
}

template<typename weight_t, typename class_t, typename distance_t, uint8_t NUM_CLASSES__, uint8_t NUM_CLASSES_INCLUDING_NULL__>
void SVM<weight_t, class_t, distance_t, NUM_CLASSES__, NUM_CLASSES_INCLUDING_NULL__>::
fitPredictedHyperplane(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__>& weight_matrix, weight_t& intercept, class_t sequence[SEQUENCE_LENGTH]){
	bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL__];
	#pragma HLS ARRAY_PARTITION variable=oneHotSequence complete
		encodeInOneHot(sequence, oneHotSequence);
	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
			#pragma HLS UNROLL
				for(int j = 0; j < NUM_CLASSES_INCLUDING_NULL__; j++){
	#pragma HLS UNROLL
					weight_matrix.weights[i][j] += (weight_t)
							oneHotSequence[i][j];
				}
			}


			intercept--;
}
template<typename weight_t, typename class_t, typename distance_t, uint8_t NUM_CLASSES__, uint8_t NUM_CLASSES_INCLUDING_NULL__>
void SVM<weight_t, class_t, distance_t, NUM_CLASSES__, NUM_CLASSES_INCLUDING_NULL__>::
fitTargetHyperplane(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__>& weight_matrix, weight_t& intercept, class_t sequence[SEQUENCE_LENGTH]){
	bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL__];
	#pragma HLS ARRAY_PARTITION variable=oneHotSequence complete
		encodeInOneHot(sequence, oneHotSequence);
	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
			#pragma HLS UNROLL
				for(int j = 0; j < NUM_CLASSES_INCLUDING_NULL__; j++){
	#pragma HLS UNROLL
					weight_matrix.weights[i][j] -= (weight_t)
							oneHotSequence[i][j];
				}
			}


			intercept++;
}

template<typename weight_t, typename class_t, typename distance_t, uint8_t NUM_CLASSES__, uint8_t NUM_CLASSES_INCLUDING_NULL__>
void SVM<weight_t, class_t, distance_t, NUM_CLASSES__, NUM_CLASSES_INCLUDING_NULL__>::
fit(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices[NUM_CLASSES__], WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices_copy[NUM_CLASSES__], weight_t intercepts[NUM_CLASSES__], weight_t intercepts_copy[NUM_CLASSES__], class_t sequence[SEQUENCE_LENGTH], const class_t target) {
#pragma HLS ARRAY_PARTITION variable=weight_matrices complete
#pragma HLS BIND_STORAGE variable=weight_matrices->weights type=RAM_2P impl=lutram latency=1

#pragma HLS ARRAY_PARTITION variable=intercepts complete

#pragma HLS ARRAY_PARTITION variable=weight_matrices_copy complete
#pragma HLS BIND_STORAGE variable=weight_matrices_copy->weights type=RAM_2P impl=lutram latency=1

#pragma HLS ARRAY_PARTITION variable=intercepts_copy complete

#pragma HLS ARRAY_PARTITION variable=sequence complete
#pragma HLS DEPENDENCE dependent=false type=inter variable=weight_matrices
#pragma HLS DEPENDENCE dependent=false type=inter variable=weight_matrices->weights
#pragma HLS DEPENDENCE dependent=false type=inter variable=intercepts
#pragma HLS DEPENDENCE dependent=false type=inter variable=weight_matrices_copy
#pragma HLS DEPENDENCE dependent=false type=inter variable=weight_matrices_copy->weights
#pragma HLS DEPENDENCE dependent=false type=inter variable=intercepts_copy

	#pragma HLS INLINE

	WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices_[NUM_CLASSES__];
	weight_t intercepts_[NUM_CLASSES__];

	distance_t distances[NUM_CLASSES__];
	#pragma HLS ARRAY_PARTITION variable=distances complete

	for (int c = 0; c < NUM_CLASSES__; c++) {
#pragma HLS UNROLL
		distances[c] = distanceToHyperplane(weight_matrices[c], intercepts[c], sequence);

	}

	for(int k = 0; k < NUM_CLASSES__; k++){
		#pragma HLS UNROLL
		distance_t distance;
		if(k == target){
			distance = (1 << SVM_LEARNING_RATE_LOG2) + distances[k];
		}
		else{
			distance = (1 << SVM_LEARNING_RATE_LOG2) - distances[k];
		}
		if(distance > 0) {
			for (int i = 0; i < SEQUENCE_LENGTH; i++) {
				#pragma HLS UNROLL
				if(k == target){
					weight_matrices[k].weights[i][sequence[i]]--;
					weight_matrices_copy[k].weights[i][sequence[i]] = weight_matrices[k].weights[i][sequence[i]];
				}
				else{
					weight_matrices[k].weights[i][sequence[i]]++;
					weight_matrices_copy[k].weights[i][sequence[i]] = weight_matrices[k].weights[i][sequence[i]];
				}
			}
			if(k == target){
				intercepts[k]++;
				intercepts_copy[k] = intercepts[k];
			}
			else {
				intercepts[k]--;
				intercepts_copy[k] = intercepts[k];
			}
		}

	}



}


template<typename weight_t, typename class_t, typename distance_t, uint8_t NUM_CLASSES__, uint8_t NUM_CLASSES_INCLUDING_NULL__>
class_t SVM<weight_t, class_t, distance_t, NUM_CLASSES__, NUM_CLASSES_INCLUDING_NULL__>::
predictAndFit(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices[NUM_CLASSES__], WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices_copy[NUM_CLASSES__], weight_t intercepts[NUM_CLASSES__], weight_t intercepts_copy[NUM_CLASSES__], class_t input[SEQUENCE_LENGTH], class_t target) {
	
#pragma HLS ARRAY_PARTITION variable=weight_matrices complete
#pragma HLS BIND_STORAGE variable=weight_matrices->weights type=RAM_2P impl=lutram latency=1
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

template<typename weight_t, typename class_t, typename distance_t, uint8_t NUM_CLASSES__, uint8_t NUM_CLASSES_INCLUDING_NULL__>
void SVM<weight_t, class_t, distance_t, NUM_CLASSES__, NUM_CLASSES_INCLUDING_NULL__>::
recursivelyPredictAndFit(WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices[NUM_CLASSES__], WeightMatrix<weight_t, NUM_CLASSES_INCLUDING_NULL__> weight_matrices_copy[NUM_CLASSES__], weight_t intercepts[NUM_CLASSES__], weight_t intercepts_copy[NUM_CLASSES__], class_t input[SEQUENCE_LENGTH], class_t target,
		class_t outputs[MAX_PREFETCHING_DEGREE], int numPredictions) {
	#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=weight_matrices complete
#pragma HLS ARRAY_PARTITION variable=intercepts complete
#pragma HLS ARRAY_PARTITION variable=input complete
#pragma HLS ARRAY_PARTITION variable=outputs complete
#pragma HLS DEPENDENCE dependent=false type=inter variable=weight_matrices
#pragma HLS DEPENDENCE dependent=false type=inter variable=weight_matrices->weights
#pragma HLS DEPENDENCE dependent=false type=inter variable=intercepts

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


