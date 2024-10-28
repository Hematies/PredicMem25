#pragma once

template<typename weight_t>
struct weigth_matrix_t{
	weight_t weights[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL];
};

template<typename class_t>
struct input_sequence_t{
	class_t sequence[SEQUENCE_LENGTH];
};


template<typename weight_t, typename class_t, typename distance_t>
class SVM {
protected:

	void encodeInOneHot(class_t classSequence[SEQUENCE_LENGTH], bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL]);
	void fit(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t input[SEQUENCE_LENGTH], class_t target);
	class_t predict_(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t input[SEQUENCE_LENGTH]);
	
	distance_t distanceToHyperplane(weigth_matrix_t<weight_t> weight_matrix, weight_t intercept, class_t sequence[SEQUENCE_LENGTH]);
public:

	class_t predict(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t input[SEQUENCE_LENGTH]);
	class_t fitAndPredict(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t input[SEQUENCE_LENGTH], class_t target);
	void fitAndRecursivelyPredict(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t input[SEQUENCE_LENGTH], class_t target,
			class_t outputs[MAX_PREFETCHING_DEGREE], int numPredictions);

};

template<typename weight_t, typename class_t, typename distance_t>
void SVM<weight_t, class_t, distance_t>::encodeInOneHot(class_t classSequence[SEQUENCE_LENGTH], bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL]) {
#pragma HLS INLINE
	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
#pragma HLS UNROLL
		for (int j = 0; j < NUM_CLASSES_INCLUDING_NULL; j++) {
#pragma HLS UNROLL
			oneHotSequence[i][j] = classSequence[i] == j;
		}
	}
}

template<typename weight_t, typename class_t>
weight_t selectWeight(weight_t weights[NUM_CLASSES_INCLUDING_NULL], class_t correspondingClass){
	return weights[correspondingClass];
}

template<typename weight_t, typename class_t, typename distance_t>
distance_t SVM<weight_t, class_t, distance_t>::distanceToHyperplane(weigth_matrix_t<weight_t> weight_matrix, weight_t intercept, class_t sequence[SEQUENCE_LENGTH]) {
#pragma HLS ARRAY_PARTITION variable=weight_matrix->weights dim=0 complete
// #pragma HLS DEPENDENCE array false inter variable=weight_matrices->weights

#pragma HLS INLINE
// #pragma HLS PIPELINE
	distance_t res = 0;

	distance_t selectedWeights[SEQUENCE_LENGTH];
	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
	#pragma HLS UNROLL
			class_t correspondingClass = sequence[i];
			selectedWeights[i] = selectWeight<weight_t, class_t>(weight_matrix.weights[i], correspondingClass);
		}

	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
#pragma HLS UNROLL
		res += selectedWeights[i];
	}
	res -= intercept;
	return res;
}

template<typename weight_t, typename class_t, typename distance_t>
class_t SVM<weight_t, class_t, distance_t>::predict_(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t sequence [SEQUENCE_LENGTH]) {
#pragma HLS ARRAY_PARTITION variable=weight_matrix complete
#pragma HLS ARRAY_PARTITION variable=weight_matrix->weights dim=0 complete
#pragma HLS ARRAY_PARTITION variable=intercepts complete
#pragma HLS ARRAY_PARTITION variable=sequence complete

	#pragma HLS INLINE
// #pragma HLS PIPELINE

	class_t res = 0;
	/*
	distance_t minDistance = distanceToHyperplane(weight_matrices[0], intercepts[0], sequence); // oneHotSequence);
	for (int c = 1; c < NUM_CLASSES; c++) {
#pragma HLS UNROLL
		distance_t distance = distanceToHyperplane(weight_matrices[c], intercepts[c], sequence); // oneHotSequence);
		if (distance < minDistance) {
			minDistance = distance;
			res = c;
		}
	}
	*/
	distance_t distances[NUM_CLASSES];
	for (int c = 0; c < NUM_CLASSES; c++) {
#pragma HLS UNROLL
		distances[c] = distanceToHyperplane(weight_matrices[c], intercepts[c], sequence);

	}

	distance_t minDistance = distances[0];
	for (int c = 0; c < NUM_CLASSES; c++) {
#pragma HLS UNROLL
		if (distances[c] < minDistance) {
			minDistance = distances[c];
			res = c;
		}
	}

	return res;
}


template<typename weight_t, typename class_t, typename distance_t>
class_t SVM<weight_t, class_t, distance_t>::predict(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t sequence [SEQUENCE_LENGTH]) {
	// #pragma HLS INLINE
	#pragma HLS PIPELINE
	return predict_(weight_matrices, intercepts, sequence);
}


template<typename weight_t, typename class_t, typename distance_t>
void SVM<weight_t, class_t, distance_t>::fit(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t sequence[SEQUENCE_LENGTH], class_t target) {
#pragma HLS INLINE

	for (int c = 0; c < NUM_CLASSES; c++) {
#pragma HLS UNROLL
		distance_t distance = distanceToHyperplane(weight_matrices[c], intercepts[c], sequence);// oneHotSequence);
		bool classIsTarget = target == c;
		bool isPredictionCorrect = classIsTarget ? distance <= -1 : distance >= +1;

		for (int i = 0; i < SEQUENCE_LENGTH; i++) {
#pragma HLS UNROLL
			class_t correspondingClass = sequence[i];
			if (!isPredictionCorrect)
				weight_matrices[c].weights[i][correspondingClass] += (classIsTarget ? -1 : +1) >> SVM_LEARNING_RATE_LOG2;

		}
		if (!isPredictionCorrect)
			intercepts[c] += (classIsTarget ? +1 : -1) >> SVM_LEARNING_RATE_LOG2;
			
	}
}


/*
template<typename weight_t, typename class_t, typename distance_t>
void SVM<weight_t, class_t, distance_t>::fit(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t sequence[SEQUENCE_LENGTH], class_t target) {
#pragma HLS INLINE

	class_t predictedClass = predict_(weight_matrices, intercepts, sequence);

	if(target != predictedClass){
		for (int i = 0; i < SEQUENCE_LENGTH; i++) {
#pragma HLS UNROLL
			class_t correspondingClass = sequence[i];
			weight_matrices[predictedClass].weights[i][correspondingClass] += +1 >> SVM_LEARNING_RATE_LOG2;

		}
		intercepts[predictedClass] += -1 >> SVM_LEARNING_RATE_LOG2;

		for (int i = 0; i < SEQUENCE_LENGTH; i++) {
#pragma HLS UNROLL
			class_t correspondingClass = sequence[i];
			weight_matrices[target].weights[i][correspondingClass] += -1 >> SVM_LEARNING_RATE_LOG2;

		}
		intercepts[target] += +1 >> SVM_LEARNING_RATE_LOG2;

	}

}
*/

template<typename weight_t, typename class_t, typename distance_t>
class_t SVM<weight_t, class_t, distance_t>::fitAndPredict(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t input[SEQUENCE_LENGTH], class_t target) {
	// #pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=weight_matrix complete
// #pragma HLS ARRAY_PARTITION variable=weight_matrix->weights dim=1 block factor=6
#pragma HLS ARRAY_PARTITION variable=weight_matrix->weights complete
#pragma HLS ARRAY_PARTITION variable=intercepts complete
	#pragma HLS PIPELINE
#pragma HLS ARRAY_PARTITION variable=input complete

	class_t res;


	class_t newInput[SEQUENCE_LENGTH];
	for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
#pragma HLS UNROLL
		newInput[i] = input[i + 1];
	}
	newInput[SEQUENCE_LENGTH - 1] = target;

	res = predict_(weight_matrices, intercepts, newInput);
	fit(weight_matrices, intercepts, input, target);

	return res;
}

template<typename weight_t, typename class_t, typename distance_t>
void SVM<weight_t, class_t, distance_t>::fitAndRecursivelyPredict(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t input[SEQUENCE_LENGTH], class_t target,
		class_t outputs[MAX_PREFETCHING_DEGREE], int numPredictions) {
	// #pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=weight_matrix complete
#pragma HLS ARRAY_PARTITION variable=weight_matrix->weights dim=0 complete
#pragma HLS ARRAY_PARTITION variable=intercepts complete
#pragma HLS ARRAY_PARTITION variable=input complete
#pragma HLS ARRAY_PARTITION variable=outputs complete
#pragma HLS INLINE
	#pragma HLS PIPELINE


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
			res = predict_(weight_matrices, intercepts, newInput);
			outputs[k] = res;
			for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
			#pragma HLS UNROLL
					newInput[i] = input[i + 1];
				}
			newInput[SEQUENCE_LENGTH - 1] = res;

		}

	}

	fit(weight_matrices, intercepts, input, target);



}


