#pragma once

template<typename weight_t>
struct weigth_matrix_t{
	weight_t weights[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL];
};

template<typename weight_t, typename class_t, typename distance_t>
class SVM {
protected:

	void encodeInOneHot(class_t classSequence[SEQUENCE_LENGTH], bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL]);
	void fit(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t input[SEQUENCE_LENGTH], class_t target);
	
	distance_t distanceToHyperplane(weigth_matrix_t<weight_t> weight_matrix, weight_t intercept, class_t sequence[SEQUENCE_LENGTH]);
public:

	class_t predict(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t input[SEQUENCE_LENGTH]);
	class_t fitAndPredict(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t input[SEQUENCE_LENGTH], class_t target);
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

template<typename weight_t, typename class_t, typename distance_t>
distance_t SVM<weight_t, class_t, distance_t>::distanceToHyperplane(weigth_matrix_t<weight_t> weight_matrix, weight_t intercept, class_t sequence[SEQUENCE_LENGTH]) {
#pragma HLS INLINE

	distance_t res = 0;
	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
#pragma HLS UNROLL
		class_t correspondingClass = sequence[i];
		res += weight_matrix.weights[i][correspondingClass];
	}
	res -= intercept;
	return res;
}

template<typename weight_t, typename class_t, typename distance_t>
class_t SVM<weight_t, class_t, distance_t>::predict(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t sequence [SEQUENCE_LENGTH]) {
#pragma HLS PIPELINE

	class_t res = 0;
	distance_t minDistance = distanceToHyperplane(weight_matrices[0], intercepts[0], sequence); // oneHotSequence);
	for (int c = 1; c < NUM_CLASSES; c++) {
#pragma HLS UNROLL
		distance_t distance = distanceToHyperplane(weight_matrices[c], intercepts[c], sequence); // oneHotSequence);
		if (distance < minDistance) {
			minDistance = distance;
			res = c;
		}
	}
	return res;
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

template<typename weight_t, typename class_t, typename distance_t>
class_t SVM<weight_t, class_t, distance_t>::fitAndPredict(weigth_matrix_t<weight_t> weight_matrices[NUM_CLASSES], weight_t intercepts[NUM_CLASSES], class_t input[SEQUENCE_LENGTH], class_t target) {
#pragma HLS PIPELINE
	class_t res;

	fit(weight_matrices, intercepts, input, target);
	class_t newInput[SEQUENCE_LENGTH];
	for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
#pragma HLS UNROLL
		newInput[i] = input[i + 1];
	}
	newInput[SEQUENCE_LENGTH - 1] = target;

	res = predict(weight_matrices, intercepts, newInput);
	
	return res;
}
