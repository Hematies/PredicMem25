#pragma once

template<typename weight_t, typename class_t, typename distance_t>
class SVM {
protected:
	weight_t *weights;
	weight_t *intercepts;

	void encodeInOneHot(class_t classSequence[SEQUENCE_LENGTH], bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL]);
	void fit(class_t input[SEQUENCE_LENGTH], class_t target);
	class_t predict(class_t input[SEQUENCE_LENGTH]);
	
	distance_t distanceToHyperplane(bool oneHotSequence [SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL], class_t hyperplane);
public:
	SVM(weight_t weights[NUM_CLASSES][SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL], weight_t intercepts[NUM_CLASSES]):
		weights(weights), intercepts(intercepts){}
	class_t operator()(bool opOnlyPredict, class_t input[SEQUENCE_LENGTH], class_t target);
};

template<typename weight_t, typename class_t, typename distance_t>
void SVM<weight_t, class_t, distance_t>::encodeInOneHot(class_t classSequence[SEQUENCE_LENGTH], bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL]) {
	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
		for (int j = 0; j < NUM_CLASSES_INCLUDING_NULL; j++) {
			oneHotSequence[i][j] = classSequence[i] == j;
		}
	}
}

template<typename weight_t, typename class_t, typename distance_t>
distance_t SVM<weight_t, class_t, distance_t>::distanceToHyperplane(bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL], class_t hyperplane) {
	distance_t res = 0;
	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
		for (int j = 0; j < NUM_CLASSES_INCLUDING_NULL; j++) {
			if (oneHotSequence[i][j])
				res += weights[hyperplane][i][j];
		}
	}
	res -= intercepts[hyperplane];
	return res;
}

template<typename weight_t, typename class_t, typename distance_t>
class_t SVM<weight_t, class_t, distance_t>::predict(class_t input [SEQUENCE_LENGTH]) {
	bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL];
	encodeInOneHot(input, oneHotSequence);

	class_t res = 0;
	distance_t minDistance = distanceToHyperplane(oneHotSequence, 0);
	for (int c = 1; c < NUM_CLASSES; c++) {
		distance_t distance = distanceToHyperplane(oneHotSequence, c);
		if (distance < minDistance) {
			minDistance = distance;
			res = c;
		}
	}
	return c;
}

template<typename weight_t, typename class_t, typename distance_t>
void SVM<weight_t, class_t, distance_t>::fit(class_t input[SEQUENCE_LENGTH], class_t target) {
	bool oneHotSequence[SEQUENCE_LENGTH][NUM_CLASSES_INCLUDING_NULL];
	encodeInOneHot(input, oneHotSequence);

	for (int c = 0; c < NUM_CLASSES; c++) {
		distance_t distance = distanceToHyperplane(oneHotSequence, c);
		bool classIsTarget = target == c;
		bool isPredictionCorrect = classIsTarget ? distance <= -1 : distance >= +1;

		if (!isPredictionCorrect) {
			for (int i = 0; i < SEQUENCE_LENGTH; i++) {
				for (int j = 0; j < NUM_CLASSES_INCLUDING_NULL; j++) {
					if (oneHotSequence[i][j])
						weights[c][i][j] += (classIsTarget ? -1 : +1) >> SVM_LEARNING_RATE_LOG2;
				}
			}
			intercepts[c] += (classIsTarget ? +1 : -1) >> SVM_LEARNING_RATE_LOG2;
			
		}

	}
}

template<typename weight_t, typename class_t, typename distance_t>
class_t SVM<weight_t, class_t, distance_t>::operator()(bool opOnlyPredict, class_t input[SEQUENCE_LENGTH], class_t target) {
	class_t res;

	if (!opOnlyPredict) {
		fit(input, target);
		class_t newInput[SEQUENCE_LENGTH];
		for (int i = 0; i < SEQUENCE_LENGTH - 1; i++) {
			newInput[i] = input[i + 1];
		}
		newInput[SEQUENCE_LENGTH - 1] = target;

		res = predict(newInput);
	}
	else {
		res = predict(input);
	}
	
	return res;
}

