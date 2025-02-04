#pragma once
#include <iostream>
#include <vector>
#include "../include/global.hpp"
#include "dirent.h"
#include "../top/top.hpp"

using namespace std;


/*
DictionaryEntry<delta_t, dic_confidence_t> operateDictionary(dic_index_t index, delta_t delta, bool performRead, dic_index_t &resultIndex, bool &isHit);
InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> operateInputBuffer(address_t addr, InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> entry,
		bool performRead, bool& isHit);
void operateSVM(class_t input[SEQUENCE_LENGTH], class_t target, class_t output[MAX_PREFETCHING_DEGREE]);
void prefetchWithGASP(address_t instructionPointer, block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE]
		);
void prefetchWithSGASP(block_address_t memoryAddress,
		block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE]
		);
*/


enum ExperimentType {
  INPUT_BUFFER_VALIDATION,
  INPUT_BUFFER_SOFT_VALIDATION,
  DICTIONARY_VALIDATION,
  DICTIONARY_SOFT_VALIDATION,
  SVM_VALIDATION,
  SVM_SOFT_VALIDATION,
  GASP_SOFT_VALIDATION,
  SGASP_SOFT_VALIDATION
}; 


class Experiment{
protected:
    string filePath;
    ExperimentType type;
    int numOperations;
public:
    bool checkConfiguration(){
    	return true;
    }
    void readTraceFile(string filePath);
    ExperimentType getType(){
        return type;
    }
    string getTracePath(){
        return filePath;
    }
    int getNumOperations(){
    	return numOperations;
    }
};

struct InputBufferValidationInput{
    address_t inputBufferAddr; 
    InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> entry;
	bool performRead;
};

struct InputBufferValidationOutput{
    InputBufferEntry<ib_tag_t, block_address_t, class_t, ib_confidence_t, ib_lru_t> entry;
    bool isHit;
};

bool areInputBufferOutputsEqual(InputBufferValidationOutput &output1, InputBufferValidationOutput &output2){
    return output1.entry.confidence == output2.entry.confidence
        && output1.entry.lastAddress == output2.entry.lastAddress
        && output1.entry.lastPredictedAddress == output2.entry.lastPredictedAddress
        && output1.entry.lruCounter == output2.entry.lruCounter
        && output1.entry.sequence == output2.entry.sequence
        && output1.entry.tag == output2.entry.tag
        && output1.entry.valid == output2.entry.valid
        && output1.isHit == output2.isHit;
}

class InputBufferValidation : public Experiment{
protected:
    vector<InputBufferValidationInput> inputs;
    vector<InputBufferValidationOutput> outputs;
    int i = 0;
    bool allChecksAreCorrect = true;
public:
    InputBufferValidation(){}
    InputBufferValidation(string filePath){
        type = ExperimentType::INPUT_BUFFER_VALIDATION;
        readTraceFile(filePath);
        numOperations = inputs.size();
    }
    void readTraceFile(string filePath);

    void reset(){
    	i = 0;
    	allChecksAreCorrect = true;
    }

    InputBufferValidationInput getNextInput(){
    	auto input = inputs[i];
    	return input;
    }

    void saveOutput(InputBufferValidationOutput output){
    	auto target = outputs[i];
    	if(!areInputBufferOutputsEqual(output, target)){
    		allChecksAreCorrect = false;
		}
    	i++;
    }

    bool hasPassed(){
    	return allChecksAreCorrect;
    }

};

class InputBufferSoftValidation : public InputBufferValidation{
protected:
    double hitRateDifferenceThreshold;
    int numHits = 0, numTargetHits = 0;
	int numReads = 0;
public:
    InputBufferSoftValidation(){}

    InputBufferSoftValidation(string filePath, double hitRateDifferenceThreshold = 0.05){
        type = ExperimentType::INPUT_BUFFER_SOFT_VALIDATION;
        readTraceFile(filePath);
        numOperations = inputs.size();
        this->hitRateDifferenceThreshold = hitRateDifferenceThreshold;
    }

    void reset(){
		i = 0;
		numHits = 0;
		numTargetHits = 0;
		numReads = 0;
	}

    void saveOutput(InputBufferValidationOutput output){
    	auto input = inputs[i];
		auto target = outputs[i];
		if(input.performRead){
			numHits += (int)output.isHit;
			numTargetHits += (int)target.isHit;
			numReads++;
		}
		i++;
	}

	bool hasPassed(){
		double hitRate = ((double)numHits) / numReads;
		double targetHitRate = ((double)numTargetHits) / numReads;
		bool res = hitRate > targetHitRate - this->hitRateDifferenceThreshold;

		std::cout << "Input buffer hit rate: " << std::to_string(hitRate) << std::endl;
		std::cout << "Target input buffer hit rate: " << std::to_string(targetHitRate) << std::endl;
		// std::cout << "Test passed? " << std::to_string(res) << std::endl;

		return res;
	}

};

struct DictionaryValidationInput{
    dic_index_t index; 
    delta_t delta; 
    bool performRead;
};

struct DictionaryValidationOutput{
    DictionaryEntry<delta_t, dic_confidence_t> entry;
    dic_index_t resultIndex;
    bool isHit;
};

bool areDictionaryOutputsEqual(DictionaryValidationOutput &output1, DictionaryValidationOutput &output2){
    return output1.entry.valid == output2.entry.valid
        && output1.entry.delta == output2.entry.delta
        && output1.entry.confidence == output2.entry.confidence
        && output1.isHit == output2.isHit;
}

class DictionaryValidation : public Experiment{
protected:
    vector<DictionaryValidationInput> inputs;
    vector<DictionaryValidationOutput> outputs;
    int i = 0;
	bool allChecksAreCorrect = true;
public:
    DictionaryValidation(){}
    DictionaryValidation(string filePath){
        type = ExperimentType::DICTIONARY_VALIDATION;
        readTraceFile(filePath);
        numOperations = inputs.size();
    }

    void reset(){
		i = 0;
		allChecksAreCorrect = true;
	}

    DictionaryValidationInput getNextInput(){
		auto input = inputs[i];
		return input;
	}

	void saveOutput(DictionaryValidationOutput output){
		auto target = outputs[i];
		if(!areDictionaryOutputsEqual(output, target)){
			allChecksAreCorrect = false;
		}
		i++;
	}

	bool hasPassed(){
		return allChecksAreCorrect;
	}

    void readTraceFile(string filePath);
};

class DictionarySoftValidation : public DictionaryValidation{
protected:
    double hitRateDifferenceThreshold;
    int numHits = 0, numTargetHits = 0;
	int numWrites = 0;
public:
    DictionarySoftValidation(){}
    DictionarySoftValidation(string filePath, double hitRateDifferenceThreshold = 0.05){
        type = ExperimentType::DICTIONARY_SOFT_VALIDATION;
        readTraceFile(filePath);
        this->hitRateDifferenceThreshold = hitRateDifferenceThreshold;
        numOperations = inputs.size();
    }

    void reset(){
		i = 0;
		numHits = 0;
		numTargetHits = 0;
		numWrites = 0;
	}

	void saveOutput(DictionaryValidationOutput output){
		auto input = inputs[i];
		auto target = outputs[i];
		if(!input.performRead){
			numHits += (int)output.isHit;
			numTargetHits += (int)target.isHit;
			numWrites++;
		}
		i++;
	}

	bool hasPassed(){
		double hitRate = ((double)numHits) / numWrites;
		double targetHitRate = ((double)numTargetHits) / numWrites;

		bool res = hitRate > targetHitRate - this->hitRateDifferenceThreshold;

		std::cout << "Dictionary hit rate: " << std::to_string(hitRate) << std::endl;
		std::cout << "Target dictionary hit rate: " << std::to_string(targetHitRate) << std::endl;
		// std::cout << "Test passed? " << std::to_string(res) << std::endl;

		return res;

	}

};


struct SVMValidationInput{
    class_t input[SEQUENCE_LENGTH]; 
    class_t target;
};

struct SVMValidationOutput{
    class_t output[MAX_PREFETCHING_DEGREE];
};

bool areSVMOutputsEqual(SVMValidationOutput &output1, SVMValidationOutput &output2){
    bool res = true;
    for(int i = 0; i < MAX_PREFETCHING_DEGREE; i++){
        res = res && output1.output[i] == output2.output[i];
    }
    return res;
}

class SVMValidation : public Experiment{
protected:
    vector<SVMValidationInput> inputs;
    vector<SVMValidationOutput> outputs;
    int i = 0;
	bool allChecksAreCorrect = true;
public:
    SVMValidation(){}
    SVMValidation(string filePath){
        type = ExperimentType::SVM_VALIDATION;
        readTraceFile(filePath);
        numOperations = inputs.size();
    }
    void reset(){
		i = 0;
		allChecksAreCorrect = true;
	}

	SVMValidationInput getNextInput(){
		auto input = inputs[i];
		return input;
	}

	void saveOutput(SVMValidationOutput output){
		auto target = outputs[i];
		if(!areSVMOutputsEqual(output, target)){
			allChecksAreCorrect = false;
		}
		i++;
	}

	bool hasPassed(){
		return allChecksAreCorrect;
	}

    void readTraceFile(string filePath);
};

class SVMSoftValidation : public SVMValidation{
protected:
    double matchingThreshold;
    int numMatches = 0;
    int numHits = 0;
    int numTargetHits = 0;
	int numPredictions = 0;
public:
    SVMSoftValidation(){}
    SVMSoftValidation(string filePath, double matchingThreshold = 0.85){
        type = ExperimentType::SVM_SOFT_VALIDATION;
        readTraceFile(filePath);
        this->matchingThreshold = matchingThreshold;
        numOperations = inputs.size();
    }

    void reset(){
		i = 0;
		numMatches = 0;
		numHits = 0;
		numTargetHits = 0;
		numPredictions = 0;
	}

	void saveOutput(SVMValidationOutput output){
		auto input = inputs[i];
		auto target = outputs[i];
		numHits += inputs[i].target == output.output[0];
		numTargetHits += inputs[i].target == target.output[0];

		for(int k = 0; k < MAX_PREFETCHING_DEGREE; k++){
			// Only counting true positives over all predictions (precision):
			if(target.output[k] == NUM_CLASSES){
				break;
			}
			// else if(inputs[i].target == target.output[0]){
				numMatches += target.output[k] == output.output[k];

				numPredictions++;
			// }
		}


		i++;
	}

	bool hasPassed(){
		double matchRate = ((double) numMatches) / numPredictions;
		double precisionDifference = ((double)numHits - (double)numTargetHits) / numPredictions;

		// bool res = matchRate > this->matchingThreshold;
		// bool res = abs(hitDifference) < 0.05;

		std::cout << "SVM results match rate: " << std::to_string(matchRate) << std::endl;
		std::cout << "SVM results precision difference: " << std::to_string(precisionDifference) << std::endl;
		// std::cout << "Test passed? " << std::to_string(res) << std::endl;

		return res;

	}

};

struct PrefetchingValidationInput{
	address_t instructionPointer;
	block_address_t memoryAddress;
};

struct PrefetchingValidationOutput{
	block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE];
};

template<typename output_t>
bool arePrefetchingOutputsEqual(output_t &output1, output_t &output2){
    bool res = true;
    for(int i = 0; i < MAX_PREFETCHING_DEGREE; i++){
        res = res && output1.output[i] == output2.output[i];
    }
    return res;
}

class PrefetchingSoftValidation : public Experiment{
protected:
    vector<PrefetchingValidationInput> inputs;
    vector<PrefetchingValidationOutput> outputs;
    int i = 0;
    double matchingThreshold;
    int numPrefetches = 0;
    int numMatches = 0;
public:
	PrefetchingSoftValidation(){}
	PrefetchingSoftValidation(string filePath, double matchingThreshold = 0.8){
        readTraceFile(filePath);
        numOperations = inputs.size();
        this->matchingThreshold = matchingThreshold;
    }
    void reset(){
    	i = 0;
		numMatches = 0;
		numPrefetches = 0;
	}

    PrefetchingValidationInput getNextInput(){
		auto input = inputs[i];
		return input;
	}

	void saveOutput(PrefetchingValidationOutput output){
		auto input = inputs[i];
		auto target = outputs[i];

		for(int k = 0; k < MAX_PREFETCHING_DEGREE; k++){
			// Only counting true positives over all predictions (precision):
			if(target.addressesToPrefetch[k] == NUM_CLASSES){
				break;
			}
			else {
				numMatches += target.addressesToPrefetch[k] == output.addressesToPrefetch[k];
				numPrefetches++;
			}
		}

		i++;
	}

	bool hasPassed(){
		double matchRate = ((double) numMatches) / numPrefetches;

		bool res = matchRate > this->matchingThreshold;

		std::cout << "Prefetching results match rate: " << std::to_string(matchRate) << std::endl;
		// std::cout << "Test passed? " << std::to_string(res) << std::endl;

		return res;

	}

    void readTraceFile(string filePath){
    	// Pass...
    }
};

class GASPSoftValidation : public PrefetchingSoftValidation{
public:
	GASPSoftValidation(){}
	GASPSoftValidation(string filePath, double matchingThreshold = 0.8){
		type = ExperimentType::GASP_SOFT_VALIDATION;
        readTraceFile(filePath);
        numOperations = inputs.size();
        this->matchingThreshold = matchingThreshold;
    }
	void readTraceFile(string filePath);
};

class SGASPSoftValidation : public PrefetchingSoftValidation{
public:
	SGASPSoftValidation(){}
	SGASPSoftValidation(string filePath, double matchingThreshold = 0.8){
		type = ExperimentType::SGASP_SOFT_VALIDATION;
        readTraceFile(filePath);
        numOperations = inputs.size();
        this->matchingThreshold = matchingThreshold;
    }
	void readTraceFile(string filePath);
};

template<class Experiment>
class Experimentation{
protected:
	string headerPath;
	vector<string> getTracePaths();
public:
	vector<Experiment> experiments;
    Experimentation(){}
    Experimentation(string headerPath// , ExperimentType type
    		){
    	this->headerPath = headerPath;
    	auto tracePaths = this->getTracePaths();
    	for(auto& tracePath : tracePaths){
    		cout << "Trace path: " << tracePath << endl;
    		experiments.push_back(Experiment(tracePath));
    	}
	}



};

