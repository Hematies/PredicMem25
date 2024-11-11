#pragma once
#include <iostream>
#include <vector>
#include "../include/global.hpp"
#include <dirent.h>

using namespace std;

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

enum ExperimentType {
  INPUT_BUFFER_VALIDATION,
  INPUT_BUFFER_SOFT_VALIDATION,
  DICTIONARY_VALIDATION,
  DICTIONARY_SOFT_VALIDATION,
  SVM_VALIDATION,
  SVM_SOFT_VALIDATION,
}; 


class Experiment{
protected:
    string filePath;
    ExperimentType type;
public:
    bool perform();
    bool checkConfiguration();
    void readTraceFile(string filePath);
    ExperimentType getType(){
        return type;
    }
    string getTracePath(){
        return filePath;
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
public:
    InputBufferValidation(){}
    InputBufferValidation(string filePath){
        type = ExperimentType::INPUT_BUFFER_VALIDATION;
        readTraceFile(filePath);
    }
    void readTraceFile(string filePath);
    bool perform(){
        bool res = false;
        if(checkConfiguration()){
            res = true;
            for(int i = 0; i < inputs.size(); i++){
                auto input = inputs[i];
                auto targetOutput = outputs[i];

                InputBufferValidationOutput output;
                auto entry = operateInputBuffer(input.inputBufferAddr, input.entry, input.performRead, output.isHit);
                output.entry = entry;

                if(!areInputBufferOutputsEqual(output, targetOutput)){
                    res = false;
                    break;
                }
            }
        }
        return res;
    }
    bool checkConfiguration(){
        return true;
    }
};

class InputBufferSoftValidation : public InputBufferValidation{
protected:
    double hitRateDifferenceThreshold;
public:
    InputBufferSoftValidation(){}

    InputBufferSoftValidation(string filePath, double hitRateDifferenceThreshold = 0.05){
        type = ExperimentType::INPUT_BUFFER_SOFT_VALIDATION;
        readTraceFile(filePath);
        this->hitRateDifferenceThreshold = hitRateDifferenceThreshold;
    }

    bool perform(){
        bool res = false;
        if(checkConfiguration()){
            res = true;
            double hitRate = 0.0, targetHitRate = 0.0;
            int numReads = 0;

            for(int i = 0; i < inputs.size(); i++){
                auto input = inputs[i];
                auto targetOutput = outputs[i];

                InputBufferValidationOutput output;
                auto entry = operateInputBuffer(input.inputBufferAddr, input.entry, input.performRead, output.isHit);
                output.entry = entry;

                if(input.performRead){
                    hitRate += (int)output.isHit;
                    targetHitRate += (int)targetOutput.isHit;
                    numReads++;
                }
            }
            hitRate = hitRate / numReads;
            targetHitRate = targetHitRate / numReads;

            res = abs(hitRate - targetHitRate) < this->hitRateDifferenceThreshold;
        }
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
public:
    DictionaryValidation(){}
    DictionaryValidation(string filePath){
        type = ExperimentType::DICTIONARY_VALIDATION;
        readTraceFile(filePath);
    }
    bool perform(){
        bool res = false;
        if(checkConfiguration()){
            res = true;
            for(int i = 0; i < inputs.size(); i++){
                auto input = inputs[i];
                auto targetOutput = outputs[i];

                DictionaryValidationOutput output;
                auto entry = operateDictionary(input.index, input.delta, input.performRead, output.resultIndex, output.isHit);
                output.entry = entry;

                if(!areDictionaryOutputsEqual(output, targetOutput)){
                    res = false;
                    break;
                }
            }
        }
        return res;
    }
    bool checkConfiguration(){
        return true;
    }
    void readTraceFile(string filePath);
};

class DictionarySoftValidation : public DictionaryValidation{
protected:
    double hitRateDifferenceThreshold;
public:
    DictionarySoftValidation(){}
    DictionarySoftValidation(string filePath, double hitRateDifferenceThreshold = 0.05){
        type = ExperimentType::DICTIONARY_SOFT_VALIDATION;
        readTraceFile(filePath);
        this->hitRateDifferenceThreshold = hitRateDifferenceThreshold;
    }

    bool perform(){
        bool res = false;
        if(checkConfiguration()){
            res = true;
            double hitRate = 0.0, targetHitRate = 0.0;
            int numWrites = 0;

            for(int i = 0; i < inputs.size(); i++){
                auto input = inputs[i];
                auto targetOutput = outputs[i];

                DictionaryValidationOutput output;
                auto entry = operateDictionary(input.index, input.delta, input.performRead, output.resultIndex, output.isHit);
                output.entry = entry;

                if(!input.performRead){
                    hitRate += (int)output.isHit;
                    targetHitRate += (int)targetOutput.isHit;
                    numWrites++;
                }

            }
            hitRate = hitRate / numWrites;
            targetHitRate = targetHitRate / numWrites;

            res = abs(hitRate - targetHitRate) < this->hitRateDifferenceThreshold;
        }
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
public:
    SVMValidation(){}
    SVMValidation(string filePath){
        type = ExperimentType::SVM_VALIDATION;
        readTraceFile(filePath);
    }
    bool perform(){
        bool res = false;
        if(checkConfiguration()){
            res = true;
            for(int i = 0; i < inputs.size(); i++){
                auto input = inputs[i];
                auto targetOutput = outputs[i];

                SVMValidationOutput output;
                for(int i = 0; i < MAX_PREFETCHING_DEGREE; i++){
                    output.output[i] = NUM_CLASSES;
                }

                operateSVM(input.input, input.target, output.output);

                if(!areSVMOutputsEqual(output, targetOutput)){
                    res = false;
                    break;
                }
            }
        }
        return res;
    }
    bool checkConfiguration(){
        return true;
    }
    void readTraceFile(string filePath);
};

class SVMSoftValidation : public SVMValidation{
protected:
    double matchingThreshold;
public:
    SVMSoftValidation(){}
    SVMSoftValidation(string filePath, double matchingThreshold = 0.05){
        type = ExperimentType::SVM_SOFT_VALIDATION;
        readTraceFile(filePath);
        this->matchingThreshold = matchingThreshold;
    }

    bool perform(){
        bool res = false;
        if(checkConfiguration()){
            res = true;
            double matchRate = 0.0;
            int numPredictions = 0;

            for(int i = 0; i < inputs.size(); i++){
                auto input = inputs[i];
                auto targetOutput = outputs[i];

                SVMValidationOutput output;
                for(int i = 0; i < MAX_PREFETCHING_DEGREE; i++){
                    output.output[i] = NUM_CLASSES;
                }

                operateSVM(input.input, input.target, output.output);

                for(int i = 0; i < MAX_PREFETCHING_DEGREE; i++){
                    // Only counting true positives over all predictions (precision):
                    if(targetOutput.output[i] == NUM_CLASSES){
                        break;
                    }
                    else{
                        matchRate += targetOutput.output[i] == output.output[i];
                        numPredictions++;
                    }

                }

            }
            matchRate = matchRate / numPredictions;

            res = matchRate > this->matchingThreshold;
        }
        return res;
    }

};

template<class Experiment>
class Experimentation{
protected:
    vector<Experiment> experiments;
public:
    Experimentation(){}
    Experimentation(string folderPath// , ExperimentType type
    		){
        DIR    *dir;
        dirent *pdir;
        dir = opendir(folderPath.c_str());
        vector<string> files;
        while (pdir = readdir(dir))
        {
        	if(strcmp(pdir->d_name, "..") != 0 && strcmp(pdir->d_name, ".") != 0)
        		files.push_back(pdir->d_name);
        }
        for (const auto & filename : files) {
        	/*
            switch (type)
            {
            case INPUT_BUFFER_VALIDATION:
                experiments.push_back(InputBufferValidation(folderPath + "//" + filename));
                break;
            case INPUT_BUFFER_SOFT_VALIDATION:
                experiments.push_back(InputBufferSoftValidation(folderPath + "//" + filename));
                break;
            case DICTIONARY_VALIDATION:
                experiments.push_back(DictionaryValidation(folderPath + "//" + filename));
                break;
            case DICTIONARY_SOFT_VALIDATION:
                experiments.push_back(DictionarySoftValidation(folderPath + "//" + filename));
                break;
            case SVM_VALIDATION:
                experiments.push_back(SVMValidation(folderPath + "//" + filename));
                break;
            case SVM_SOFT_VALIDATION:
                experiments.push_back(SVMSoftValidation(folderPath + "//" + filename));
                break;
            default:
                break;
            }
            */
        	std::cout << folderPath + filename << "\n";
        	experiments.push_back(Experiment(folderPath + "//" + filename));
        }
    }
    bool perform(){
        bool res = true;
        for(auto& experiment : experiments){
            bool success = experiment.perform();
            res = res && success;
            string type;
            switch (type)
			{
			case INPUT_BUFFER_VALIDATION:
				type = "Input Buffer Validation";
				break;
			case INPUT_BUFFER_SOFT_VALIDATION:
				type = "Input Buffer Soft Validation";
				break;
			case DICTIONARY_VALIDATION:
				type = "Dictionary Validation";
				break;
			case DICTIONARY_SOFT_VALIDATION:
				type = "Dictionary Soft Validation";
				break;
			case SVM_VALIDATION:
				type = "SVM Validation";
				break;
			case SVM_SOFT_VALIDATION:
				type = "SVM Soft Validation";
				break;
			default:
				break;
			}

            if(!success){
                cout << "FAILURE for experiment of type " << type << " and trace path " << experiment.getTracePath() << "\n";
            }
            else{
                cout << "SUCCESS for experiment of type " << type << " and trace path " << experiment.getTracePath() << "\n";
            }
        }
        return res;
    }

};

