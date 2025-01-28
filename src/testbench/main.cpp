#include <iostream>
#include <string>
#include "../include/global.hpp"
#include "reading.hpp"
#include "../top/top.hpp"

using namespace std;

string traceDirPath = "//home//pablo//Escritorio//PredicMem25//traces//";
string inputBufferTracesDirName = "inputBufferTraces//";
string dictionaryTracesDirName = "dictionaryTraces//";
string svmTracesDirName = "svmTraces//";


int main(int argc, char **argv)
{
	bool validateInputBuffer = false, validateDictionary = false, validateSVM = false;

	for(int i = 1; i < argc; i++){
		string argument = string(argv[i]);
		if(argument == "--validateInputBuffer" || argument == "-vIB"){
			validateInputBuffer = true;
		}
		else if(argument == "--validateDictionary" || argument == "-vD"){
			validateDictionary = true;
		}
		else if(argument == "--validateSVM" || argument == "-vSVM"){
			validateSVM = true;
		}
		else{
			cout << "No type of validation has been indicated\n";
			return 1;
		}
	}


	bool passed = true;

	if(validateInputBuffer){
		auto inputBufferValidation = Experimentation<InputBufferSoftValidation>(traceDirPath + string("inputBufferTrace.txt"));
		auto experiments = inputBufferValidation.experiments;
		for(auto& experiment : experiments){
			for(int i = 0; i < experiment.getNumOperations(); i++){
				auto input = experiment.getNextInput();
				bool isHit = false;
				InputBufferValidationOutput output;
				auto entry = operateInputBuffer(input.inputBufferAddr, input.entry, input.performRead, isHit);
				output.entry = entry;
				output.isHit = isHit;
				experiment.saveOutput(output);
			}
			passed = passed && experiment.hasPassed();
		}
	}


	if(validateDictionary){
		auto dictionaryValidation = Experimentation<DictionarySoftValidation>(traceDirPath + string("dictionaryTrace.txt"));
		auto experiments = dictionaryValidation.experiments;
		for(auto& experiment : experiments){
			for(int i = 0; i < experiment.getNumOperations(); i++){
				auto input = experiment.getNextInput();
				bool isHit = false;
				dic_index_t resultIndex = 0;
				DictionaryValidationOutput output;
				auto entry = operateDictionary(input.index, input.delta, input.performRead, resultIndex, isHit);
				output.entry = entry;
				output.isHit = isHit;
				output.resultIndex = resultIndex;;
				experiment.saveOutput(output);
			}
			passed = passed && experiment.hasPassed();
		}
	}


	if(validateSVM){
		auto svmValidation = Experimentation<SVMSoftValidation>(traceDirPath + string("svmTrace.txt"));
		auto experiments = svmValidation.experiments;
		for(auto& experiment : experiments){
			for(int i = 0; i < experiment.getNumOperations(); i++){
				auto input = experiment.getNextInput();
				class_t outputClasses[MAX_PREFETCHING_DEGREE];
				SVMValidationOutput output;
				operateSVM(input.input, input.target, outputClasses);
				for(int k = 0; k < MAX_PREFETCHING_DEGREE; k++){
					output.output[k] = outputClasses[k];
				}
				experiment.saveOutput(output);
			}
			passed = passed && experiment.hasPassed();
		}
	}

	cout << "Passed: " << to_string(passed) << "\n";
	return !passed;

}
