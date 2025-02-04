#include <iostream>
#include <string>
#include "../include/global.hpp"
#include "reading.hpp"
#include "../top/top.hpp"

using namespace std;

string traceDirPath = "/home/pablo/Escritorio/PredicMem25/traces/";
string inputBufferTracesDirName = "inputBufferTraces/";
string dictionaryTracesDirName = "dictionaryTraces/";
string svmTracesDirName = "svmTraces/";


int main(int argc, char **argv)
{
 	bool validateInputBuffer = false, validateDictionary = false, validateSVM = false,
 			validateGASP = false, validateSGASP = false;

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
		else if(argument == "--validateGASP" || argument == "-vG"){
			validateGASP = true;
		}
		else if(argument == "--validateSGASP" || argument == "-vS"){
			validateSGASP = true;
		}

	}

	if(!validateInputBuffer && !validateDictionary && !validateSVM
			&& !validateGASP && !validateSGASP){
		cout << "No type of validation has been indicated\n";
		return 1;
	}


	bool passed = true;

	if(validateInputBuffer){
		auto inputBufferValidation = Experimentation<InputBufferSoftValidation>(traceDirPath + string("inputBufferTraceHeader.txt"));
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
			passed = experiment.hasPassed() && passed;
		}
	}


	if(validateDictionary){
		auto dictionaryValidation = Experimentation<DictionarySoftValidation>(traceDirPath + string("dictionaryTraceHeader.txt"));
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
			passed = experiment.hasPassed() && passed;
		}
	}


	if(validateSVM){
		auto svmValidation = Experimentation<SVMSoftValidation>(traceDirPath + string("svmTraceHeader.txt"));
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
			passed = experiment.hasPassed() && passed;
		}
	}

	if(validateGASP){
		auto gaspValidation = Experimentation<GASPSoftValidation>(traceDirPath + string("prefetcherTraceHeader.txt"));
		auto experiments = gaspValidation.experiments;
		for(auto& experiment : experiments){
			for(int i = 0; i < experiment.getNumOperations(); i++){
				auto input = experiment.getNextInput();
				block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE];
				PrefetcherValidationOutput output;
				prefetchWithGASP(input.instructionPointer, input.memoryAddress, addressesToPrefetch);
				for(int k = 0; k < MAX_PREFETCHING_DEGREE; k++){
					output.addressesToPrefetch[k] = addressesToPrefetch[k];
				}
				experiment.saveOutput(output);
			}
			passed = experiment.hasPassed() && passed;
		}
	}

	if(validateSGASP){
		auto sgaspValidation = Experimentation<SGASPSoftValidation>(traceDirPath + string("prefetcherTraceHeader.txt"));
		auto experiments = sgaspValidation.experiments;
		for(auto& experiment : experiments){
			for(int i = 0; i < experiment.getNumOperations(); i++){
				auto input = experiment.getNextInput();
				block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE];
				PrefetcherValidationOutput output;
				prefetchWithSGASP(input.memoryAddress, addressesToPrefetch);
				for(int k = 0; k < MAX_PREFETCHING_DEGREE; k++){
					output.addressesToPrefetch[k] = addressesToPrefetch[k];
				}
				experiment.saveOutput(output);
			}
			passed = experiment.hasPassed() && passed;
		}
	}

	cout << "Passed: " << to_string(passed) << "\n";
	return !passed;

}
