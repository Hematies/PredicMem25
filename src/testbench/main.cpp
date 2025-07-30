#include <iostream>
#include <string>
#include "../include/global.hpp"
#include "reading.hpp"
#include "../top/top.hpp"

using namespace std;


// string traceDirPath = "/home/pablo/Escritorio/PredicMem25/traces/";
// string traceDirPath = "/home/pablo/Escritorio/PredicMem25/traces_sgasp/";
string traceDirPath = "/home/pablo/Escritorio/PredicMem25/traces_sgasp_high_confidence/";
string inputBufferTracesDirName = "inputBufferTraces/";
string dictionaryTracesDirName = "dictionaryTraces/";
string svmTracesDirName = "svmTraces/";


int main(int argc, char **argv)
{
 	bool validateInputBuffer = false, validateDictionary = false, validateSVM = false,
 			validateGASP = false, validateSGASP = false, validateBSGASP = false;

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
		else if(argument == "--validateBSGASP" || argument == "-vB"){
			validateBSGASP = true;
		}

	}

	if(!validateInputBuffer && !validateDictionary && !validateSVM
			&& !validateGASP && !validateSGASP && !validateBSGASP){
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
			unsigned long long cycle = 0L;
			for(int i = 0; i < experiment.getNumOperations(); i++){
				auto input = experiment.getNextInput();
				unsigned long long nextCycle = input.cycle;
				class_t outputClasses[MAX_PREFETCHING_DEGREE];
				SVMValidationOutput output;
				operateSVM(input.input, input.target, outputClasses);
				for(int k = 0; k < MAX_PREFETCHING_DEGREE; k++){
					output.output[k] = outputClasses[k];
				}
				while(cycle < nextCycle){
					operateSVMWithNop(input.input, input.target, outputClasses,
												true);
					if((nextCycle - experiment.maxNumNopCycles) > cycle)
						cycle = nextCycle - experiment.maxNumNopCycles;
					else
						cycle++;
				}
				operateSVMWithNop(input.input, input.target, outputClasses,
												false);
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
			unsigned long long cycle = 0L;
			for(int i = 0; i < experiment.getNumOperations(); i++){
				auto input = experiment.getNextInput();
				unsigned long long nextCycle = input.cycle;
				block_address_t addressesToPrefetch[MAX_PREFETCHING_DEGREE];
				PrefetcherValidationOutput output;

				while(cycle < nextCycle){
					prefetchWithGASPWithNop(input.instructionPointer, input.memoryAddress, addressesToPrefetch,
												true);
					if((nextCycle - experiment.maxNumNopCycles) > cycle)
						cycle = nextCycle - experiment.maxNumNopCycles;
					else
						cycle++;
				}
				prefetchWithGASPWithNop(input.instructionPointer, input.memoryAddress, addressesToPrefetch,
												false);
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
			unsigned long long cycle = 0L;
			for(int i = 0; i < experiment.getNumOperations(); i++){
				auto input = experiment.getNextInput();
				unsigned long long nextCycle = input.cycle;
				block_address_t addressToPrefetch;
				PrefetcherValidationOutput output;

				while(cycle < nextCycle){
					prefetchWithSGASPWithNopWithDataflow(input.memoryAddress, addressToPrefetch, true);
					if((nextCycle - experiment.maxNumNopCycles) > cycle)
						cycle = nextCycle - experiment.maxNumNopCycles;
					else
						cycle++;
				}
				prefetchWithSGASPWithNopWithDataflow(input.memoryAddress, addressToPrefetch, false);

				output.addressesToPrefetch[0] = addressToPrefetch;

				experiment.saveOutput(output);
			}
			passed = experiment.hasPassed() && passed;
		}
	}

	if(validateBSGASP){
		auto bsgaspValidation = Experimentation<BSGASPSoftValidation>(traceDirPath + string("burstPrefetcherTraceHeader.txt"));
		auto experiments = bsgaspValidation.experiments;
		for(auto& experiment : experiments){
			unsigned long long cycle = 0L;
			for(int i = 0; i < experiment.getNumOperations(); i++){
				auto input = experiment.getNextInput();
				unsigned long long nextCycle = input.cycle;
				block_address_t addressToPrefetch;
				prefetch_block_burst_length_t blockBurstLength;
				BurstPrefetchingValidationOutput output;

				while(cycle < nextCycle){
					prefetchWithBSGASPWithNopWithDataflowForTesting(input.memoryAddress, 
						input.burstLength,
						addressToPrefetch, 
						blockBurstLength
						true);
					if((nextCycle - experiment.maxNumNopCycles) > cycle)
						cycle = nextCycle - experiment.maxNumNopCycles;
					else
						cycle++;
				}
				prefetchWithBSGASPWithNopWithDataflowForTesting(input.memoryAddress, 
					input.burstLength,
					addressToPrefetch, 
					blockBurstLength,
					false);

				output.addressesToPrefetch[0] = addressToPrefetch;
				output.nextBurstLength = blockBurstLength;

				experiment.saveOutput(output);
			}
			passed = experiment.hasPassed() && passed;
		}
	}

	cout << "Passed: " << to_string(passed) << "\n";
	return !passed;

}
