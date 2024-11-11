#include <iostream>
#include <string>
#include "../include/global.hpp"
#include "reading.hpp"

using namespace std;

string traceDirPath = "/home/pablo/Escritorio/PredicMem25/traces/";
string inputBufferTracesDirName = "inputBufferTraces/";
string dictionaryTracesDirName = "dictionaryTraces/";
string svmTracesDirName = "svmTraces/";


int main()
{
	auto inputBufferValidation = Experimentation<InputBufferSoftValidation>(traceDirPath + inputBufferTracesDirName// ,
			// ExperimentType::INPUT_BUFFER_SOFT_VALIDATION
			);
	auto dictionaryValidation = Experimentation<DictionarySoftValidation>(traceDirPath + dictionaryTracesDirName// ,
			// ExperimentType::DICTIONARY_SOFT_VALIDATION
			);
	auto svmValidation = Experimentation<SVMSoftValidation>(traceDirPath + svmTracesDirName// ,
			// ExperimentType::SVM_SOFT_VALIDATION
			);

	bool passed = inputBufferValidation.perform() && dictionaryValidation.perform() && svmValidation.perform();
	return !passed;

}
