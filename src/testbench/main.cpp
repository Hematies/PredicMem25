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
	auto inputBufferValidation = Experimentation<InputBufferSoftValidation>(traceDirPath + inputBufferTracesDirName);
	auto dictionaryValidation = Experimentation<DictionarySoftValidation>(traceDirPath + dictionaryTracesDirName);
	auto svmValidation = Experimentation<SVMSoftValidation>(traceDirPath + svmTracesDirName);

	bool passed = // inputBufferValidation.perform();
			dictionaryValidation.perform()
			// && svmValidation.perform()
			;

	cout << "Passed: " << to_string(passed) << "\n";
	return !passed;

}
