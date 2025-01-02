#include "../include/global.hpp"
#include "experimentation.hpp"
#include <string>
#include "traceReader.hpp"

// https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
std::vector<std::string> split(const std::string& s, const std::string& delimiter) {
    std::vector<std::string> tokens;
    std::string s_ = std::string(s);
    size_t pos = 0;
    std::string token;
    while ((pos = s_.find(delimiter)) != std::string::npos) {
        token = s_.substr(0, pos);
        tokens.push_back(token);
        s_.erase(0, pos + delimiter.length());
    }
    tokens.push_back(s_);

    return tokens;
}

void parseInputBufferInOutLine(string line, InputBufferValidationInput& input, InputBufferValidationOutput& output){
    auto chains = split(line, ";");
    string inputLine = chains[0], outputLine = chains[1];
    vector<string> inputElements = split(inputLine, ","), outputElements = split(outputLine, ",");

    /*
    for(auto element : inputElements){
    	std::cout << element << ", ";
    }
    std::cout << "; ";

    for(auto element : outputElements){
        	std::cout << element << ", ";
        }
        std::cout << std::endl;
	*/
    // Input:
    input.inputBufferAddr = (address_t) std::stol(inputElements[0]);
    // input.entry.valid = std::stoi(inputElements[1]);
    input.entry.valid = true;
    input.entry.tag = (ib_tag_t) std::stol(inputElements[2]);
    input.entry.lastAddress = (block_address_t) std::stol(inputElements[3]);
    for(int i = 0; i < SEQUENCE_LENGTH; i++){
        input.entry.sequence[i] = (class_t) std::stoi(inputElements[i + 4]);
    }    
    input.entry.confidence = (ib_confidence_t) std::stoi(inputElements[4 + SEQUENCE_LENGTH]);
    input.entry.lastPredictedAddress = (block_address_t) std::stol(inputElements[5 + SEQUENCE_LENGTH]);
    input.entry.lruCounter = (ib_lru_t) std::stoi(inputElements[6 + SEQUENCE_LENGTH]);
    input.performRead = std::stoi(inputElements[7 + SEQUENCE_LENGTH]);

    // Output:
    output.entry.valid = std::stoi(outputElements[0]);
    output.entry.tag = (ib_tag_t) std::stol(outputElements[1]);
    output.entry.lastAddress = (block_address_t) std::stol(outputElements[2]);
    for(int i = 0; i < SEQUENCE_LENGTH; i++){
        output.entry.sequence[i] = (class_t) std::stoi(outputElements[i + 3]);
    }    
    output.entry.confidence = (ib_confidence_t) std::stoi(outputElements[3 + SEQUENCE_LENGTH]);
    output.entry.lastPredictedAddress = (block_address_t) std::stol(outputElements[4 + SEQUENCE_LENGTH]);
    output.entry.lruCounter = (ib_lru_t) std::stoi(outputElements[5 + SEQUENCE_LENGTH]);
    output.isHit = std::stoi(outputElements[6 + SEQUENCE_LENGTH]);

}

void parseDictionaryInOutLine(string line, DictionaryValidationInput& input, DictionaryValidationOutput& output){
    auto chains = split(line, ";");
    string inputLine = chains[0], outputLine = chains[1];
    vector<string> inputElements = split(inputLine, ","), outputElements = split(outputLine, ",");

    // Input: 
    input.index = (dic_index_t) std::stoi(inputElements[0]);
    input.delta = (delta_t) std::stol(inputElements[1]);
    input.performRead = std::stoi(inputElements[2]);

    // Output:
    output.entry.valid = std::stoi(outputElements[0]);
    output.entry.delta = (delta_t) std::stol(outputElements[1]);
    output.entry.confidence = (dic_confidence_t) std::stoi(outputElements[2]);
    output.resultIndex = (dic_index_t) std::stoi(outputElements[3]);
    output.isHit = std::stoi(outputElements[4]);

}

void parseSVMInOutLine(string line, SVMValidationInput& input, SVMValidationOutput& output){
    auto chains = split(line, ";");
    string inputLine = chains[0], outputLine = chains[1];
    vector<string> inputElements = split(inputLine, ","), outputElements = split(outputLine, ",");

    // Input:
    for(int i = 0; i < SEQUENCE_LENGTH; i++){
        input.input[i] = (class_t) std::stoi(inputElements[i]);
    } 
    input.target = (class_t) std::stoi(inputElements[SEQUENCE_LENGTH]);

    // Output:
    for(int i = 0; i < MAX_PREFETCHING_DEGREE; i++){
        output.output[i] = (class_t) std::stoi(outputElements[i]);
    } 

}

void InputBufferValidation::readTraceFile(string filePath){
    auto traceReader = TraceReader(filePath);
    auto lines = traceReader.readAllLines();

    for(auto& line : lines){
        InputBufferValidationInput input;
        InputBufferValidationOutput output;
        parseInputBufferInOutLine(line, input, output);
        inputs.push_back(input);
        outputs.push_back(output);
    }
}

void DictionaryValidation::readTraceFile(string filePath){
    auto traceReader = TraceReader(filePath);
    auto lines = traceReader.readAllLines();

    for(auto& line : lines){
        DictionaryValidationInput input;
        DictionaryValidationOutput output;
        parseDictionaryInOutLine(line, input, output);
        inputs.push_back(input);
        outputs.push_back(output);
    }
}

void SVMValidation::readTraceFile(string filePath){
    auto traceReader = TraceReader(filePath);
    auto lines = traceReader.readAllLines();

    for(auto& line : lines){
        SVMValidationInput input;
        SVMValidationOutput output;
        parseSVMInOutLine(line, input, output);
        inputs.push_back(input);
        outputs.push_back(output);
    }
}
