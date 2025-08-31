/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//    Copyright (c) 2024  Pablo S�nchez Cuevas                    //
//                                                                             //
//    This file is part of PredicMem23.                                            //
//                                                                             //
//    PredicMem23 is free software: you can redistribute it and/or modify          //
//    it under the terms of the GNU General Public License as published by     //
//    the Free Software Foundation, either version 3 of the License, or        //
//    (at your option) any later version.                                      //
//                                                                             //
//    PredicMem23 is distributed in the hope that it will be useful,               //
//    but WITHOUT ANY WARRANTY; without even the implied warranty of           //
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the              //
//    GNU General Public License for more details.                             //
//                                                                             //
//    You should have received a copy of the GNU General Public License        //
//    along with PredicMem23. If not, see <
// http://www.gnu.org/licenses/>.
//
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include<set>
#include<vector>

using namespace std;


class TraceReader {
public:
	string filename = "";
	ifstream file = ifstream();
	unsigned long currentLine = 0;

	string endLine = "#eof";

	TraceReader() {
		this->filename = "";
	}

	TraceReader(string filename) {
		this->filename = filename;
		file = ifstream(filename);
		file.open(filename);
	}
	
	TraceReader(const TraceReader& t) {
		this->filename = t.filename;
		try {
			this->file = ifstream(filename);
		}
		catch(...){
			this->file = ifstream();
		}
		this->currentLine = t.currentLine;
		this->endLine = t.endLine;
	}

	void copy(TraceReader& t) {
		t.filename = filename;
		t.file = ifstream(filename);
		t.file.open(filename);
	}

	TraceReader operator=(const TraceReader& t) {
		return TraceReader(t);
	}

	~TraceReader() {
		closeFile();
	}

	void closeFile() {
		file.close();
	}

	unsigned long countNumLines() {
		file.clear();
		file.seekg(0);
		string line;
		unsigned long res = 0;
		while (getline(file, line))
		{
			if (line.compare(endLine) == 0) {
				break;
			}
			res++;
			
				
		}
		file.clear();
		file.seekg(0);
		return res;
	}


	vector<string> readAllLines() {
		return readNextLines(countNumLines());
	}

	vector<string> readNextLines(unsigned long numLines) {
		string line;
		int k = 0;
		string delimiter = ": ", space = " ";
		string aux; 
		vector<string> res;


		long start = 0, end = numLines;

		if (file.is_open())
		{
			file.clear();
			while (getline(file, line))
			{
				if ((k >= end) || (line.compare(endLine) == 0)) break;
				else 
					if (k < end) {
						res.push_back(line);
					}
				k++;
				currentLine++;

			}
		}
		return res;
	}

};
