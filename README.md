# PredicMem25

## What is this repository?

This repository contains all the C++ code of the PredicMem25 project: a hardware description of the Spatial Greedily Accurate SVM-based Prefetcher (SGASP) cache memory prefetcher in High Level Synthesis (HLS), oriented to specific-purpose. A full description of the project, its proposal and its implementation details is included [in this report file](report.pdf), which is written in a scientific paper style.

The SGASP prefetcher (along with the GASP family of prefetchers) was originally proposed and thoroughly described in a paper that is under peer-review right now. Nevertheless, its description is also detailed in the provided [report file](report.pdf).

Both the SGASP and the rest of prefetchers of the GASP family are based on the memory address predictor Support Vector Machine For Address Prediction (SVM4AP), presented in the publication: "Competitive Cost-effective Memory Access Predictor through Short-Term Online SVM and Dynamic Vocabularies", Sánchez-Cuevas et al., Future Generation Computer Systems, Volume 164, 2025, 107592,ISSN 0167-739X, https://doi.org/10.1016/j.future.2024.107592. (https://www.sciencedirect.com/science/article/pii/S0167739X24005569).

## Why HLS?

HLS is a hardware description method that exploits a high-level abstraction language (such as C++) to provide behavioral definition of the target hardware. Among the advantages that this provides, in this project we considerably leverage: (1) the ability to integrate testbenches based on complex, traced-based tests; and (2) the potential of C++ for the description of the hardware by using data structures, functions, classes, loops, etc, which greatly simplifies the design and development.

## How is the repository's source code structured?

All the code is stored under src/, which contains the following sub-directories:

### src/include/

It contains all the hardware descriptions of each of the modules that are described in [the report](report.pdf). It also contains the basic definition of the data types that are used, and the configuration of the hardware and the data types (including their bitfields).

### src/top/

It contains the descriptions of all possible top functions. A top function defines the interface of a target hardware plus its implementation. In this project, each top function is a synthesisable hardware module, either for testing (such as the input buffer, SVM and dictionary modules) or for packaging a prefetcher implementation as an IP core.

### src/testbench/

It contains the implementation of the trace-based testbench used for validating the HLS designs via either C-simulation (software) or co-simulation (software + hardware). It features a command-line interface for all requested operations.

## How can I import this repository as a Vitis HLS project?

This repository stores the project and solution files for importing it as a Vitis HLS project. Thus, it is as simple as opening Vitis HLS, open the directory as a project throught the option `Open project` and create a new solution that fits your platform. In this case, the Ultra96-V2 (incorporating the Zynq UltraScale+ MPSoC ZU3EG A484) was used. See https://github.com/Avnet/bdf to get its board definition file.

## How can I synthesise a hardware implementation?

Any of the top functions that are described in [src/top/src.cpp](src/top/src.cpp) can be synthesisable as a hardware module. In Vitis HLS, this can be done by adding a `#pragma HLS TOP name=<function name>` under the function header, and indicating the function name as top in `Project` > `Project Settings` > `Synthesis` > `Top funcion`. The module is synthesised by clicking in `Solution` > `Run synthesis`. Finally, it can exported as IP core in `Solution` > `Export as RTL`.

## How can I validate a hardware implementation?

Six different modules can be validated in the current version of PredicMem25: the input buffer, the dictionary, the SVM, the GASP prefetcher, the SGASP prefetcher and the BSGASP prefetcher. The validation of GASP and BSGASP prefetchers are in a work-in-progress-state, and thus they are not included in [the report](report.pdf).

The file [src/testbench/main.cpp](src/testbench/main.cpp) file can be modified to call to a custom implementation (given as top function) of a hardware module. For example, it is possible modify the code to call a top function of SGASP based on a single pipeline design, like `prefetchWithSGASPWithNopWithDataflow()`. The file [src/testbench/experimentation.hpp](src/testbench/experimentation.hpp) includes validation models for monitoring the functioning of the target module.

A trace header file has to be indicated through the command-line interface for the validation. In our work, the header is used for locating the trace files, and we employ a trace format for each hardware module to validate. They can be checked in the documentation in src/testbench/README.md. In this repository, the directory [traces_gasp/](traces_gasp/) contains the traces that were used for the experimentation shown in [the report](report.pdf).

Finally, in order to run the validation, either C-simulation or co-simulation are the available options in Vitis HLS. The C-simulation can be started simply by clicking `Project` > `Run C simulation`. In the case of a co-simulation, the target hardware module of src/top/top.cpp has to be synthesised first, and then the process can start by clicking `Solution` > `Run C/RTL Cosimulation`.

## Authorship and contact.

This repository was made thanks to Pablo Sánchez (owner of this repository), Antonio Pérez, Plácido Fernández, Dagnier Curra, Santiago Díaz and Antonio Ríos. The detailed authorship and contact information is presented in [the report file](report.pdf).

## Licensing

MIT License

Copyright (c) 2025 Pablo Sánchez Cuevas

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.