############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project PredicMem25
add_files PredicMem25/src/include/config.hpp
add_files PredicMem25/src/include/const_expr.hpp
add_files PredicMem25/src/include/data_type.hpp
add_files PredicMem25/src/include/dictionary.hpp
add_files PredicMem25/src/include/gasp.hpp
add_files PredicMem25/src/include/global.hpp
add_files PredicMem25/src/include/input_buffer.hpp
add_files PredicMem25/src/include/svm.hpp
open_solution "original" -flow_target vivado
set_part {xczu3eg-sbva484-1-i}
create_clock -period 10 -name default
#source "./PredicMem25/original/directives.tcl"
#csim_design
csynth_design
#cosim_design
export_design -format ip_catalog
