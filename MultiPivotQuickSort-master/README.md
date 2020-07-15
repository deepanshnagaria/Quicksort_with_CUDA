# MultiPivotQuickSort
CUDA project with implementation QuictSort on GPU with multiple pivots for sorting integers.

AUTHORS
=======
	
	Deepansh Nagaria
	
REQUIREMENTS
============

	- CUDA device with 2.0 Compute Capability and above
	- CUDA drivers and CUDA toolkit
	- C++ compiler
	- make
	
COMPILATION & EXECUTION
=======================

	For C++
		1.	Compile program by make
		3.	Execute the program by
			hello.exe <input_file> <output_file> <id_gpu>
			Example launching: hello.exe input.csv output.csv 0
			
ORGANISATION OF DATA
====================
	First line: Number of elements to sort
	Second line: Element by element in one line separated by comma
