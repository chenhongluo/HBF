/*------------------------------------------------------------------------------
Copyright © 2015 by Nicola Bombieri

H-BF is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#include "../include/GraphWeight.hpp"
#include <sstream>
#include <iterator>
#include <algorithm>
#include<random>

#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>

using namespace fUtil;

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
static std::default_random_engine generator(seed);
static std::uniform_int_distribution<int> distribution (1, 100);

namespace graph {

void GraphWeight::read(const char* File, const int nof_lines) {
	GraphBase::read(File, nof_lines);
}

void GraphWeight::readMatrixMarket(std::ifstream& fin, const int nof_lines) {
	fUtil::Progress progress(nof_lines);

	while (fin.peek() == '%')
		fileUtil::skipLines(fin);
	fileUtil::skipLines(fin);

	for (int lines = 0; lines < nof_lines; ++lines) {
		node_t index1, index2;
		fin >> index1 >> index2;

		OriginEdges[lines].x = index1 - 1;
        OriginEdges[lines].y = index2 - 1;
		OriginEdges[lines].z = distribution(generator);

		progress.next(lines + 1);
		fileUtil::skipLines(fin);
	}
	TriSize = nof_lines;

    ToCSR();
}


void GraphWeight::readDimacs10(std::ifstream& fin) {
	fUtil::Progress progress(V);
	while (fin.peek() == '%')
		fileUtil::skipLines(fin);
	fileUtil::skipLines(fin);

    OutNodes[0] = 0;
	int countEdges = 0;
	for (int lines = 0; lines < V; lines++) {
		std::string str;
		std::getline(fin, str);

		std::istringstream stream(str);
		std::istream_iterator<std::string> iis(stream >> std::ws);

		degree_t degree = std::distance(iis, std::istream_iterator<std::string>());
        std::istringstream stream2(str);
        for (int j = 0; j < degree; j++) {
            node_t dest;
            stream2 >> dest;
            dest--;

            OriginEdges[countEdges].x = lines;
            OriginEdges[countEdges].y = dest;
			OriginEdges[countEdges].z = distribution(generator);
            countEdges++;
        }
        progress.next(lines + 1);
	}
	TriSize = countEdges;
	ToCSR();
}


void GraphWeight::readSnap(std::ifstream& fin, const int nof_lines) {
	fUtil::Progress progress(nof_lines);
	while (fin.peek() == '#')
		fileUtil::skipLines(fin);

	fUtil::UniqueMap<node_t, node_t> Map;
	for (int lines = 0; lines < nof_lines; lines++) {
		node_t ID1, ID2;
		fin >> ID1 >> ID2;

		OriginEdges[lines].x = Map.insertValue(ID1);
        OriginEdges[lines].y = Map.insertValue(ID2);
		OriginEdges[lines].z = distribution(generator);

		progress.next(lines + 1);
	}
	TriSize = nof_lines;
    ToCSR();
}


void GraphWeight::readBinary(const char* File) {
	const int fileSize = fileUtil::fileSize(File);
	FILE *fp = fopen(File, "r");

	int* memory_mapped = (int*) mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fp->_fileno, 0);
	if (memory_mapped == MAP_FAILED)
		__ERROR("memory_mapped error");
	madvise(memory_mapped, fileSize, MADV_SEQUENTIAL);
	fUtil::Progress progress(fileSize);

	memory_mapped += 3;

	std::copy(memory_mapped, memory_mapped + V + 1, OutNodes);
	progress.perCent(V + 1);
	memory_mapped += V + 1;

	std::copy(memory_mapped, memory_mapped + V + 1, InNodes);
	progress.perCent((V + 1) * 2);
	memory_mapped += V + 1;

	std::copy(memory_mapped, memory_mapped + V, OutDegrees);
	progress.perCent((V + 1) * 2 + V);
	memory_mapped += V;

	std::copy(memory_mapped, memory_mapped + V, InDegrees);
	progress.perCent((V + 1) * 2 + V * 2);
	memory_mapped += V;

	std::copy(memory_mapped, memory_mapped + E, OutEdges);
	progress.perCent((V + 1) * 2 + V * 2 + E);
	memory_mapped += E;

	std::copy(memory_mapped, memory_mapped + E, InEdges);

	munmap(memory_mapped, fileSize);
	fclose(fp);
}

void GraphWeight::readDimacs9(std::ifstream& fin, const int nof_lines) {
	fUtil::Progress progress(nof_lines);

	char c;
	int lines = 0;
	std::string nil;
	while ((c = fin.peek()) != EOF) {
		if (c == 'a') {
			node_t index1, index2;
			fin >> nil >> index1 >> index2 >> Weights[lines];

            OriginEdges[lines].x = index1 - 1;
        	OriginEdges[lines].y = index2 - 1;
			OriginEdges[lines].z = Weights[lines];

			lines++;
			progress.next(lines + 1);
		}
		fileUtil::skipLines(fin);
	}
	TriSize = lines;
    ToCSR();
    dimacs9 = true;
}

#if __linux__

#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>

#endif

} //@graph
