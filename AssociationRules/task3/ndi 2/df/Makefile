objects = data.o ndi.o qie.o testndi.o
CC = g++
flags = -g -O3 -Wall -ansi -pedantic

ndi: $(objects)
	$(CC) $(flags) $(objects) -o ndi

data.o: data.cpp data.h
	$(CC) $(flags) -c data.cpp

ndi.o: ndi.cpp data.h item.h ndi.h
	$(CC) $(flags) -c ndi.cpp

qie.o: qie.cpp item.h ndi.h
	$(CC) $(flags) -c qie.cpp

testndi.o: testndi.cpp data.h item.h ndi.h
	$(CC) $(flags) -c testndi.cpp

clean: 
	rm -f $(objects) ndi
