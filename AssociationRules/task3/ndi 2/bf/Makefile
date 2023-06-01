objects = nditest.o item.o ndi.o qie.o data.o
Cpp = g++
flags = -O3 -Wall -pedantic -ansi

ndi : $(objects) Makefile
	$(Cpp) $(flags) $(objects) -o ndi

nditest.o : nditest.cpp ndi.h
	$(Cpp) -c $(flags) nditest.cpp

item.o : item.cpp item.h
	$(Cpp) -c $(flags) item.cpp

ndi.o : ndi.cpp ndi.h data.h item.h
	$(Cpp) -c $(flags) ndi.cpp

qie.o : qie.cpp ndi.h data.h item.h
	$(Cpp) -c $(flags) qie.cpp

data.o : data.cpp data.h
	$(Cpp) -c $(flags) data.cpp

clean :
	rm -f $(objects) ndi

