/*----------------------------------------------------------------------
  File    : Data.cpp
  Contents: data set management
  Author  : Bart Goethals
  ----------------------------------------------------------------------*/

#include "data.h"

Data::Data(char *filename)
{
  in = fopen(filename,"rt");
  transize = 1024;
  trans = new Transaction(transize);
}

Data::~Data()
{
  if(in) fclose(in);
  delete trans;
}

Transaction *Data::getNext()
{
  unsigned size=0;
  char c;

  // read list of items
  do {
    unsigned item=0, pos=0;
    c = getc(in);
    while((c >= '0') && (c <= '9')) {
      item *=10;
      item += int(c)-int('0');
      c = getc(in);
      pos++;
    }
    if(pos) {
      if(size==transize) {
	unsigned *tmp = trans->t;
	trans->t = new unsigned[transize+1024];
	for(unsigned i=0;i<transize;i++) trans->t[i] = tmp[i];
	transize += 1024;
	delete [] tmp;
      }
      trans->t[size++] = item;
    }
  }while(c != '\n' && !feof(in));
  
  trans->length = size;

  // if end of file is reached, rewind to beginning for next pass
  // Note, also last transaction must end with newline, 
  // else, it will be ignored
  if(feof(in)){
    rewind(in);
    return 0;
  }
  
  return trans;
}
