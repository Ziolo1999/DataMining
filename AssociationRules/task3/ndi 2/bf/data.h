/*----------------------------------------------------------------------
  File    : Data.h
  Contents: data set management
  Author  : Bart Goethals
----------------------------------------------------------------------*/

#include <stdio.h>

class Transaction
{
public:
  
  Transaction(int l) : length(l) {t = new unsigned[l];}
  Transaction(const Transaction &tr);
  ~Transaction(){delete t;}
  
  unsigned length;
  unsigned *t;
};

class Data
{
 public:
  
  Data(char *filename);
  ~Data();
  
  Transaction *getNext();
  
 private:
  
  FILE *in;
  char *fn;
  Transaction *trans;
  unsigned transize;
};
