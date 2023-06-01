#include <iostream>
using namespace std;
#include "data.h"
#include "item.h"
#include "ndi.h"

int main(int argc, char *argv[])
{
  NDI a;
  a.setData(argv[1]);
  a.setMinSup(atoi(argv[2]));
  a.setIEdepth(atoi(argv[3]));
  if(argc == 5) a.setOutputSets(argv[4]);
  int sets = a.generateSets();
  cout << sets << endl;
  return 0;
}
