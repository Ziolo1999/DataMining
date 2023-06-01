#include <iostream>
using namespace std;
#include <stdlib.h>
#include <stdio.h>
#include "data.h"
#include "item.h"
#include "ndi.h"

int main(int argc, char *argv[])
{
    NDI ndi;
    
    ndi.setData(argv[1]);
    ndi.setMinsup(atoi(argv[2]));
    ndi.setIEdepth(atoi(argv[3]));
    if(argc==5) ndi.setOutput(argv[4]);
    
    cout << ndi.mine() << endl;
    return 0;
}
