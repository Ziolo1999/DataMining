/*----------------------------------------------------------------------
  File    : ndi.h
  Contents: ndi algorithm for finding all non derivable frequent sets
  Author  : Bart Goethals
  Update  : 7/4/2003
  ----------------------------------------------------------------------*/

#include <set>
#include <map>
#include <vector>
#include <fstream>
using namespace std;

class NDI
{
 public:

  NDI();
  ~NDI();

  void setData(char *fn){data = new Data(fn);}
  void setOutputSets(char *fn){setsout.open(fn);}
  void setMinSup(unsigned ms){minsup=ms;}
  void setIEdepth(unsigned ied){iedepth=ied;}
  int generateSets();

 protected:
	
  int generateCandidates(unsigned level);
  int generateCandidates(unsigned level, set<Item> *items, unsigned depth, unsigned *current);
  int pruneNodes(unsigned level);
  int pruneNodes(unsigned level, set<Item> *items, unsigned depth);

  bool checksubs(unsigned sl, unsigned *iset, set<sItem> *items, unsigned spos=0, unsigned depth=0);
  void qie(unsigned cl, unsigned sl, unsigned *iset, int &l, int &u);
  void getsubs(unsigned sl, unsigned min, unsigned *iset, set<sItem> *items, unsigned start, unsigned depth, vector<int> &V, vector<int> &P);
	void propagate(vector<int> &V, int vb, int v, vector<int> &P, int pb, int p, int m, int n);
  void getBounds(vector<int> &V, int &l, int &u, unsigned sl, unsigned min, unsigned start, unsigned depth, unsigned &index);

  int countCandidates(unsigned level);
  int processTransaction(unsigned level, Transaction *t, set<Item> *items, unsigned spos, unsigned depth);
  int processTransaction1(unsigned level, Transaction *t, set<Item> *items, unsigned spos, unsigned depth, unsigned suppfirstitem=0);
  int processTransaction2(unsigned level, Transaction *t, set<Item> *items, unsigned spos, unsigned depth);
  void ReOrder();

  int pruneCandidates(unsigned level);
  int pruneCandidates(unsigned level, set<Item> *items, unsigned depth, unsigned *itemset, set<sItem> *sitems);
  
  void printSet(unsigned *itemset, unsigned length, unsigned supp, int lb, int ub);

 private:

  Data *data;
  unsigned minsup;
  unsigned iedepth;
  Item *trie;
  sItem *store;
  map<int,int> norder, oorder;
  ofstream setsout;
};
