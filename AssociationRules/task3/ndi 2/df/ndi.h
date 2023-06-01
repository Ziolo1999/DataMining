#include <set>
using namespace std;

class NDI
{
public:
	
	NDI();
	~NDI();
	
	void setData(char *file){data = new Data(file);}
	void setMinsup(unsigned ms){minsup = ms;}
	void setOutput(char *of){out = fopen(of, "wt");}
	void setIEdepth(unsigned ied){iedepth=ied;}
	
	double mine();
		
protected:
	
	double grow(multiset<Item> *items, unsigned *itemset, unsigned depth);
	void print(const unsigned *itemset, unsigned il, unsigned support, unsigned l, unsigned u);	
	void storeSet(const unsigned *itemset, unsigned il, unsigned support);
	bool checksubs(unsigned sl, unsigned *iset, set<sItem> *items, unsigned spos=0, unsigned depth=0);

	void qie(unsigned cl, unsigned sl, unsigned *iset, int &l, int &u);
	void getsubs(unsigned sl, unsigned min, unsigned *iset, set<sItem> *items, unsigned start, unsigned depth, vector<int> &V, vector<int> &P);
	void propagate(vector<int> &V, int vb, int v, vector<int> &P, int pb, int p, int m, int n);
	void getBounds(vector<int> &V, int &l, int &u, unsigned sl, unsigned min, unsigned start, unsigned depth, unsigned &index);

	sItem *store;
	unsigned minsup, iedepth;
	Data *data;	
	FILE *out;
};
