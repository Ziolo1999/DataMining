/*----------------------------------------------------------------------
  File    : Item.h
  Contents: itemset management
  Author  : Bart Goethals
  Update  : 7/4/2003
----------------------------------------------------------------------*/

#include <set>
using namespace std;

struct Interval
{
  int l,u;
};

class Item
{
public:

	Item(int i) : id(i), support(0), children(0) {bound.l=0; bound.u=0;}
	Item(int i, Interval b) : id(i), support(0), bound(b), children(0) {}
	Item(const Item &i) : id(i.id), support(i.support), bound(i.bound), children(i.children) {}
	~Item(){}

	unsigned getId() const {return id;}

	int Increment(int inc = 1) const {return support+=inc;}
	set<Item> *makeChildren() const;
	int deleteChildren() const;

	unsigned getSupport() const {return support;}
	int getLB() const {return bound.l;}
	int getUB() const {return bound.u;}
	set<Item> *getChildren() const {return children;}
	void setUB(int u) const {bound.u=u;}

	bool operator<(const Item &i) const{return id < i.id;}

private:

	const unsigned id;

	mutable unsigned support;
	mutable Interval bound;
	mutable set<Item> *children;
};

class sItem
{
public:

        sItem(int i) : id(i), support(0), children(0) {}
        sItem(const sItem &i) : id(i.id), support(i.support), children(i.children) {}
        ~sItem(){}

        int getId() const {return id;}

        int Increment(unsigned inc = 1) const {return support+=inc;}
        set<sItem> *makeChildren() const;
        int deleteChildren() const;

        unsigned getSupport() const {return support;}
        set<sItem> *getChildren() const {return children;}

        bool operator<(const sItem &i) const{return id < i.id;}

private:

        const int id;
        mutable unsigned support;
        mutable set<sItem> *children;
};


