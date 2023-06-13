#include <vector>
#include <set>
using namespace std;

class Item
{
 public:

  Item(unsigned i, unsigned s = 0) : id(i), support(s) {}

  bool operator< (const Item &i) const {return support > i.support;}

  mutable vector<unsigned> transactions;
  unsigned id;
  unsigned support;
};

class sItem
{
public:

        sItem(int i) : id(i), support(0), children(0) {}
        sItem(const sItem &i) : id(i.id), support(i.support), children(i.children) {}
        ~sItem(){}

        unsigned getId() const {return id;}

        unsigned Increment(int inc) const {return support+=inc;}
        set<sItem> *makeChildren() const {
          if(children) return children;
          return children = new set<sItem>;
        }
        void deleteChildren() const {
          if(children) {
            for(set<sItem>::iterator it = children->begin(); it != children->end(); it++)
              it->deleteChildren();
            delete children;
            children = 0;
          }
        }

        unsigned getSupport() const {return support;}
        set<sItem> *getChildren() const {return children;}

        bool operator<(const sItem &i) const{return id < i.id;}

private:

        const unsigned id;
        mutable unsigned support;
        mutable set<sItem> *children;
};

