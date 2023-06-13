#include <iostream>
#include <algorithm>
using namespace std;
#include <stdio.h>
#include "data.h"
#include "item.h"
#include "ndi.h"

NDI::NDI() : data(0), out(0) {}

NDI::~NDI() 
{
	if(data) delete data;
	if(out) fclose(out);
}

double NDI::mine() 
{
	store = new sItem(0);
	set<Item> root;
	multiset<Item> *allitems = new multiset<Item>;  
	set<Item>::iterator it;
	int tnr=0;
	
	// read all transactions
	while(Transaction *t = data->getNext()) {
		for(int i=0; i<t->length;i++) {
			it = root.find(Item(t->t[i],t->t[i]));
			if(it == root.end()) it = root.insert(Item(t->t[i],t->t[i])).first;
			it->transactions.push_back(tnr);
		}
		tnr++;
		delete t;
	}
	store->Increment(tnr);
	if(out) fprintf(out,"(%d)\n",store->getSupport());
	
	// remove infrequent items and put items in support ascending order
	while((it = root.begin()) != root.end()) {
		if(it->transactions.size() >= minsup) {
			Item item(it->id, it->transactions.size());
			item.transactions = it->transactions;
			allitems->insert(item);
			print(&(it->id), 1, item.support, 0, tnr);
			if(item.support!=tnr)
			  storeSet(&(it->id), 1, item.support);
		}
		root.erase(it);
	}
	
	// finding all itemsets
	double added = 1+allitems->size();
	unsigned *itemset = new unsigned[allitems->size()];
	added += grow(allitems, itemset, 1);
	delete [] itemset;
	delete allitems;
	
	return added;
}

double NDI::grow(multiset<Item> *items, unsigned *itemset, unsigned depth)
{
	double added = 0;
	
	multiset<Item>::iterator it, it2;
	for(it = items->begin(); it != items->end(); it++) {
		multiset<Item> *children = new multiset<Item>;    
		itemset[depth-1] = it->id;
		for(it2 = items->begin(); it2 != it; it2++) {
			Item item(it2->id);
			itemset[depth] = it2->id;
			unsigned *sorted = new unsigned[depth+1];
			for(unsigned i=0;i<depth+1;i++) sorted[i] = itemset[i];
			sort(sorted,sorted+depth+1);
			
			if(checksubs(depth+1,sorted,store->getChildren())) {
				insert_iterator< vector<unsigned> > res_ins(item.transactions, item.transactions.begin());
				if(depth < 2) {
					set_intersection(it->transactions.begin(), it->transactions.end(), it2->transactions.begin(), it2->transactions.end(), res_ins);
					item.support = item.transactions.size();
				} else {
					if(depth == 2) 
						set_difference(it->transactions.begin(), it->transactions.end(), it2->transactions.begin(), it2->transactions.end(), res_ins);
        	else set_difference(it2->transactions.begin(), it2->transactions.end(), it->transactions.begin(), it->transactions.end(), res_ins);
        		item.support = it->support - item.transactions.size();
				}
			
				if(item.support >= minsup) {
					int l,u;
					unsigned cl = (iedepth>depth? 0:depth+1-iedepth);
					qie(cl, depth+1, sorted, l, u);					
					if(u-l) {
						print(sorted, depth+1, item.support, l, u);
						if((l!=(int)item.support) && (u!=(int)item.support)) 
							storeSet(sorted, depth+1,item.support);
							children->insert(item);
						added++;
					}
				}
			} 
			delete [] sorted;
		}
		added += grow(children, itemset, depth+1);
		delete children;
	}
	
	return added;
}

void NDI::storeSet(const unsigned *itemset, unsigned il, unsigned support)
{
	set<sItem> *current = store->makeChildren();
	for(unsigned k=0; k<il; k++) {
		set<sItem>::iterator it = current->find(sItem(itemset[k]));
		if(it == current->end()) {
			it = current->insert(sItem(itemset[k])).first; 
		}
		if(k+1==il) it->Increment(support);
		else current = it->makeChildren();
	}
}

void NDI::print(const unsigned *itemset, unsigned il, unsigned support, unsigned l, unsigned u)
{
	for(unsigned k=0; k<il; k++) 
		if(out) fprintf(out, "%d ", itemset[k]); 
	if(out) fprintf(out, "(%d) [%d,%d]\n", support, l, u);
}
