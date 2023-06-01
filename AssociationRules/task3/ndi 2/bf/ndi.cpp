#include <time.h>
#include <algorithm>
#include <iostream>
#include <time.h>
#include <math.h>
using namespace std;
#include "data.h"
#include "item.h"
#include "ndi.h"

NDI::NDI() : data(0), minsup(0)
{
	trie = new Item(0);
	store = new sItem(0);
}

NDI::~NDI()
{
	if(data) delete data;
	if(trie) {
		trie->deleteChildren();
		delete trie;
	}
	if(store) {
		store->deleteChildren();
		delete store;
	}
}

int NDI::generateSets()
{
	unsigned total=1, pass=0, pruned=0;

	do {
		pass++;  cout << pass << " " << flush;
		int generated=0;
		if(pass>2) {
			generated = generateCandidates(pass);
			pruneNodes(pass);
		}
		cout << generated << " " << flush;
		if (pass<=2 || generated) {
			countCandidates(pass);
			pruned = pruneCandidates(pass);
			total += pruned; 
		} 
		else pruned = 0;
		cout << pruned << " " << flush;

		if(pass==1) {
			ReOrder();
		}	
		cout << endl;
	} while(pruned>pass);

	return total;
}

int NDI::generateCandidates(unsigned level)
{
	unsigned *tmp = new unsigned[level];
	int generated = generateCandidates(level, trie->getChildren(), 1, tmp);
	delete [] tmp;

	return generated;
}

int NDI::generateCandidates(unsigned level, set<Item> *items, unsigned depth, unsigned *current)
{
	if(items == 0) return 0;
	int generated = 0;

	if(depth == level-1) {
		for(set<Item>::reverse_iterator runner = items->rbegin(); runner != items->rend(); runner++) {
			current[depth-1] = runner->getId();
			set<Item> *children = runner->makeChildren();
			for(set<Item>::reverse_iterator it = items->rbegin(); it != runner; it++) {
				current[depth] = it->getId();
				if(checksubs(depth+1,current,store->getChildren())) {	  
					Interval lu;
					unsigned cl = (iedepth>depth? 0:depth+1-iedepth);
					qie(cl,depth+1,current,lu.l,lu.u);	
					//cout << lu.l << " " << lu.u << endl << endl;
					if((lu.u >= (int)minsup) && (lu.u-lu.l)) {
						children->insert(Item(it->getId(),lu));
						generated++;
					}
				}
			}
		}
	}
	else {
		for(set<Item>::reverse_iterator runner = items->rbegin(); runner != items->rend(); runner++) {
			current[depth-1] = runner->getId();
			generated += generateCandidates(level, runner->getChildren(), depth+1, current);
		}
	}

	return generated;
}

int NDI::pruneNodes(unsigned level)
{
	return pruneNodes(level,trie->getChildren(),1);
}

int NDI::pruneNodes(unsigned level, set<Item> *items, unsigned depth)
{
	if(items == 0) return 0;
	int nodes = 0;

	if(depth==level) nodes = items->size();
	else {
		for(set<Item>::iterator runner = items->begin(); runner != items->end(); ) {
			int now = pruneNodes(level, runner->getChildren(), depth+1);
			if(now) {
				nodes += now;
				nodes++;
				runner++;
			}
			else {
				runner->deleteChildren();
				set<Item>::iterator tmp = runner++;
				items->erase(tmp);
			}
		}
	}

	return nodes;
}

int NDI::countCandidates(unsigned level)
{
	int trans=0;

	// count all single items
	if(level==1) {
		while(Transaction *t = data->getNext()) {
			trie->Increment();

			unsigned *iset = t->t, sl = t->length;
			set<Item> *items = trie->makeChildren();
			for(unsigned i=0; i<sl; i++) {
				Item item(iset[i]);
				set<Item>::iterator runner = items->find(item);
				if(runner == items->end()) runner = (items->insert(item)).first;
				runner->Increment();
			}
			trans++;
		}
		store->Increment(trie->getSupport());		
	}
	else {
		while(Transaction *t = data->getNext()) {
			if(t->length >= level) {
				// Reorder transaction
				vector<int> list;
				for(unsigned i=0; i<t->length; i++) 
					if(norder.count(t->t[i])) 
						list.push_back(norder[t->t[i]]);
				int size=list.size();
				sort(list.begin(), list.end());
				for(int i=0; i<size; i++) t->t[i] = list[i];
				t->length = size;
				if(t->length >= level)
					if(processTransaction(level, t, trie->getChildren(), 0, 1)) trans++;
			}
		}
	}
	return trans;
}

int NDI::processTransaction(unsigned level, Transaction *t, set<Item> *items, unsigned spos, unsigned depth)
{
	if( (level==2) || (items->size() > t->length-level+depth-spos)) return processTransaction1(level,t,items,spos,depth); 
	else return processTransaction2(level,t,items,spos,depth);
}

int NDI::processTransaction2(unsigned level, Transaction *t, set<Item> *items, unsigned spos, unsigned depth)
{
	if(items == 0) return 0;
	unsigned max = t->length-level+depth;
	int used=0;
	for(set<Item>::iterator it = items->begin(); spos<max  && it!=items->end(); it++) {
		spos = lower_bound(&t->t[spos], &t->t[max], it->getId()) - t->t;
		if(spos<max && (t->t[spos]==it->getId())) {
			if(depth==level) { 
				it->Increment();
				used++;
			}
			else used += processTransaction(level,t,it->getChildren(),spos+1,depth+1);
			spos++;
		}
	}	
	return used;
}

int NDI::processTransaction1(unsigned level, Transaction *t, set<Item> *items, unsigned spos, unsigned depth, unsigned suppfirstitem)
{
	if(items == 0) return 0;
	int used=0;
	unsigned max = t->length-level+depth, *iset=t->t;
	set<Item>::iterator runner;

	while(spos<max) {
		runner = items->find(Item(iset[spos]));
		if(runner != items->end()) {
			if(depth==level) {
				runner->Increment();
				used++;
			}
			else {
				if(depth==1 && level==2) {
					runner->makeChildren();
					used += processTransaction1(level, t, runner->getChildren(), spos+1, depth+1,runner->getSupport());
				}
				else  used += processTransaction(level, t, runner->getChildren(), spos+1, depth+1);   
			}
		}
		else if(depth==2 && level==2) {
			set<Item> *singles = trie->getChildren();
			runner = singles->find(Item(iset[spos]));
			if(runner != singles->end()) {
				Interval lu;
				lu.u = (suppfirstitem < runner->getSupport()? suppfirstitem : runner->getSupport());
				lu.l = suppfirstitem + runner->getSupport() - trie->getSupport();
				if(lu.l<0) {lu.l=0;}

				if(lu.u >= (int)minsup) {
					if(lu.u-lu.l) {
						runner = items->insert(Item(iset[spos],lu)).first;
						runner->Increment();
						used++;
					}
				}
			}
		}
		spos++;
	}   

	return used;
}

int NDI::pruneCandidates(unsigned level)
{
	unsigned *tmp = new unsigned[level];
	int pruned = pruneCandidates(level,trie->getChildren(),1,tmp,store->makeChildren());
	delete [] tmp;
	return pruned;
}

int NDI::pruneCandidates(unsigned level, set<Item> *items, unsigned depth, unsigned *itemset, set<sItem> *sitems)
{
	if(items == 0) return 0;
	int left = 0;
	if(level==1)
		printSet(0,0,trie->getSupport(),trie->getSupport(),trie->getSupport());

	for(set<Item>::iterator runner = items->begin(); runner != items->end(); ) {
		itemset[depth-1] = runner->getId();

		if(depth == level) {	
			if(depth==1) runner->setUB(trie->getSupport());

			if(runner->getSupport() < minsup) {
				runner->deleteChildren();
				set<Item>::iterator tmp = runner++;
				items->erase(tmp);
			}
			else {
				if(setsout.is_open()) printSet(itemset, depth, runner->getSupport(), runner->getLB(), runner->getUB());
				left++;

				if(((int)runner->getSupport() == runner->getLB()) || ((int)runner->getSupport() == runner->getUB())) {
					set<Item>::iterator tmp = runner++;
					items->erase(tmp);
				}
				else {
				  set<sItem>::iterator srunner = sitems->insert(sItem(runner->getId())).first;
				  srunner->Increment(runner->getSupport());      
					runner++;
				}
			}
		}
		else {	
			set<sItem>::iterator srunner = sitems->find(sItem(runner->getId()));
			int now = pruneCandidates(level, runner->getChildren(), depth+1, itemset, srunner->makeChildren());
			if(now) {
				left += now;
				runner++;
			}
			else {
				runner->deleteChildren();
				set<Item>::iterator tmp = runner++;
				items->erase(tmp);
			}
		}
	}

	return left;
}

void NDI::printSet(unsigned *itemset, unsigned length, unsigned supp, int lb, int ub)
{
	for(unsigned j=0; j<length; j++) 
		if(length>1) {
			setsout << oorder[itemset[j]] << " ";
		}
		else setsout << itemset[j] << " ";
		setsout << "(" << supp << ") [" << lb << "," << ub << "]" << endl;
}

void NDI::ReOrder()
{
	set<Item> *src = trie->getChildren();
	set<sItem> *dst = store->makeChildren();
	multimap<int,int> tmporder;
	for(set<Item>::iterator it1 = src->begin();it1 != src->end(); it1++)
		tmporder.insert(pair<int,int>(it1->getSupport(),it1->getId()));
	src->clear();
	dst->clear();
	int counter=1;
	for(multimap<int,int>::iterator it2 = tmporder.begin(); it2 != tmporder.end(); it2++) {
		norder[it2->second] = counter;
		oorder[counter] = it2->second;
		Item tmpitem(counter);
		sItem tmpsitem(counter);
		tmpitem.Increment(it2->first);
		tmpsitem.Increment(it2->first);
		src->insert(tmpitem);
		dst->insert(tmpsitem);
		counter++;
	}
}
