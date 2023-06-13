#include <iostream>
using namespace std;
#include <stdio.h>
#include "data.h"
#include "item.h"
#include "ndi.h"


void NDI::qie(unsigned cl, unsigned sl, unsigned *iset, int &l, int &u)
{
	vector<int> V,P;
	if(cl==0) V.push_back(store->getSupport());	
	getsubs(sl,cl,iset,store->getChildren(),0,1,V,P);
	P.push_back(V.size());
	V.push_back(0);
	propagate(V,0,V.size(),P,0,P.size(),cl,sl);
	unsigned index=0;
	if(cl==0) {
		if(sl%2) {u = V[0];l=0;}
		else {l = (-V[0]<0?0:-V[0]);u=store->getSupport();}
		index++;
	} else { l=0; u=store->getSupport();}
	getBounds(V,l,u,sl,cl,0,1,index);
}

void NDI::getsubs(unsigned sl, unsigned min, unsigned *iset, set<sItem> *items, unsigned start, unsigned depth, vector<int> &V, vector<int> &P)
{
	unsigned stop = (depth<min? sl-(min-depth) : sl);
	if(depth<sl) {
		for(unsigned i=stop-1; i>=start && i<stop; i--) {
			set<sItem>::iterator runner = items->find(sItem(iset[i]));
			if(depth>=min) {
				if(depth>min) P.push_back(V.size());	
				V.push_back(runner->getSupport());
			}
			getsubs(sl,min,iset,runner->getChildren(),i+1,depth+1,V,P);
		}
	}
}

void NDI::propagate(vector<int> &V, int vb, int v, vector<int> &P, int pb, int p, int m, int n) 
{
	if(v==1) return;

	if(m==0) {
		for(int pos=2; pos<=v;pos*=2) {
			for(int i=vb; i<vb+v; i+=pos) {
				for(int j=0;j<pos/2;j++) {
					V[i+j] -= V[i+j+(pos/2)];
				}
			}
		}
	}
	else {
		int binom=0, v_split, p_split;
		binom = m*(v-p)/n;
		v_split = (v-binom)/2;
		p_split = v_split - ((n-m)*binom)/m;

		for(int i=0;i<v_split;i++) {
			V[vb+i] -= V[P[pb+p_split+i]];
		}

		if(v>2) { 
			if(v>3) propagate(V,vb,v_split,P,pb,p_split,m,n-1);
			propagate(V,vb+v_split,v-v_split,P,pb+p_split+(m?0:1),p-p_split-(m?0:1),(m?m-1:0),n-1);
		}
	}
}

bool NDI::checksubs(unsigned sl, unsigned *iset, set<sItem> *items, unsigned spos, unsigned depth)
{
	if(spos < sl-1) {
		if(items==0) return false;
		if(depth==0) {
			if(!checksubs(sl,iset,items,spos,depth+1)) return false;
		}
		set<sItem>::iterator runner = items->find(sItem(iset[spos+depth]));
		if(runner != items->end()) {
			if(!checksubs(sl,iset,runner->getChildren(),spos+1,depth)) return false;
		} else return false;
	}
	return true;
}

void NDI::getBounds(vector<int> &V, int &l, int &u, unsigned sl, unsigned min, unsigned start, unsigned depth, unsigned &index)
{
	unsigned stop = (depth<min? sl-(min-depth) : sl);
	if(depth<sl) 
		for(unsigned i=stop-1; i>=start && i<stop; i--) {
			if(depth>=min) {
				if((sl-depth)%2) u = (u<V[index]?u:V[index]);
				else l = (l>-V[index]?l:-V[index]);
				index++;
			}
			getBounds(V,l,u,sl,min,i+1,depth+1,index);
		}
}
