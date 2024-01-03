#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define fir first
#define sec second
typedef long long ll;

using namespace std;

#pragma GCC optimize("Ofast,unroll-loops")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,avx,mmx,tune=native")

int n,q;
int mask[30005] = {0};
int arr[8][8] = {0};
int MOD = 1e9+7;

int mul(int a, int b)
{
	ll multi = (ll) a * (ll) b % (ll) MOD;
	return (int) multi;
}

int add(int a,int b) { return (a+b)%MOD;}

struct matrix
{
	int e[8][8];
	inline void init(int i)
	{
		for(int curr=0;curr<8;curr++)
		{
			for(int prv=0;prv<8;prv++)
			{   
				if(curr>mask[i]) e[prv][curr] = 0;
				if((mask[i]-curr)&(7-mask[i])) e[prv][curr] = 0;
                else e[prv][curr] = arr[prv][mask[i]-curr];
			}
		}
		return;
	}

	matrix operator * (matrix& other) const 
	{
		matrix ret;

		for(int i = 0; i < 8; i++){
			for(int j = 0; j < 8; j++){
				ret.e[i][j] = 0;
				for(int k = 0; k < 8; k++){
					ret.e[i][j] = add(ret.e[i][j], mul(e[i][k], other.e[k][j]));
				}
			}
		}

    	return ret;
	}
};
matrix iden;

vector<ll> ans;
vector<ll> vtmp;
inline void apply(matrix &a)
{
	vtmp.clear();
	vtmp.assign(8,0);
	for(int i=0;i<8;i++)
	{
		for(int j=0;j<8;j++)
		{
			vtmp[i] = (vtmp[i]+ans[j]*a.e[j][i])%MOD;
		}
	}
	ans.clear();
	ans = vtmp;
	return;
}

int N = 1<<15;
struct segtree
{
	vector<matrix> v;
	
	void init()
	{
		v.assign(2*N,iden);
		return;
	}

	void build()
	{
		for(int i=N-1;i>0;--i) v[i] = v[i<<1]*v[i<<1|1];
	}

	inline void upd(int pos,int node,int lx,int rx)
	{
		if(rx-lx==1) 
		{
			v[node].init(pos);
			return;
		}
		int mid = (lx+rx)>>1;
		if(pos<mid) upd(pos,node<<1,lx,mid);
		else upd(pos,(node<<1)|1,mid,rx);
		v[node] = v[node<<1]*v[(node<<1)|1];
		return;
	}

	inline void query(int l,int r,int node,int lx,int rx)
	{
		if(l>=rx || lx>=r) return;
		if(l<=lx && rx<=r) 
		{
			apply(v[node]);
			return;
		}
		int mid = (lx+rx)/2;
		if(r<=mid)
		{
			query(l,r,2*node,lx,mid);
			return;
		}
		else if(l>=mid) 
		{
			query(l,r,2*node+1,mid,rx);
			return;
		}
		else
		{
			query(l,r,2*node,lx,mid);
			query(l,r,2*node+1,mid,rx);
		}
	}
};

void solve()
{
	segtree st;
	st.init();
	for(int i=1;i<=n;i++)
	{
		matrix tmp;
		tmp.init(i);
		st.v[N+i] = tmp;
	}
	st.build();

	int type,l,r;
	for(int i=1;i<=q;i++)
	{
		scanf("%d %d %d",&type,&l,&r);
		if(type==2)
		{
			ans.clear();
			ans.assign(8,0); ans[0] = 1;
			st.query(l,r+1,1,0,N);

			ll ret = 0;
			for(int i=0;i<8;i++)
			{
				ret += ans[i];
				ret %= MOD;
			}	
			printf("%lld\n",ret);
		}
		else 
		{
			mask[r]^=(1<<(3-l));
			st.upd(r,1,0,N);
		}
	}
	return;
}

void brute()
{
	for(int prv=0;prv<8;prv++)
    {
        for(int curr=0;curr<8;curr++)
        {
            int mask = curr;
            int cnt = 0;
            if((prv&mask)==mask) cnt = 1;
            else cnt = 0;
            if(mask==7)
            {
				cnt = 0;
                if(prv&1) cnt++;
				if(prv&4) cnt++;
                if(prv==7) cnt++;
            }
            else if(mask==3 || mask==6) cnt++;

            arr[prv][curr] = cnt;
        }
    }

	for(int i=0;i<8;i++)
	{
		iden.e[i][i] = 1;
	}
	return;
}

void input()
{
	scanf("%d %d",&n,&q);
	char a[3][30005];
	for(int i=0;i<3;i++) {scanf("%s",&a[i]);}
	for(int i=1;i<=n;i++)
	{
		int t[3];
		for(int j=0;j<3;j++)
		{
			if(a[j][i-1]=='.') t[j] = 1;
			else t[j] = 0;
		}
		mask[i] = t[0]*4+t[1]*2+t[2];
		//cout << mask[i] << endl;
	}
	return;
}

int main()
{
	brute();
	input();
	solve();
 	return 0;
}
