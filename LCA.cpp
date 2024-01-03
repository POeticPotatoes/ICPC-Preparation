#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define fir first
#define sec second
#define all(x) (x).begin(), (x).end()
typedef long long ll;

using namespace std;
//mt19937 rng(time(NULL));
int T[100005] = {0};
int L[100005] = {0};
int P[100005][105] = {0};
ll n;

int dfs(int node)
{
	if(L[node]!=-1) return L[node];
	if(T[node]==0) return L[node] = 0;
	return L[node] = dfs(T[node])+1;
}

void process()
{
	int i, j; 
	
	for (i = 1; i <= n; i++)
		P[i][0] = T[i]; 
	
	for (j = 1; (1 << j) <= n; j++)
		for (i = 1; i <= n; i++)
			if (P[i][j - 1] != -1) P[i][j] = P[P[i][j - 1]][j - 1];
}

int LCA(int p, int q)
{ 
	
	int tmp, log, i; 

	//swap
	if (L[p] < L[q]) tmp = p, p = q, q = tmp; 
	
	for (log = 1; 1 << log <= L[p]; log++);
		log--; 
	
	for (i = log; i >= 0; i--)
	{
		if (L[p] - (1 << i) >= L[q])
		{
			p = P[p][i]; 
		}	
	}
	
	if (p == q) return p;
	
	for (i = log; i >= 0; i--)
	{
		if (P[p][i] != -1 && P[p][i] != P[q][i])
		{
			p = P[p][i];
			q = P[q][i]; 
		}
	}
			
	return T[p];
}

void solve()
{
	memset(T,0,sizeof(T));
	memset(L,-1,sizeof(L));
	memset(P,-1,sizeof(P));
	cin >> n;
	for(int i=1;i<=n;i++)
	{
		int noc;
		cin >> noc;
		for(int c=0;c<noc;c++)
		{
			int tmp;
			cin >> tmp;
			T[tmp] = i;
		}
	}
	
	for(int i=1;i<=n;i++)
	{
		L[i] = dfs(i);
	}
	process();
	
	int q;
	cin >> q;
	for(int i=0;i<q;i++)
	{
		int a,b;
		cin >> a >> b;
		cout << LCA(a,b) << endl;
	}
	
	return;
}

int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(0); cout.tie(0);
	//freopen("1.in","r",stdin);
	//freopen("1.out","w",stdout);
	int tc=1;
	cin>>tc;
	for(int i=1;i<=tc;i++)
	{
		cout << "Case " << i << ":" << endl;
		solve();
	}
 	return 0;
}



