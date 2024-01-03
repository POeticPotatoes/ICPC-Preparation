#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define fir first
#define sec second
typedef long long ll;
 
using namespace std;
int n,k;
ll arr[4005][4005] = {0};
ll cost[4005][4005] = {0};
ll dp[805][4005] = {0};
ll INF = 1e9;
 
ll calc(int prv,int ind)
{
    return cost[prv][ind];
}
 
void dpdnc(int i,int l,int r,int optl,int optr)
{
    if(l>r || optl>optr) return;
    int mid = (l+r)/2;
    
    int opt = optl;
    dp[i][mid] = INF;
    for(int k=optl;k<=min(mid-1,optr);k++)
    {
        ll val = dp[i-1][k]+calc(k+1,mid);
        if(dp[i][mid]>val)
        {
            dp[i][mid] = val;
            opt = k;
        }
    }
    dpdnc(i,l,mid-1,optl,opt);
    dpdnc(i,mid+1,r,opt,optr);
}

const int bufsz = 40101010;
struct fastio{
	char buf[bufsz];
	int cur;
	fastio(){
		cur = bufsz;
	}
	inline char nextchar(){
		if(cur==bufsz){
			fread(buf, bufsz, 1, stdin);
			cur=0;
		}
		return buf[cur++];
	}
	inline int nextint(){
		int x = 0;
		char c = nextchar();
		while(!('0' <= c && c <= '9')){
			c = nextchar();
		}
		while('0' <= c && c <= '9'){
			x = x*10+c-'0';
			c = nextchar();
		}
		return x;
	}
}io;
 
void solve()
{
    n = io.nextint(); k = io.nextint();
    for(int i=1;i<=n;i++)
    {
        for(int j=1;j<=n;j++)
        {
            arr[i][j] = io.nextint();
            arr[i][j] += arr[i][j-1];
        }
    }
    
    for(int i=1;i<=n;i++)
    {
        cost[i][i] = 0;
        for(int j=i+1;j<=n;j++)
        {
            cost[i][j] = cost[i][j-1]+arr[j][j]-arr[j][i-1];
        }
    }
 
    for(int i=0;i<=k;i++)
    {
        for(int j=0;j<=n;j++)
        {
            dp[i][j] = INF;
        }
    }
    dp[0][0] = 0;
 
    for(int i=1;i<=k;i++)
    {
        dpdnc(i,0,n,0,n);
    }
    cout << dp[k][n] << endl;
	return;
}
 
int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(0); cout.tie(0);
	int tc=1;
	//cin>>tc;
	for(int i=1;i<=tc;i++)
	{
		solve();
	}
 	return 0;
}