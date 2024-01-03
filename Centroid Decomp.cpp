#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define fir first
#define sec second
typedef long long ll;

#pragma GCC optimize ("unroll-loops")
#pragma GCC target ("sse,sse2,sse3,ssse3,sse4,abm,avx,avx2,fma,mmx,tune=native")

using namespace std;
ll n,MOD;
ll sz[200005] = {0};
ll vis[200005] = {0};
ll ans = 0;
vector<pair<ll,ll>> conn[200005];
vector<pair<pair<ll,ll>,ll>> up_down;
ll invpow[200005] = {0};
ll pow10[200005] = {0};

inline ll totient(ll x)
{
	ll res = x;
	for (ll i = 2; i*i <= x; ++i)
	{
		if (x % i == 0)
		{
			res = res * (i-1) / i;
			while (x % i == 0) x /= i;
		}
	}
	if (x > 1) res = res * (x-1) / x;
 
	return res;
}

ll exp(ll a,ll pow)
{
	if(pow==0) return 1;
	if(pow==1) return a%MOD;
	ll tmp = exp(a,pow/2);
	tmp = (tmp*tmp)%MOD;
	if(pow%2==1) tmp = (tmp*a)%MOD;
	return tmp%MOD;
}

ll inv(ll x)
{
	return exp(x,totient(MOD)-1);
}

void precexp()
{
    ll tmp = inv(10);
    invpow[0] = 1;
    pow10[0] = 1;
    for(ll i=1;i<=200000;i++)
    {
        invpow[i] = (invpow[i-1]*tmp)%MOD;
        pow10[i] = (pow10[i-1]*10)%MOD;
    }
    return;
}

ll find_size(ll node,ll prv)
{
    ll size = 1;
    for(auto x:conn[node])
    {
        if(x.fir==prv) continue;
        size += find_size(x.fir,node);
    }
    return sz[node] = size;
}

ll find_centroid(ll node,ll prv,ll subsize)
{
    ll centroid = -1;
    for(auto x:conn[node])
    {
        if(x.fir==prv || vis[x.fir]==1) continue;
        if(sz[x.fir]>subsize/2) 
        {
            sz[node] -= sz[x.fir];
            sz[x.fir] += sz[node];
            centroid = find_centroid(x.fir,node,subsize);
        }
    }
    if(centroid==-1) 
    {
        centroid = node;
        vis[node] = 1;
    }
    return centroid;
}

inline void dfs(ll node,ll prv,ll up,ll down,ll dep)
{
    up_down.pb(mp(mp(up,down),dep));
    for(auto x:conn[node])
    {
        if(x.fir==prv || vis[x.fir]) continue;
        dfs(x.fir,node,(up+pow10[dep]*x.sec)%MOD,(10*down+x.sec)%MOD,dep+1);
    }
    return;
}

vector<ll> appear;
inline ll pivot(ll root)
{ 
    up_down.clear();
    dfs(root,-1,0,0,0);

	appear.clear();
	for(auto x:up_down)
	{
        appear.pb(x.fir.fir);
	}
    sort(appear.begin(),appear.end());

	ll tmp = 0;
	for(auto x:up_down)
	{
		ll val = invpow[x.sec]%MOD;
        val = (val*(MOD-(x.fir.sec)%MOD))%MOD;
		tmp += upper_bound(appear.begin(),appear.end(),val)-upper_bound(appear.begin(),appear.end(),val-1);
		if(val==x.fir.fir) tmp--; 
	}
    return tmp;
}

vector<pair<pair<ll,ll>,ll>> up_down2[100005];
ll fcnt;
inline void dfs2(ll node,ll prv,ll up,ll down,ll dep,bool isroot,ll cnt)
{
    up_down2[cnt].pb(mp(mp(up,down),dep));
    for(auto x:conn[node])
    {
        if(x.fir==prv || vis[x.fir]) continue;
        if(isroot) dfs2(x.fir,node,(up+pow10[dep]*x.sec)%MOD,(10*down+x.sec)%MOD,dep+1,0,++cnt);
        else dfs2(x.fir,node,(up+pow10[dep]*x.sec)%MOD,(10*down+x.sec)%MOD,dep+1,0,cnt);
    }
    if(isroot) fcnt = cnt;
    return;
}

inline ll dc(ll root)
{
    dfs2(root,-1,0,0,0,1,0);
    ll tmp = 0;
    for(ll i=1;i<=fcnt;i++)
    {
        appear.clear();
        for(auto x:up_down2[i])
        {
            appear.pb(x.fir.fir%MOD);
        }
        sort(appear.begin(),appear.end());

        for(auto x:up_down2[i])
        {
            ll val = invpow[x.sec]%MOD;
            val = (val*(MOD-(x.fir.sec)%MOD))%MOD;
            tmp += upper_bound(appear.begin(),appear.end(),val)-upper_bound(appear.begin(),appear.end(),val-1);
            if(val==x.fir.fir) tmp--; 
        }

        up_down2[i].clear();
    }
    return tmp;
}

void rec_centroid(ll curr)
{
    ll centroid = find_centroid(curr,-1,sz[curr]);

    ans += pivot(centroid);
    ans -= dc(centroid);

    for(auto x:conn[centroid])
    {
        if(vis[x.fir]) continue;
        rec_centroid(x.fir);
    }
    return;
}

void solve()
{
    cin >> n >> MOD;
    precexp();
    
    for(ll i=1;i<n;i++)
    {
        ll a,b,w;
        cin >> a >> b >> w;
		a++; b++;
        conn[a].push_back(mp(b,w));
        conn[b].push_back(mp(a,w));
    }

    find_size(1,-1);
    rec_centroid(1);
    cout << ans << endl;
}

int main()
{
    ::ios_base::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    solve();
    return 0;
}