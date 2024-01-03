#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define fir first
#define sec second
typedef long long ll;

using namespace std;
ll n,sqrtn,m,k;
ll val[200005] = {0};
ll cnt[1800005] = {0};
ll ans[200005] = {0};
struct query
{
    int l,r,idx;
};
bool cmpMo(query lhs,query rhs)
{
    int l1 = lhs.l/sqrtn; int l2 = rhs.l/sqrtn;
    if(l1!=l2) return l1<l2;
    else return lhs.r<rhs.r;
    //return l1 != l2 ? l1 < l2 : lhs.r < rhs.r;
}

vector<query> queries;

void solve()
{
    cin >> n >> m >> k;
    sqrtn = sqrt(2*n);
    for(int i=1;i<=n;i++)
    {
        cin >> val[i];
        val[i] ^= val[i-1];
    }
    for(int i=1;i<=m;i++)
    {
        query tmp;
        cin >> tmp.l >> tmp.r;
        tmp.idx = i;
        queries.pb(tmp);
    }
    sort(queries.begin(),queries.end(),cmpMo);

    int l = 0, r = -1;
    ll currcnt = 0;
    for(auto q:queries)
    {
        q.l--;
        while(r<q.r)
        {
            ++r;
            currcnt += cnt[val[r]^k];
            cnt[val[r]]++;
        }
        while(r>q.r)
        {
            cnt[val[r]]--;
            currcnt -= cnt[val[r]^k];
            r--;
        }
        while (l < q.l)
        {
            cnt[val[l]]--;
            currcnt -= cnt[val[l]^k];
            l++;
        }
        while (l > q.l)
        {
            --l;
            currcnt += cnt[val[l]^k];
            cnt[val[l]]++;
        }
        ans[q.idx] = currcnt;
    }

    for(int i=1;i<=m;i++)
    {
        cout << ans[i] << endl;
    }
    return;
}

int main()
{
    solve();
    return 0;
}