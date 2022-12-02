#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef long double ld;
typedef pair<ll, ll> pll;
template <typename T>
using t3 = tuple<T,T,T>;
template <typename T>
using t4 = tuple<T,T,T,T>;
typedef vector<ll> vl;
template <typename T>
using vv = vector<vector<T>>;
template <typename T>
using vvv = vector<vv<T>>;
#define FOR(i, a, b) for (ll (i) = (a); (i) < (b); (i)++)
#define ROF(i, a, b) for (ll (i) = (a); (i) > (b); (i)--)
#define REP(i, n) FOR(i, 0, (n))
#define all(x) (x).begin(), (x).end()
#define eb emplace_back
#define em emplace
#define mp make_pair
#define um unordered_map
#define lb_pos(arr, key) lower_bound(all(arr), key) - (arr).begin()
#define ub_pos(arr, key) upper_bound(all(arr), key) - (arr).begin()
#define FILL(arr, num) memset(arr, num, sizeof(arr))
#ifndef ONLINE_JUDGE
#define dbg(x...) do { cout << "\033[32;1m " << #x << " -> "; err(x); } while (0)
void err() { cout << "\033[39;0m" << endl; }
template<template<typename...> class T, typename t, typename... A>
void err(T<t> a, A... x) { for (auto v: a) cout << v << ' '; err(x...); }
template<typename T, typename... A>
void err(T a, A... x) { cout << a << ' '; err(x...); }
#else
#define dbg(...)
#define err(...)
#endif
template <typename T>
using MinHeap = priority_queue<T, vector<T>, greater<T>>;
template <typename T>
using MaxHeap = priority_queue<T>;
ll MOD = 1e9+7;
ll M(ll n) {return (n % MOD + MOD) % MOD;}
mt19937 rng((unsigned int) chrono::steady_clock::now().time_since_epoch().count());
#define uid(a, b) uniform_int_distribution<int>19(a, b)(rng)

ll n,a,b;
um<ll, ll> all, A, B;
vector<ll> nums;
vector<bool> visited, ans;

bool allMatched() {
    REP(i, n) if (!all.count(a-nums[i])) return 0;
    return 1;
}

bool check(ll i, bool odd, bool side) {
    visited[i] = 1;
    ans[i] = side;
    ll cur = nums[i];
    // odd = 1, side = 0
    // side = 1, odd = 0
    // look for an A
    if (odd^side) {
        if (A[i]==i) return true;
        if (!all.count(a-cur)) return !odd;
        return check(all[a-cur], !odd, side);
    }
    // look for a B
    if (B[i]==b) return true;
    if (!all.count(b-cur)) return !odd;
    return check(all[b-cur], !odd, side);
}

bool twoSAT() {
    REP(i, n) {
        if (visited[i]) continue;
        if (A[i]!=-1 && B[i]!=-1) continue;

        if (!check(i, 1, A[i]==-1)) return false;
    }

    REP(i, n)
        if (!visited[i]) return false;
    return true;
}

void solve() {
    cin>>n>>a>>b;
    nums = vector<ll>(n);
    visited = ans = vector<bool>(n);
    
    REP(i, n) {
        cin >> nums[i];
        all[nums[i]] = i;
    }

    // If both conditions are equal
    if (a==b) {
        if (!allMatched()) {
            cout<<"NO"<<endl;
            return;
        }
        cout<<"YES\n0";
        FOR(i,1,n) cout<<" 0";
        cout<<endl;
        return;
    }

    // Build adjacency list
    REP(i, n) {
        A[i] = (all.count(a-nums[i]))? all[a-nums[i]]: -1;
        B[i] = (all.count(b-nums[i]))? all[b-nums[i]]: -1;
    }

    if (!twoSAT()) {
        cout<<"NO"<<endl;
        return;
    }
    cout<<"YES\n";
    cout<<ans[0];
    FOR(i, 1, n) cout<<" "<<ans[i];
    cout<<endl;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int t=1;
    // cin >> t;
    while (t--) solve();
}
