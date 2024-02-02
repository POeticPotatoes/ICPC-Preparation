#include <bits/stdc++.h>
using namespace std;
#ifdef DEBUG
    #include </home/poeticpotato/work/cpp/debug.h>
#else
  #define deb(x...)
#endif
#define IO cin.sync_with_stdio(false); cin.tie(0); cout.tie(0);
#define FOR(i, a, b) for (ll i = (a); (i) < (b); (i)++)
#define FORN(i, a, b) for (ll i = (a); (i) <= (b); (i)++)
#define ROF(i, a, b) for (ll i = (a); (i) > (b); (i)--)
#define REP(i, n) FOR(i, 0, n)
#define all(x) (x).begin(), (x).end()
#define eb emplace_back
typedef long long ll;
typedef long double ld;
typedef vector<ll> vll;
template <typename T>
using vv = vector<vector<T>>;
template <typename T>
using vvv = vector<vv<T>>;
template <typename T, typename N>
using um = unordered_map<T, N>;
template <typename T>
using MinHeap = priority_queue<T, vector<T>, greater<T>>;
template <typename T>
using MaxHeap = priority_queue<T>;

const ll MOD[] = {999727999, 1070777777, 1000000007, 998244353};
mt19937_64 rng(chrono::system_clock::now().time_since_epoch().count());
const int M = MOD[2];
const int inf = (int)1e9;
const ll INF = 1e18;

const ll N = 1e8;

int P[N+1];
vll primes;

void init() {
    memset(P, 0, sizeof(int)*(N+1));
    FORN(i, 2, N) if (!P[i]) {
        primes.eb(i);
        for (int j=i+i;j<=N;j+=i)
            P[j] = 1;
    }
}

void solve() {
    ll n;
    cin>>n;
    if (!P[n]) return (void) (cout<<n<<" is prime.\n");
    cout<<"n = ";
    vll fac, exp;
    ll t = n;
    for (ll p : primes) {
        if (p>t) break;
        if (!(t%p)) {
            ll id = fac.size();
            fac.eb(p);
            exp.eb(0);
            while (!(t%p)) exp[id]++, t/=p;
        }
    }
    ll m = fac.size(), cnt=0, div=1;
    REP(i, m) {
        cout<<fac[i]<<"^"<<exp[i]<<"  ";
        cnt+=exp[i];
        div *= exp[i]+1;
    }
    cout<<"\nnumber of primes: "<<cnt<<"\n";
    cout<<"number of distinct primes: "<<m<<"\n";
    cout<<"number of divisors: "<<div<<"\n";
}

int main() {
    init();
    while (true) solve();
}
