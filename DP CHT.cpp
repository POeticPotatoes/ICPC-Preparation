#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define fir first
#define sec second
typedef long long ll;
using namespace std;

int n;
ll dp[1000005] = {0};
pair<ll, pair<ll, ll>> arr[1000005];

struct Line
{
	mutable ll m, c, p;
	bool operator<(const Line &o) const { return m < o.m; }
	bool operator<(ll x) const { return p < x; }
};

struct LineContainer : multiset<Line, less<>>
{
	static const ll inf = LLONG_MAX;
	ll div(ll a, ll b) { return a / b - ((a ^ b) < 0 && a % b); }

	bool isect(iterator x, iterator y)
	{
		if (y == end())
			return x->p = inf, 0;
		if (x->m == y->m)
			x->p = x->c > y->c ? inf : -inf;
		else
			x->p = div(y->c - x->c, x->m - y->m);
		return x->p >= y->p;
	}

	void add(ll m, ll c)
	{
		auto z = insert({m, c, 0}), y = z++, x = y;
		while (isect(y, z))
			z = erase(z);
		if (x != begin() && isect(--x, y))
			isect(x, y = erase(y));
		while ((y = x) != begin() && (--x)->p >= y->p)
			isect(x, erase(y));
	}

	ll query(ll x)
	{
		assert(!empty());
		auto l = *lower_bound(x);
		return l.m * x + l.c;
	}
};