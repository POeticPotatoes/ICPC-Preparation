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
typedef vector<ll> vll;
#define LSOne(S) ((S) & -(S))

// 1-based Fenwick Tree
class FenwickTree {
private:
    vll ft;

public:
    FenwickTree(int m) {ft.assign(m+1, 0);}

    void build(const vll &f) {
        int m=f.size();
        ft.assign(m+1, 0);

        for (int i=1;i<m;i++) {
            ft[i] += f[i];
            if (i + LSOne(i)<m)
                ft[i+LSOne(i)] += ft[i];
        }
    }

    FenwickTree(const vll &f) {build(f);}

    // Counting value occurrences in vll
    // m - largest value
    FenwickTree(int m, const vll &s) {
        vll f(m+1);
        for (auto i:s) f[i]++;
        build(f);
    }

    ll rsq(int j) {
        ll ans=0;
        for (;j;j-=LSOne(j))
            ans+=ft[j];
        return ans;
    }

    ll rsq(int i, int j) { return rsq(j)-rsq(i-1); }

    // Updates a single value by v
    void update(int i, ll v) {
        for(;i<(int)ft.size();i+=LSOne(i))
            ft[i]+=v;
    }

    // Binary search for the first element greater than k
    int select(ll k) {
        int lo=1, hi=ft.size()-1;
        for (int i=0;i<30;i++) { // Works because 2^30 > 10^9
            int mid = (lo+hi)/2;
            (rsq(1,mid)<= k)? lo=mid: hi=mid;
        }
        return hi;
    }
};

class RUPQ {
private:
    FenwickTree ft;
public:
    RUPQ(int m) : ft(m) {}   
    
    void range_update(int ui, int uj, ll v) {
        ft.update(ui, v);
        ft.update(uj+1, -v);
    }

    ll point_query(int i) { return ft.rsq(i); }
};

class RURQ {
private:
    FenwickTree purq;
    RUPQ rupq;
public:
    RURQ(int m): purq(m), rupq(m) {}

    void range_update(int ui, int uj, ll v) {
        rupq.range_update(ui, uj, v);
        purq.update(ui, (ui-1)*v);
        purq.update(uj+1, -v*uj);
    }

    ll rsq(int j) { return rupq.point_query(j)*j - purq.rsq(j); }
    ll rsq(int i, int j) { return rsq(j)-rsq(i-1); }
};

/* TESTS
 * Uncomment this section to test the code
 * Expected output:
 * 7
 * 7
 * 4
 * 0
 * 12
 * =====
 * RSQ(1, 10) = 62
 * RSQ(6, 7) = 20
 */
int main() {
    vll f = {0,0,1,0,1,2,3,2,1,1,0};
    FenwickTree ft(f);
    printf("%lld\n", ft.rsq(1, 6)); // 7
    printf("%d\n", ft.select(7)); // 7
    printf("%lld\n", ft.rsq(5)); // 4
    printf("%lld\n", ft.rsq(1)); // 0
    ft.update(5, 1);
    printf("%lld\n", ft.rsq(1, 10)); // 12
    printf("=====\n");
    RUPQ rupq(10);
    RURQ rurq(10);
    rupq.range_update(2,9,7); // update 2-9 by 7
    rurq.range_update(2,9,7);
    rupq.range_update(6,7,3);
    rurq.range_update(6,7,3);
    // for (int i=1;i<=100;i++) printf("%d -> %lld\n", i, rupq.point_query(i));
    printf("RSQ(1, 10) = %lld\n", rurq.rsq(1,10)); // 62
    printf("RSQ(6,7) = %lld\n", rurq.rsq(6, 7)); // 20
}
