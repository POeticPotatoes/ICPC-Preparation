#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef long double ld;
typedef pair<ll, ll> pll;
typedef vector<ll> vll;
template <typename T>
using t3 = tuple<T,T,T>;
template <typename T>
using t4 = tuple<T,T,T,T>;
typedef vector<ll> vl;
template <typename T>
using vv = vector<vector<T>>;
template <typename T>
using vvv = vector<vv<T>>;


#define MIN -1 // Change this if negative
class SegmentTree {
private:
    int n;

    vll A, st, lazy; // A -> Original Array
    int l(int p) { return (p<<1); }
    int r(int p) { return (p<<1)+1; }

    int conquer(ll a, ll b) {
        if (a==MIN) return b;
        if (b==MIN) return a;
        return min(a, b); // Change this if RMaxQ
    }

    void build(int p, int L, int R) {
        if (L==R) {
            st[p] = A[L];
            return;
        }
        int m = (L+R)/2;
        build(l(p), L,   m);
        build(r(p), m+1, R);
        st[p] = conquer(st[l(p)], st[r(p)]);
    }

    void propagate(int p, int L, int R) {
        if (lazy[p]==MIN) return;
        st[p] = lazy[p];
        if (L==R) 
            A[L] = lazy[p]; 
        else 
            lazy[l(p)] = lazy[r(p)] = lazy[p];
        lazy[p] = MIN;
    }

    ll RMQ(int p, int L, int R, int i, int j) {
        propagate(p, L, R);
        if (i>j) return MIN;
        if ((L>=i)&&(R<=j)) return st[p];
        int m = (L+R)/2;
        return conquer(RMQ(l(p), L,   m, i,           min(m, j)), 
                       RMQ(r(p), m+1, R, max(m+1, i), j       ));
    }

    void update(int p, int L, int R, int i, int j, ll val) {
        propagate(p, L, R);
        if (i>j) return;
        if ((L>=i)&&(R<=j)) {
            lazy[p] = val;
            propagate(p, L, R);
        } else {
            int m = (L+R)/2;
            update(l(p), L,   m, i,           min(m, j), val);
            update(r(p), m+1, R, max(m+1, i), j,         val);

            ll lsubtree = (lazy[l(p)] != MIN)? lazy[l(p)] : st[l(p)];
            ll rsubtree = (lazy[r(p)] != MIN)? lazy[r(p)] : st[r(p)];
            st[p] = (lsubtree <= rsubtree)? st[l(p)]: st[r(p)]; // Change if RMaxQ
        }
    }

public:
    SegmentTree(int sz): n(sz), st(4*n), lazy(4*n, MIN) {}

    SegmentTree(const vll &initialA): SegmentTree((int)initialA.size()) {
        A = initialA;
        build(1, 0, n-1);
    }

    void update(int i, int j, ll val) { update(1, 0, n-1, i, j, val); }

    int RMQ(int i, int j) { return RMQ(1, 0, n-1, i, j); }
};
/* TESTS
 * Uncomment this section to test the code
 * Expected output:
 * 13
 * 11
 * 15
 * =======
 * 13
 * 15
 * 15
 * =======
 * 30
 * 15
 * 15
 * =======
 * 7
 * 15
 * 7
 */

int main() {
    vll A = {18, 17, 13, 19, 15, 11, 20, 99};
    SegmentTree st(A);

    printf("%d\n", st.RMQ(1, 3)); // 13
    printf("%d\n", st.RMQ(4, 7)); // 11
    printf("%d\n", st.RMQ(3, 4)); // 15

    st.update(5,5,77);
    printf("=======\n%d\n", st.RMQ(1, 3)); // 13
    printf("%d\n", st.RMQ(4, 7)); // 15
    printf("%d\n", st.RMQ(3, 4)); // 15

    st.update(0,3,30);
    printf("=======\n%d\n", st.RMQ(1, 3)); // 30
    printf("%d\n", st.RMQ(4, 7)); // 15
    printf("%d\n", st.RMQ(3, 4)); // 15

    st.update(3,3,7);
    printf("=======\n%d\n", st.RMQ(1, 3)); // 7
    printf("%d\n", st.RMQ(4, 7)); // 15
    printf("%d\n", st.RMQ(3, 4)); // 7
}
