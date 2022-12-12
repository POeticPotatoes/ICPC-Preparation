```c++
// Geometry  =====================
struct Point { double x, y; }                               // Change data type as required

// Rotation
int orientation(Point p, Point q, Point r) {                // Returns rotation of p->q->r
    auto val = (q.y - p.y) * (r.x - q.x) - 
               (q.x - p.x) * (r.y - q.y);
    if (!val) return 0;                                     // If the points are collinear
    return 2-(val>0);                                       // 1: clockwise, 2: counterclockwise
}

// Convex Hull (Jarvis' Algorithm)
vector<Point> convexHull(Point points[], int n) {
    if (n<3) return;                                        // Not a line
    vector<Point> hull;
    int l = 0;
    REP(i, n) if (points[i].x < points[l].x) l=i;
    int p = l, q;
    do {
        hull.push_back(points[p]);
        q = (p+1)%n;
        REP(i, n) if (orientation(points[p], points[i], points[q]) == 2) q=i;
        p = q;
    } while (p != l);
    return hull;
}

void test() {
    Point points[]{{0, 3}, {2, 2}, {1, 1}, {2, 1}, {3, 0}, {0, 0}, {3, 3}};
    auto hull = convexHull(points, 7);
    for (auto p: hull) printf("%d, %d\n", p.x, p.y);        // (0, 3), (0, 0), (3, 0), (3, 3)
}

// Scan Line Area Filling (With segment tree)

int lazy[maxn << 3];                                        // 标记了这条线段出现的次数
double s[maxn << 3];

struct node1 {
  double l, r;
  double sum;
} cl[maxn << 3];                                            // 线段树 (segment tree)

struct node2 {
  double x, y1, y2;
  int flag;
} p[maxn << 3];                                             // 坐标 (coordinates)

bool cmp(node2 a, node2 b) { return a.x < b.x; }

void pushup(int rt) {
    if (lazy[rt] > 0)
        cl[rt].sum = cl[rt].r - cl[rt].l;
    else
        cl[rt].sum = cl[rt * 2].sum + cl[rt * 2 + 1].sum;
}

void build(int rt, int l, int r) {
    if (r - l > 1) {
        cl[rt].l = s[l], cl[rt].r = s[r];
        build(rt * 2, l, (l + r) / 2);
        build(rt * 2 + 1, (l + r) / 2, r);
        pushup(rt);
    } else
        cl[rt].l = s[l], cl[rt].r = s[r], cl[rt].sum = 0;
    return;
}

void update(int rt, double y1, double y2, int flag) {
    if (cl[rt].l == y1 && cl[rt].r == y2) {
        lazy[rt] += flag;
        pushup(rt);
        return;
    }
    if (cl[rt * 2].r > y1) update(rt * 2, y1, min(cl[rt * 2].r, y2), flag);
    if (cl[rt * 2 + 1].l < y2)
        update(rt * 2 + 1, max(cl[rt * 2 + 1].l, y1), y2, flag);
    pushup(rt);
}

int test() {
    int temp = 1, n;
    double x1, y1, x2, y2, ans;
    scanf("%d", &n)
    ans = 0;
    for (int i = 0; i < n; i++) {
        scanf("%lf %lf %lf %lf", &x1, &y1, &x2, &y2);
        p[i].x = x1; p[i].y1 = y1; p[i].y2 = y2; p[i].flag = 1;
        p[i + n].x = x2; p[i + n].y1 = y1; p[i + n].y2 = y2; p[i + n].flag = -1;
        s[i + 1] = y1; s[i + n + 1] = y2;
    }
    sort(s + 1, s + (2 * n + 1));
    sort(p, p + 2 * n, cmp);
    build(1, 1, 2 * n);
    memset(lazy, 0, sizeof(lazy));
    update(1, p[0].y1, p[0].y2, p[0].flag);
    for (int i = 1; i < 2 * n; i++) {                           // Scan from 1 to 2n
        ans += (p[i].x - p[i - 1].x) * cl[1].sum;
        update(1, p[i].y1, p[i].y2, p[i].flag);
    }
    printf("%d", ans);
}

// 旋转卡壳 Rotary Jam (Longest diameter)
int sta[N], top;                        // 将凸包上的节点编号存在栈里，第一个和最后一个节点编号相同
bool is[N];                             // Assume points are stored in some array a[]

ll pf(ll x) { return x * x; }
ll dis(int p, int q) { return pf(a[p].x - a[q].x) + pf(a[p].y - a[q].y); }
ll sqr(int p, int q, int y) { return abs((a[q] - a[p]) * (a[y] - a[q])); }

ll get_longest() {  // 求凸包直径
    ll mx;
    int j = 3;
    if (top < 4) {
        mx = dis(sta[1], sta[2]);
        return;
    }
    for (int i = 1; i <= top; ++i) {
        while (sqr(sta[i], sta[i + 1], sta[j]) <=
               sqr(sta[i], sta[i + 1], sta[j % top + 1]))
            j = j % top + 1;
        mx = max(mx, max(dis(sta[i + 1], sta[j]), dis(sta[i], sta[j])));
    }
}

// Rotary Jam (Area)
void get_biggest() {
    int j = 3, l = 2, r = 2;
    double t1, t2, t3, ans = 2e10;
    for (int i = 1; i <= top; ++i) {
        while (sqr(sta[i], sta[i + 1], sta[j]) <=
               sqr(sta[i], sta[i + 1], sta[j % top + 1]))
            j = j % top + 1;
        while (dot(sta[i + 1], sta[r % top + 1], sta[i]) >=
               dot(sta[i + 1], sta[r], sta[i]))
        r = r % top + 1;
        if (i == 1) l = r;
        while (dot(sta[i + 1], sta[l % top + 1], sta[i]) <=
               dot(sta[i + 1], sta[l], sta[i]))
            l = l % top + 1;
        t1 = sqr(sta[i], sta[i + 1], sta[j]);
        t2 = dot(sta[i + 1], sta[r], sta[i]) + dot(sta[i + 1], sta[l], sta[i]);
        t3 = dot(sta[i + 1], sta[i + 1], sta[i]);
        ans = min(ans, t1 * t2 / t3);
    }
}

// 半平面交 Half Plane Intersection
friend bool operator<(seg x, seg y) {                       // Requires segment definition
    db t1 = atan2((x.b - x.a).y, (x.b - x.a).x);
    db t2 = atan2((y.b - y.a).y, (y.b - y.a).x);            // 求极角
    if (fabs(t1 - t2) > eps)                                // 如果极角不等
        return t1 < t2;
    return (y.a - x.a) * (y.b - x.a) > eps;         // 判断向量x在y的哪边，令最靠左的排在最左边
}

void half_plane() {
    // s[]是极角排序后的向量         s[] is a sorted array of vectors(segments)
    // q[]是向量队列                q[] is a queue of vectors(segments)
    // t[i]是s[i-1]与s[i]的交点     t[i] is the intersection of s[i-1] and s[i]
    // 【码风】队列的范围是(l,r]     bounds are (l, r]
    int l = 0, r = 0;
    for (int i = 1; i <= n; ++i)
        if (s[i] != s[i - 1]) {
            // 注意要先检查队尾
            while (r - l > 1 && (s[i].b - t[r]) * (s[i].a - t[r]) > eps)  
                --r;
            while (r - l > 1 && (s[i].b - t[l + 2]) * (s[i].a - t[l + 2]) > eps)  
                ++l;
            q[++r] = s[i];
            if (r - l > 1) t[r] = its(q[r], q[r - 1]);  // 求新交点
      }
    while (r - l > 1 &&
           (q[l + 1].b - t[r]) * (q[l + 1].a - t[r]) > eps)  // 注意删除多余元素
        --r;
    t[r + 1] = its(q[l + 1], q[r]);  // 再求出新的交点
    ++r;
}

// 平面最近点对 Nearest point pair on a plane
struct pt { int x, y, id; };                                // Custom definition of point
struct cmp_x {                                              // Custom comparator for sort()
    bool operator()(const pt& a, const pt& b) const {
        return a.x < b.x || (a.x == b.x && a.y < b.y); }};
struct cmp_y {
    bool operator()(const pt& a, const pt& b) const { return a.y < b.y; }};
int n;
vector<pt> a;
double mindist;
int ansa, ansb;
inline void upd_ans(const pt& a, const pt& b) {             // Calculates distance between points
    double dist = sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + .0);
    if (dist < mindist) mindist = dist, ansa = a.id, ansb = b.id;     // Update answer
}

void rec(int l, int r) {
    if (r - l <= 3) {
        for (int i = l; i <= r; ++i)
            for (int j = i + 1; j <= r; ++j) upd_ans(a[i], a[j]);
        sort(a + l, a + r + 1, &cmp_y);
        return;
    }

    int m = (l + r) >> 1;
    int midx = a[m].x;
    rec(l, m), rec(m + 1, r);
    inplace_merge(a + l, a + m + 1, a + r + 1, &cmp_y);

    static pt t[MAXN];
    int tsz = 0;
    for (int i = l; i <= r; ++i)
        if (abs(a[i].x - midx) < mindist) {
            for (int j = tsz - 1; j >= 0 && a[i].y - t[j].y < mindist; --j)
                upd_ans(a[i], t[j]);
            t[tsz++] = a[i];
        }
}

void sortPoints() {
    sort(a, a+n, &cmp_x);
    mindist = INF;
    rec(0, n-1);
}

// Math ==========================
// Extended Euclidean Algorithm
void ex_gcd(ll a, ll b, ll &d, ll &x, ll &y){
    if(b == 0) y = 0, x = 1, d = a;
    else ex_gcd(b, a % b, d, y, x), y -= a / b * x;
}

// Chinese Remainder Theorem
LL CRT(int k, LL* a, LL* r) {
  LL n = 1, ans = 0;
  for (int i = 1; i <= k; i++) n = n * r[i];
  for (int i = 1; i <= k; i++) {
    LL m = n / r[i], b, y;
    exgcd(m, r[i], b, y);  // b * m mod r[i] = 1
    ans = (ans + a[i] * m * b % n) % n;
  }
  return (ans % n + n) % n;
}

// Algorithms ====================
// Bellman-Ford
vector<int> BellmanFord(int graph[][3], int v, int e, int src) {
    vector<int> dis(v);
    memset(dis, inf, v);
    dis[src] = 0;
    REP(i, v-1) REP(j, e)
        if (dis[graph[j][0]] != inf && dis[graph[j][0]] + graph[j][2] <
            dis[graph[j][i]])
            dis[graph[j][1]] = dis[graph[j][0]] + graph[j][2];
    REP(i, e) {
        int x = graph[i][0], y = graph[i][1], weight = graph[i][2];
        if (dis[x] != inf && dis[x] + weight < dis[y])      // Negative weight check
            return null;
        return dis;
    }
}

void test() {
    int graph[][3] = {{0, 1, -1}, {0, 2, 4},
                      {1, 2, 3}, {1, 3, 2},
                      {1, 4, 2}, {3, 2, 5},
                      {3, 1, 1}, {4, 3, -3}};
    auto dis = BellmanFord(graph, 5, 8, 0);
    REP(i, 5) printf("%d ", dis[i]);                        // -1, 2, -2, 1
    printf("\n");
}

// 2-SAT Algorithm
ll n,a,b;
um<ll, ll> all, A, B;                                       // Build adjacency lists
vector<ll> nums;
vector<bool> visited, ans;

bool check(ll i, bool odd, bool side) {                     // odd-> first or second in pair
    visited[i] = 1;
    ans[i] = side;
    ll cur = nums[i];
    if (odd^side) {                                         // Satisfy first condition
        if (A[i]==i) return true;                           // Prevent self loops
        if (!all.count(a-cur)) return !odd;
        return check(all[a-cur], !odd, side);
    }                                                       // Else satisfy second condition
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

    REP(i, n) if (!visited[i]) return false;
    return true;
}

// Levenshtein Distance
int levenshtein(string s1, string s2) {
    int l1=s1.size(), l2.size();
    vv<int> dist(l2+1, vector<int>(l1+1));
    REP(i, l1+1) dist[0][i] = i;
    REP(i, l2+1) dist[i][0] = j;
    FOR(j, 1, l1+1)
        FOR(i, 1, l2+1) {
            bool track = (s1[i-1] != s2[j-1])
            int t=min((dist[i-1][j]+1), (dist[i][j-1]+1));
            dist[i][j] = min(t, (dist[i-1][j-1]+track));
        }
    return dist[l2][l1];
}

// Data Structures ===============
// Sparse table
int st[K+1][n]                                                  // K = logn/log2
void build() {
    FOR(i, 1, K) for (int j=0;j + (1<<i)) <= N; j++)
        st[i][k] = f(st[i-1][j], st[i-1][j + (1 << (i-1))]);    // f is the compare function
}

ll rsq(int l, int r) {
    ll sum = 0;
    ROF(i, K, -1)
        if ((1<<i) <= r-l+1)
            sum += st[i][l], l+=1<<i;
    return sum;
}

ll rmq(int l, int r) {
    int i = __builtin_clzll(1) - __builtin_clzll(r-l+1);
    return f(st[i][l], st[i][r - (1<<i)+1]);
}

// Fenwick Trees 1-based Fenwick Tree
class FenwickTree {
private:
    vll ft;
public:
    FenwickTree(int m) {ft.assign(m+1, 0);}
    void build(const vll &f) {
        int m=f.size(u);
        ft.assign(m+1, 0);
        for (int i=1;i<m;i++) {
            ft[i] += f[i];
            if (i + LSOne(i)<m)
                ft[i+LSOne(i)] += ft[i];
        }
    }
    FenwickTree(const vll &f) {build(f);}
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
    void update(int i, ll v) {                          // Update a single value by v
        for(;i<(int)ft.size();i+=LSOne(i))
            ft[i]+=v;
    }
    int select(ll k) {                                  // Binary search for upper bound
        int lo=1, hi=ft.size()-1;
        for (int i=0;i<30;i++) {                        // Works because 2^30 > 10^9
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

void test() {
    FenwickTree ft(vll{0,0,1,0,1,2,3,2,1,1,0});
    printf("%lld, %d, %lld, %lld\n",                        // 7, 7, 4, 0
        ft.rsq(1, 6), ft.select(7), ft.rsq(5), ft.rsq(1));
    ft.update(5, 1);
    printf("%lld\n", ft.rsq(1, 10));                        // 12
    RURQ rurq(10);
    rurq.range_update(2,9,7); rurq.range_update(6,7,3);
    printf("%lld, %lld\n", rurq.rsq(1,10), rurq.rsq(6, 7)); // 62, 20
}

// Segment Tree
#define MIN -INF
class SegmentTree {
private:
    int n;
    vll A, st, lazy;                                        // A -> Original Array
    int l(int p) { return (p<<1); }
    int r(int p) { return (p<<1)+1; }
    int conquer(ll a, ll b) {
        if (a==MIN) return b;
        if (b==MIN) return a;
        return min(a, b);                                   // Change this if RMaxQ
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
            st[p] = (lsubtree <= rsubtree)? st[l(p)]: st[r(p)];     // Change if RMaxQ
        }
    }
public:
    SegmentTree(int sz): n(sz), st(4*n), lazy(4*n, MIN) {}
    SegmentTree(const vll &initialA): SegmentTree((int)initialA.size()) {
        A = initialA;
        build(1, 0, n-1);
    }
    void update(int i, int j, ll val) { update(1, 0, n-1, i, j, val); }
    ll RMQ(int i, int j) { return RMQ(1, 0, n-1, i, j); }
};
void test() {
    SegmentTree st(vll{18, 17, 13, 19, 15, 11, 20, 99});
    printf("%lld, %lld, %lld\n", 
        st.RMQ(1, 3), st.RMQ(4, 7), st.RMQ(3, 4));              // 13, 11, 15
    st.update(5,5,77);
    printf("%lld, %lld, %lld\n",                                // 13, 15, 15
        st.RMQ(1, 3), st.RMQ(4, 7), st.RMQ(3, 4));
    st.update(0,3,30);
    printf("%lld, %lld, %lld\n",                                // 30, 15, 15
        st.RMQ(1, 3), st.RMQ(4, 7), st.RMQ(3, 4));
    st.update(3,3,7);
    printf("%lld, %lld, %lld\n",                                // 7, 15, 7
        st.RMQ(1, 3), st.RMQ(4, 7), st.RMQ(3, 4));
}
```
Snippets
```
Built-in bit functions (prefixed with __builtin_)
popcount - number of 1's          |  parity - whether the number of 1's is even
clz      - number of leading 0's  |  ctz    - distance from first to last 1
 
Random
srand((unsigned) time(NULL)) | rand()
 
Misc
memset(arr, val, size);
next_permutation(begin, end)
getline(cin, s)                    | stringstream ss(s); while(ss>var) {}

itertools(python) (import itertools, list(itertools.___))
permutations(vll)    - n!          |  chain(item1, item2,...)  - combine items
combinations(vll, i) - n choose i  |
```

Template
```c++
#include <bits/stdc++.h>
using namespace std;
#define IO cin.sync_with_stdio(false); cin.tie(0); cout.tie(0);
#define FOR(i, a, b) for (ll i = (a); (i) < (b); (i)++)
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

constexpr int MOD = 1e9+7;
constexpr int inf = (int)1e9;
constexpr ll INF = 1e18;

void solve() {
    
}

int main() {
    IO;
    int t=1;
    cin >> t;
    while (t--) solve();
}
```
VimConfig
```
syntax on
set nocompatible, showmatch, hlsearch, noswapfile, ignorecase, autoindent, tabstop=4, 
    expandtab, shiftwidth=4, softtabstop=4, relativenumber, number  
inoremap { {}<left>
inoremap {<BS> <nop>
inoremap {} {}
inoremap {<Esc> {<Esc>
inoremap {<Enter> {<CR>}<Esc>ko
nnoremap <silent> <Esc> :noh<cr>
```
Compile Command
`g++ -std=c++17 -Wshadow -Wall -o "${1}.out" "`
