#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define fir first
#define sec second
typedef long long ll;

using namespace std;

int gn,gm;
int grid[305][305] = {0};
int curr = 1;
int cnt = 0;
int noden[305][305][4] = {0};
set<int> hori,verti;
vector<int> conn[200005];
int src,sink; 

bool black(int i,int j)
{
    return grid[i][j];
}

struct Flow_Edge 
{
    int v,u;
    ll cap,flow = 0;
    Flow_Edge(int v, int u, ll cap) : v(v), u(u), cap(cap) {}
};

struct Dinic 
{
    const ll flow_inf = 1e18;
    int n, s, t;
    int m = 0;
    vector<Flow_Edge> edges;
    vector<vector<int>> conn;
    vector<int> level, ptr;
    queue<int> q;

    Dinic(int n, int s, int t) : n(n), s(s), t(t) 
    {
        conn.resize(n);
        level.resize(n);
        ptr.resize(n);
    }

    void add_edge(int v, int u, ll cap) 
    {
        edges.push_back({v,u,cap});
        edges.push_back({u,v,0});
        conn[v].push_back(m);
        conn[u].push_back(m+1);
        m += 2;
    }

    bool bfs() 
    {
        while (!q.empty()) 
        {
            int v = q.front();
            q.pop();
            for (int id:conn[v])
            {
                if (edges[id].cap-edges[id].flow<1 || level[edges[id].u]!=-1) continue;
                level[edges[id].u] = level[v] + 1;
                q.push(edges[id].u);
            }
        }
        return level[t] != -1;
    }

    ll dfs(int v, ll pushed) 
    {
        if (pushed == 0) return 0;
        if (v == t) return pushed;
        for (int& cid = ptr[v]; cid < conn[v].size(); cid++) 
        {
            int id = conn[v][cid];
            int u = edges[id].u;
            if (level[v]+1!=level[u] || edges[id].cap-edges[id].flow<1) continue;
            ll tr = dfs(u, min(pushed,edges[id].cap-edges[id].flow));
            if (tr == 0) continue;
            edges[id].flow += tr;
            edges[id^1].flow -= tr;
            return tr;
        }
        return 0;
    }

    ll flow() 
    {
        ll f = 0;
        while (true) 
        {
            fill(level.begin(),level.end(),-1);
            q.push(s);
            level[s] = 0;
            if (!bfs()) break;
            fill(ptr.begin(), ptr.end(), 0);
            while (ll pushed = dfs(s, flow_inf)) f += pushed;
        }
        return f;
    }
};

void solve()
{
    cin >> gn >> gm;
    for(int i=1;i<=gn;i++)
    {
        string s;
        cin >> s;
        for(int j=1;j<=gm;j++)
        {
            if(s[j-1]=='#') 
            {
                grid[i][j] = 1;
                cnt++;
            }
        }
    }

    for(int i=1;i<=gn;i++)
    {
        for(int j=1;j<=gm;j++)
        {
            int up = -1;
            int down = -1;
            int left = -1;
            int right = -1;
            
            if(black(i,j) && black(i-1,j))
            {
                int tmp = curr;
                if(noden[i][j][0]!=0) tmp = noden[i][j][0];
                else if(noden[i-1][j][2]!=0) tmp = noden[i-1][j][2];
                noden[i][j][0] = tmp;
                noden[i-1][j][2] = tmp;
                up = tmp;
                verti.insert(tmp);
                curr++;
            }

            if(black(i,j) && black(i+1,j))
            {
                int tmp = curr;
                if(noden[i][j][2]!=0) tmp = noden[i][j][2];
                else if(noden[i+1][j][0]!=0) tmp = noden[i+1][j][0];
                noden[i][j][2] = curr;
                noden[i+1][j][0] = curr;
                down = tmp;
                verti.insert(tmp);
                curr++;
            }

            if(black(i,j) && black(i,j-1))
            {
                int tmp = curr;
                if(noden[i][j][3]!=0) tmp = noden[i][j][3];
                else if(noden[i][j-1][1]!=0) tmp = noden[i][j-1][1];
                noden[i][j][3] = curr;
                noden[i][j-1][1] = curr;
                left = tmp;
                hori.insert(tmp);
                curr++;
            }

            if(black(i,j) && black(i,j+1))
            {
                int tmp = curr;
                if(noden[i][j][1]!=0) tmp = noden[i][j][3];
                else if(noden[i][j+1][3]!=0) tmp = noden[i][j+1][3];
                noden[i][j][1] = curr;
                noden[i][j+1][3] = curr;
                right = tmp;
                hori.insert(tmp);
                curr++;
            }

            //cout << up << " " << down << " " << left << " " << right << endl;
            if(up!=-1 && right!=-1) conn[up].pb(right);
            if(up!=-1 && left!=-1) conn[up].pb(left);
            if(down!=-1 && right!=-1) conn[down].pb(right);
            if(down!=-1 && left!=-1) conn[down].pb(left);
        }
    }

    src = curr++;
    int cnt2 = 0;
    for(int x:verti)
    {
        conn[src].pb(x);
        cnt2++;
    }

    sink = curr++;
    for(int x:hori)
    {
        conn[x].pb(sink);
        cnt2++;
    }

    //cout << src << endl;
    //cout << sink << endl;

    Dinic D(200000,src,sink);
    for(int i=1;i<=200000;i++)
    {
        for(auto x:conn[i])
        {
            //cout << i << " " << x << endl;
            D.add_edge(i,x,1);
        }
    }
    cout << cnt-(cnt2-D.flow()) << endl;
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
		//cout << "Case #" << i << ": ";  
		solve();
	}
 	return 0;
}