#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define fir first
#define sec second
typedef long long ll;

using namespace std;

struct point
{
    int x,y;
};

bool sortpoints(const point &lhs, const point &rhs) 
{ 
    return (lhs.x < rhs.x) || (lhs.x==rhs.x && lhs.y < rhs.y); 
}

vector<point> v;
vector<point> ans;

int cross(point a,point b){return a.x*b.y-a.y*b.x;}

int ccw(point p,point q,point r){
    point p1; p1.x = q.x-p.x; p1.y = q.y-p.y;
    point p2; p2.x = p.x-r.x; p2.y = p.y-r.y;
    int tmp = cross(p1,p2);
    if (tmp>0) return 1;
    else if(tmp==0) return 0;
    else return -1; 
}

vector<point> find_CH(vector<point>v)
{
    int n = v.size();
    vector<point> lh;
    vector<point> uh;
    sort(v.begin(),v.end(),sortpoints);
    lh.push_back(v[0]);
    lh.push_back(v[1]);
    for(int i=2;i<n;i++)
    {
        while(lh.size()>1)
        {
            if(ccw(lh[lh.size()-2],lh[lh.size()-1],v[i])!=-1) lh.pop_back();
            else break;
        }
        lh.push_back(v[i]);
    }
    
    uh.push_back(v[n-1]);
    uh.push_back(v[n-2]);
    for(int i=2;i<n;i++)
    {
        while(uh.size()>1)
        {
            if(ccw(uh[uh.size()-2],uh[uh.size()-1],v[n-i-1])!=-1) uh.pop_back();
            else break;
        }
        uh.push_back(v[n-i-1]);
    }

    for(int i=1;i<uh.size()-1;i++)
    {
        lh.push_back(uh[i]);
    }
    return lh;
}

void solve()
{
    while(true)
    {
        v.clear();
        int n;
        cin >> n;
        if(n==0) break;
        set<pair<int,int>> s;
        for (int i=0;i<n;i++)
        {
            point tmp;
            cin >> tmp.x >> tmp.y;
            if(s.find(make_pair(tmp.x,tmp.y))!=s.end()) continue;
            s.insert(make_pair(tmp.x,tmp.y));
            v.push_back(tmp);
        }
        if(v.size()==1)
        {
            cout << 1 << endl;
            cout << v[0].x << " " << v[0].y << endl;
            continue;
        }
        ans.clear();
        ans = find_CH(v);
        cout << ans.size() << endl;
        for(int i=0;i<ans.size();i++)
        {
            cout << ans[i].x << " " << ans[i].y << endl;
        }
    }
    return;
}

int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(0); cout.tie(0);
	int tc=1;
    solve();
 	return 0;
}
