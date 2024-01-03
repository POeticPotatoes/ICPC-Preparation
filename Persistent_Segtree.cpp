#include <bits/stdc++.h>
#define pb push_back
#define mp make_pair
#define fir first
#define sec second
typedef long long ll;

using namespace std;

struct Node
{
    int val;
    Node *left,*right;

    Node(Node* l, Node* r, int v) 
    { 
        left = l; 
        right = r; 
        val = v; 
    } 
};

int n,q;
int N = 1<<17;

pair<int,int> arr[100005]; //val,index
int v[100005] = {0};
int pos[100005] = {0};
Node* version[100005];

struct segtree
{
    void build(Node* node,int lx,int rx) //node start with root
    { 
        if (rx-lx==1) return; 
        int mid = (lx+rx)/2; 
        node->left = new Node(NULL, NULL, 0); 
        node->right = new Node(NULL, NULL, 0); 
        build(node->left,lx,mid); 
        build(node->right,mid,rx); 
        node->val = node->left->val + node->right->val;
        return;
    }

    void upd(int pos,Node* prev,Node* curr,int lx,int rx) //prev start with root
    {
        if(rx-lx==1)
        {
            curr->val++;
            return;
        }
        int mid = (lx+rx)/2;
        if(pos<mid)
        {
            curr->right = prev->right;
            curr->left = new Node(NULL, NULL, 0);
            upd(pos,prev->left,curr->left,lx,mid);
        }
        else
        {
            curr->left = prev->left; 
            curr->right = new Node(NULL, NULL, 0); 
            upd(pos,prev->right,curr->right,mid,rx);
        }
        curr->val = curr->left->val + curr->right->val;
        return;
    }

    ll query(int k,Node* node1,Node* node2,int lx,int rx)
    {
        if(rx-lx==1) return lx;
        int mid = (lx+rx)/2;
        int leftval = node2->left->val - node1->left->val;
        if(leftval>=k) return query(k,node1->left,node2->left,lx,mid);
        else query(k-leftval,node1->right,node2->right,mid,rx);
    }
};

void solve()
{
    cin >> n >> q;
    for(int i=1;i<=n;i++)
    {
        cin >> arr[i].fir;
        arr[i].sec = i;
    }
    sort(arr+1,arr+n+1);
    for(int i=1;i<=n;i++)
    {
        pos[arr[i].sec]=i;
    }
    
    Node *root = new Node(NULL, NULL, 0);
    segtree st;
    st.build(root,0,N);
    version[0] = root;

    for(int i=1;i<=n;i++)
    {
        version[i] = new Node(NULL, NULL, 0);
        st.upd(pos[i],version[i-1],version[i],0,N);
    }

    for(int i=0;i<q;i++)
    {
        int l,r,k;
        cin >> l >> r >> k;
        ll ind = st.query(k,version[l-1],version[r],0,N);
        cout << arr[ind].first << endl;
    }
    return;
}

int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(0); cout.tie(0);
    solve();
 	return 0;
}