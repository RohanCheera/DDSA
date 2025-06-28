#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int MOD = 1e9 + 7;
const int INF = 1e9;
const ll LINF = 1e18;

// 1. Topological Sort (BFS - Kahn's Algorithm)
// Time: O(V + E), Space: O(V)
vector<int> topoSortBFS(int n, vector<vector<int>>& adj) {
    vector<int> inDegree(n, 0);
    for (int u = 0; u < n; u++)
        for (int v : adj[u])
            inDegree[v]++;
    queue<int> q;
    for (int i = 0; i < n; i++)
        if (inDegree[i] == 0) q.push(i);
    vector<int> topo;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        topo.push_back(u);
        for (int v : adj[u])
            if (--inDegree[v] == 0) q.push(v);
    }
    return topo.size() == n ? topo : vector<int>(); // Return empty if cycle exists
}

// 2. Topological Sort (DFS)
// Time: O(V + E), Space: O(V)
void topoSortDFSUtil(int u, vector<vector<int>>& adj, vector<bool>& vis, vector<int>& topo) {
    vis[u] = true;
    for (int v : adj[u])
        if (!vis[v])
            topoSortDFSUtil(v, adj, vis, topo);
    topo.push_back(u);
}
vector<int> topoSortDFS(int n, vector<vector<int>>& adj) {
    vector<bool> vis(n, false);
    vector<int> topo;
    for (int i = 0; i < n; i++)
        if (!vis[i])
            topoSortDFSUtil(i, adj, vis, topo);
    reverse(topo.begin(), topo.end());
    return topo;
}

// 3. Breadth-First Search (BFS)
// Time: O(V + E), Space: O(V)
vector<int> bfs(int n, vector<vector<int>>& adj, int start) {
    vector<int> dist(n, INF);
    queue<int> q;
    dist[start] = 0;
    q.push(start);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) {
            if (dist[v] == INF) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
    return dist;
}

// 4. Depth-First Search (DFS)
// Time: O(V + E), Space: O(V)
void dfsUtil(int u, vector<vector<int>>& adj, vector<bool>& vis) {
    vis[u] = true;
    for (int v : adj[u])
        if (!vis[v])
            dfsUtil(v, adj, vis);
}
void dfs(int n, vector<vector<int>>& adj, int start) {
    vector<bool> vis(n, false);
    dfsUtil(start, adj, vis);
}

// 5. Dijkstra’s Algorithm (Shortest Path, Non-negative Weights)
// Time: O((V + E) log V), Space: O(V)
vector<ll> dijkstra(int n, vector<vector<pair<int, ll>>>& adj, int start) {
    vector<ll> dist(n, LINF);
    priority_queue<pair<ll, int>, vector<pair<ll, int>>, greater<>> pq;
    dist[start] = 0;
    pq.push({0, start});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

// 6. Bellman-Ford Algorithm (Shortest Path, Negative Weights)
// Time: O(V * E), Space: O(V)
vector<ll> bellmanFord(int n, vector<vector<pair<int, ll>>>& adj, int start) {
    vector<ll> dist(n, LINF);
    dist[start] = 0;
    for (int i = 0; i < n - 1; i++) {
        for (int u = 0; u < n; u++) {
            for (auto [v, w] : adj[u]) {
                if (dist[u] != LINF && dist[v] > dist[u] + w) {
                    dist[v] = dist[u] + w;
                }
            }
        }
    }
    // Check for negative cycle
    for (int u = 0; u < n; u++) {
        for (auto [v, w] : adj[u]) {
            if (dist[u] != LINF && dist[v] > dist[u] + w) {
                return {}; // Negative cycle detected
            }
        }
    }
    return dist;
}

// 7. Floyd-Warshall Algorithm (All-Pairs Shortest Path)
// Time: O(V^3), Space: O(V^2)
vector<vector<ll>> floydWarshall(int n, vector<vector<pair<int, ll>>>& adj) {
    vector<vector<ll>> dist(n, vector<ll>(n, LINF));
    for (int i = 0; i < n; i++) dist[i][i] = 0;
    for (int u = 0; u < n; u++) {
        for (auto [v, w] : adj[u]) {
            dist[u][v] = w;
        }
    }
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != LINF && dist[k][j] != LINF) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
    return dist;
}

// 8. Kruskal’s Algorithm (Minimum Spanning Tree)
// Time: O(E log E), Space: O(V + E)
struct Edge {
    int u, v; ll w;
    bool operator<(const Edge& other) const { return w < other.w; }
};
struct UnionFind {
    vector<int> par, rank;
    UnionFind(int n) : par(n), rank(n, 0) {
        iota(par.begin(), par.end(), 0);
    }
    int find(int x) {
        if (par[x] != x) par[x] = find(par[x]);
        return par[x];
    }
    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;
        if (rank[px] < rank[py]) swap(px, py);
        par[py] = px;
        if (rank[px] == rank[py]) rank[px]++;
        return true;
    }
};
ll kruskal(int n, vector<Edge>& edges) {
    sort(edges.begin(), edges.end());
    UnionFind uf(n);
    ll mstWeight = 0;
    for (auto& e : edges) {
        if (uf.unite(e.u, e.v)) mstWeight += e.w;
    }
    return mstWeight;
}

// 9. Prim’s Algorithm (Minimum Spanning Tree)
// Time: O(E log V), Space: O(V)
ll prim(int n, vector<vector<pair<int, ll>>>& adj) {
    vector<ll> dist(n, LINF);
    vector<bool> vis(n, false);
    priority_queue<pair<ll, int>, vector<pair<ll, int>>, greater<>> pq;
    dist[0] = 0;
    pq.push({0, 0});
    ll mstWeight = 0;
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (vis[u]) continue;
        vis[u] = true;
        mstWeight += d;
        for (auto [v, w] : adj[u]) {
            if (!vis[v] && dist[v] > w) {
                dist[v] = w;
                pq.push({w, v});
            }
        }
    }
    return mstWeight;
}

// 10. Tarjan’s Algorithm (Strongly Connected Components)
// Time: O(V + E), Space: O(V)
void tarjanSCCUtil(int u, vector<vector<int>>& adj, vector<int>& disc, vector<int>& low, stack<int>& st, vector<bool>& inStack, vector<vector<int>>& scc) {
    static int time = 0;
    disc[u] = low[u] = ++time;
    st.push(u);
    inStack[u] = true;
    for (int v : adj[u]) {
        if (disc[v] == -1) {
            tarjanSCCUtil(v, adj, disc, low, st, inStack, scc);
            low[u] = min(low[u], low[v]);
        } else if (inStack[v]) {
            low[u] = min(low[u], disc[v]);
        }
    }
    if (disc[u] == low[u]) {
        vector<int> component;
        while (st.top() != u) {
            component.push_back(st.top());
            inStack[st.top()] = false;
            st.pop();
        }
        component.push_back(st.top());
        inStack[st.top()] = false;
        st.pop();
        scc.push_back(component);
    }
}
vector<vector<int>> tarjanSCC(int n, vector<vector<int>>& adj) {
    vector<int> disc(n, -1), low(n, -1);
    vector<bool> inStack(n, false);
    stack<int> st;
    vector<vector<int>> scc;
    for (int i = 0; i < n; i++)
        if (disc[i] == -1)
            tarjanSCCUtil(i, adj, disc, low, st, inStack, scc);
    return scc;
}

// 11. Binary Search
// Time: O(log n), Space: O(1)
int binarySearch(vector<int>& arr, int target) {
    int l = 0, r = arr.size() - 1;
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (arr[m] == target) return m;
        if (arr[m] < target) l = m + 1;
        else r = m - 1;
    }
    return -1;
}

// 12. Binary Search on Answer (Minimize Maximum)
// Time: O(log(range) * check), Space: O(1)
ll binarySearchAnswer(vector<int>& arr, ll target) {
    ll l = 0, r = 1e9, ans = -1;
    while (l <= r) {
        ll m = l + (r - l) / 2;
        if (check(arr, m, target)) { // Custom check function
            ans = m;
            r = m - 1;
        } else {
            l = m + 1;
        }
    }
    return ans;
}

// 13. Merge Sort with Inversion Count
// Time: O(n log n), Space: O(n)
ll merge(vector<int>& arr, int l, int m, int r) {
    vector<int> temp(r - l + 1);
    ll inv = 0;
    int i = l, j = m + 1, k = 0;
    while (i <= m && j <= r) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else {
            temp[k++] = arr[j++];
            inv += m - i + 1;
        }
    }
    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];
    for (i = l; i <= r; i++) arr[i] = temp[i - l];
    return inv;
}
ll mergeSort(vector<int>& arr, int l, int r) {
    ll inv = 0;
    if (l < r) {
        int m = l + (r - l) / 2;
        inv += mergeSort(arr, l, m);
        inv += mergeSort(arr, m + 1, r);
        inv += merge(arr, l, m, r);
    }
    return inv;
}

// 14. QuickSort
// Time: O(n log n) average, O(n^2) worst, Space: O(log n)
void quickSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int pivot = arr[r], i = l - 1;
        for (int j = l; j < r; j++) {
            if (arr[j] <= pivot) {
                i++;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[r]);
        int p = i + 1;
        quickSort(arr, l, p - 1);
        quickSort(arr, p + 1, r);
    }
}

// 15. KMP Algorithm (String Matching)
// Time: O(n + m), Space: O(m)
vector<int> computeLPS(string& pat) {
    int m = pat.size();
    vector<int> lps(m, 0);
    int len = 0, i = 1;
    while (i < m) {
        if (pat[i] == pat[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) len = lps[len - 1];
            else lps[i] = 0, i++;
        }
    }
    return lps;
}
vector<int> kmpSearch(string& txt, string& pat) {
    int n = txt.size(), m = pat.size();
    vector<int> lps = computeLPS(pat), matches;
    int i = 0, j = 0;
    while (i < n) {
        if (txt[i] == pat[j]) {
            i++; j++;
        }
        if (j == m) {
            matches.push_back(i - j);
            j = lps[j - 1];
        } else if (i < n && txt[i] != pat[j]) {
            if (j != 0) j = lps[j - 1];
            else i++;
        }
    }
    return matches;
}

// 16. Rabin-Karp Algorithm (String Matching)
// Time: O(n + m) average, O(nm) worst, Space: O(1)
vector<int> rabinKarp(string& txt, string& pat) {
    int n = txt.size(), m = pat.size(), d = 256, q = 101;
    ll h = 1;
    for (int i = 0; i < m - 1; i++) h = (h * d) % q;
    ll p = 0, t = 0;
    for (int i = 0; i < m; i++) {
        p = (d * p + pat[i]) % q;
        t = (d * t + txt[i]) % q;
    }
    vector<int> matches;
    for (int i = 0; i <= n - m; i++) {
        if (p == t) {
            bool match = true;
            for (int j = 0; j < m; j++) {
                if (txt[i + j] != pat[j]) {
                    match = false;
                    break;
                }
            }
            if (match) matches.push_back(i);
        }
        if (i < n - m) {
            t = (d * (t - txt[i] * h) + txt[i + m]) % q;
            if (t < 0) t += q;
        }
    }
    return matches;
}

// 17. Manacher’s Algorithm (Longest Palindromic Substring)
// Time: O(n), Space: O(n)
string manacher(string s) {
    string t = "#";
    for (char c : s) t += c, t += '#';
    int n = t.size();
    vector<int> p(n);
    int c = 0, r = 0;
    for (int i = 0; i < n; i++) {
        if (i <= r) p[i] = min(r - i, p[2 * c - i]);
        while (i - p[i] - 1 >= 0 && i + p[i] + 1 < n && t[i - p[i] - 1] == t[i + p[i] + 1]) p[i]++;
        if (i + p[i] > r) c = i, r = i + p[i];
    }
    int maxLen = *max_element(p.begin(), p.end());
    int center = find(p.begin(), p.end(), maxLen) - p.begin();
    int start = (center - maxLen) / 2;
    return s.substr(start, maxLen);
}

// 18. Prefix Sum (1D Array)
// Time: O(n) preprocess, O(1) query, Space: O(n)
vector<ll> prefixSum(vector<int>& arr) {
    int n = arr.size();
    vector<ll> ps(n + 1);
    for (int i = 0; i < n; i++) ps[i + 1] = ps[i] + arr[i];
    return ps; // Query: ps[r] - ps[l-1]
}

// 19. Prefix Sum (2D Array)
// Time: O(n*m) preprocess, O(1) query, Space: O(n*m)
vector<vector<ll>> prefixSum2D(vector<vector<int>>& grid) {
    int n = grid.size(), m = grid[0].size();
    vector<vector<ll>> ps(n + 1, vector<ll>(m + 1));
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            ps[i][j] = ps[i - 1][j] + ps[i][j - 1] - ps[i - 1][j - 1] + grid[i - 1][j - 1];
        }
    }
    return ps; // Query: ps[x2][y2] - ps[x1-1][y2] - ps[x2][y1-1] + ps[x1-1][y1-1]
}

// 20. Sliding Window (Fixed Length)
// Time: O(n), Space: O(1)
ll slidingWindowFixed(vector<int>& arr, int k) {
    ll sum = 0;
    for (int i = 0; i < k; i++) sum += arr[i];
    ll maxSum = sum;
    for (int i = k; i < arr.size(); i++) {
        sum += arr[i] - arr[i - k];
        maxSum = max(maxSum, sum);
    }
    return maxSum;
}

// 21. Sliding Window (Variable Length)
// Time: O(n), Space: O(1)
int slidingWindowVariable(vector<int>& arr, int target) {
    int l = 0, sum = 0, maxLen = 0;
    for (int r = 0; r < arr.size(); r++) {
        sum += arr[r];
        while (sum > target && l <= r) sum -= arr[l++];
        maxLen = max(maxLen, r - l + 1);
    }
    return maxLen;
}

// 22. Mo’s Algorithm (Offline Range Queries)
// Time: O(n*sqrt(n) + q*sqrt(n)), Space: O(n)
struct Query {
    int l, r, idx;
};
void mo(vector<int>& arr, vector<Query>& queries) {
    int block = sqrt(arr.size());
    sort(queries.begin(), queries.end(), [&](Query a, Query b) {
        if (a.l / block != b.l / block) return a.l < b.l;
        return a.r < b.r;
    });
    int currL = 0, currR = -1;
    for (auto& q : queries) {
        while (currL > q.l) add(--currL); // Custom add function
        while (currR < q.r) add(++currR);
        while (currL < q.l) remove(currL++); // Custom remove function
        while (currR > q.r) remove(currR--);
        // Store result for q.idx
    }
}

// 23. Segment Tree (Point Update, Range Query)
// Time: O(log n) update/query, Space: O(n)
struct SegmentTree {
    vector<ll> tree;
    int n;
    SegmentTree(vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);
        build(arr, 0, 0, n - 1);
    }
    void build(vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
            return;
        }
        int mid = (start + end) / 2;
        build(arr, 2 * node + 1, start, mid);
        build(arr, 2 * node + 2, mid + 1, end);
        tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
    }
    void update(int idx, int val, int node, int start, int end) {
        if (start == end) {
            tree[node] = val;
            return;
        }
        int mid = (start + end) / 2;
        if (idx <= mid) update(idx, val, 2 * node + 1, start, mid);
        else update(idx, val, 2 * node + 2, mid + 1, end);
        tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
    }
    ll query(int l, int r, int node, int start, int end) {
        if (r < start || l > end) return 0;
        if (l <= start && end <= r) return tree[node];
        int mid = (start + end) / 2;
        return query(l, r, 2 * node + 1, start, mid) + query(l, r, 2 * node + 2, mid + 1, end);
    }
};

// 24. Fenwick Tree (Binary Indexed Tree)
// Time: O(log n) update/query, Space: O(n)
struct FenwickTree {
    vector<ll> bit;
    int n;
    FenwickTree(int n) : n(n), bit(n + 1) {}
    void update(int idx, ll delta) {
        for (++idx; idx <= n; idx += idx & -idx) bit[idx] += delta;
    }
    ll query(int idx) {
        ll sum = 0;
        for (++idx; idx > 0; idx -= idx & -idx) sum += bit[idx];
        return sum;
    }
    ll rangeQuery(int l, int r) { return query(r) - query(l - 1); }
};

// 25. Trie (Prefix Tree)
// Time: O(m) insert/search, Space: O(total length)
struct TrieNode {
    TrieNode* children[26];
    bool isEnd;
    TrieNode() : isEnd(false) { fill(children, children + 26, nullptr); }
};
struct Trie {
    TrieNode* root;
    Trie() : root(new TrieNode()) {}
    void insert(string& word) {
        TrieNode* node = root;
        for (char c : word) {
            int i = c - 'a';
            if (!node->children[i]) node->children[i] = new TrieNode();
            node = node->children[i];
        }
        node->isEnd = true;
    }
    bool search(string& word) {
        TrieNode* node = root;
        for (char c : word) {
            int i = c - 'a';
            if (!node->children[i]) return false;
            node = node->children[i];
        }
        return node->isEnd;
    }
};

// 26. Lowest Common Ancestor (LCA) with Binary Lifting
// Time: O(n log n) preprocess, O(log n) query, Space: O(n log n)
struct LCA {
    vector<vector<int>> up;
    vector<int> depth;
    LCA(vector<vector<int>>& adj, int root) {
        int n = adj.size(), log = 20;
        up.assign(n, vector<int>(log));
        depth.assign(n, 0);
        dfs(root, -1, adj, 0);
        for (int j = 1; j < log; j++) {
            for (int i = 0; i < n; i++) {
                if (up[i][j - 1] != -1) up[i][j] = up[up[i][j - 1]][j - 1];
            }
        }
    }
    void dfs(int u, int p, vector<vector<int>>& adj, int d consecutively) {
        depth[u] = d;
        up[u][0] = p;
        for (int v : adj[u]) {
            if (v != p) dfs(v, u, adj, d + 1);
        }
    }
    int lca(int u, int v) {
        if (depth[u] < depth[v]) swap(u, v);
        for (int i = 19; i >= 0; i--) {
            if (depth[u] - (1 << i) >= depth[v]) u = up[u][i];
        }
        if (u == v) return u;
        for (int i = 19; i >= 0; i--) {
            if (up[u][i] != up[v][i]) {
                u = up[u][i];
                v = up[v][i];
            }
        }
        return up[u][0];
    }
};

// 27. Sieve of Eratosthenes
// Time: O(n log log n), Space: O(n)
vector<int> sieve(int n) {
    vector<bool> isPrime(n + 1, true);
    isPrime[0] = isPrime[1] = false;
    for (int i = 2; i * i <= n; i++) {
        if (isPrime[i]) {
            for (int j = i * i; j <= n; j += i) isPrime[j] = false;
        }
    }
    vector<int> primes;
    for (int i = 2; i <= n; i++) if (isPrime[i]) primes.push_back(i);
    return primes;
}

// 28. Fast Modular Exponentiation
// Time: O(log b), Space: O(1)
ll pow_mod(ll a, ll b, ll mod) {
    ll res = 1;
    while (b) {
        if (b & 1) res = res * a % mod;
        a = a * a % mod;
        b >>= 1;
    }
    return res;
}

// 29. Union-Find with Path Compression
// Time: O(α(n)) amortized, Space: O(n)
struct UnionFind {
    vector<int> par, rank;
    UnionFind(int n) : par(n), rank(n, 0) {
        iota(par.begin(), par.end(), 0);
    }
    int find(int x) {
        if (par[x] != x) par[x] = find(par[x]);
        return par[x];
    }
    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;
        if (rank[px] < rank[py]) swap(px, py);
        par[py] = px;
        if (rank[px] == rank[py]) rank[px]++;
        return true;
    }
};

// 30. Monotonic Stack (Next Greater Element)
// Time: O(n), Space: O(n)
vector<int> nextGreaterElement(vector<int>& arr) {
    int n = arr.size();
    vector<int> res(n, -1);
    stack<int> s;
    for (int i = 0; i < n; i++) {
        while (!s.empty() && arr[s.top()] < arr[i]) {
            res[s.top()] = arr[i];
            s.pop();
        }
        s.push(i);
    }
    return res;
}
