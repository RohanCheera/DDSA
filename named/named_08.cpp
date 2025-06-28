#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int MOD = 1e9 + 7;
const int INF = 1e9;
const ll LINF = 1e18;

// 181. Factorial of Large Number
// Time: O(n log n), Space: O(log n)
// Purpose: Compute factorial of large number with modulo
vector<int> factorialLarge(int n) {
    vector<int> res = {1};
    auto multiply = [](vector<int>& num, int x) {
        int carry = 0;
        for (int i = 0; i < num.size(); i++) {
            int prod = num[i] * x + carry;
            num[i] = prod % 10;
            carry = prod / 10;
        }
        while (carry) { num.push_back(carry % 10); carry /= 10; }
    };
    for (int i = 2; i <= n; i++) multiply(res, i);
    reverse(res.begin(), res.end());
    return res;
}

// 182. Boyer-Moore Majority Voting
// Time: O(n), Space: O(1)
// Purpose: Find majority element (> n/2 occurrences)
int majorityElement(vector<int>& nums) {
    int count = 0, candidate = 0;
    for (int x : nums) {
        if (count == 0) candidate = x;
        count += (x == candidate) ? 1 : -1;
    }
    return candidate; // Assumes majority exists
}

// 183. Sparse Table (Range Max Query)
// Time: O(n log n) preprocess, O(1) query, Space: O(n log n)
// Purpose: Efficient range maximum queries (variation from 201: sum, 202: min)
struct SparseTableMax {
    vector<vector<int>> st;
    SparseTableMax(vector<int>& arr) {
        int n = arr.size(), log = __lg(n) + 1;
        st.assign(n, vector<int>(log));
        for (int i = 0; i < n; i++) st[i][0] = arr[i];
        for (int j = 1; j < log; j++) {
            for (int i = 0; i + (1 << j) - 1 < n; i++) {
                st[i][j] = max(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);
            }
        }
    }
    int query(int l, int r) {
        int j = __lg(r - l + 1);
        return max(st[l][j], st[r - (1 << j) + 1][j]);
    }
};

// 184. Disjoint Set Union (Path Compression)
// Time: O(α(n)) amortized, Space: O(n)
// Purpose: Optimized union-find for dynamic connectivity
struct DSU {
    vector<int> par, rank;
    DSU(int n) : par(n), rank(n, 0) { iota(par.begin(), par.end(), 0); }
    int find(int x) { return par[x] == x ? x : par[x] = find(par[x]); }
    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;
        if (rank[px] < rank[py]) swap(px, py);
        par[py] = px;
        if (rank[px] == rank[py]) rank[px]++;
        return true;
    }
};

// 185. Rolling Hash (Multiple Mods)
// Time: O(n + m), Space: O(1)
// Purpose: String hashing with multiple primes for substring comparison
struct RollingHash {
    vector<ll> h1, h2, p1, p2;
    const ll M1 = 1e9 + 9, M2 = 1e9 + 7, P = 31;
    RollingHash(string& s) {
        int n = s.size();
        h1.assign(n + 1, 0); h2.assign(n + 1, 0);
        p1.assign(n + 1, 1); p2.assign(n + 1, 1);
        for (int i = 0; i < n; i++) {
            h1[i + 1] = (h1[i] * P + s[i]) % M1;
            h2[i + 1] = (h2[i] * P + s[i]) % M2;
            p1[i + 1] = p1[i] * P % M1;
            p2[i + 1] = p2[i] * P % M2;
        }
    }
    pair<ll, ll> getHash(int l, int r) {
        ll hash1 = (h1[r + 1] - h1[l] * p1[r - l + 1] % M1 + M1) % M1;
        ll hash2 = (h2[r + 1] - h2[l] * p2[r - l + 1] % M2 + M2) % M2;
        return {hash1, hash2};
    }
};

// 186. Fast Fourier Transform
// Time: O(n log n), Space: O(n)
// Purpose: Polynomial multiplication (same as 215, included for completeness)
using cd = complex<double>;
void fft(vector<cd>& a, bool invert) {
    int n = a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) for (auto& x : a) x /= n;
}
vector<ll> multiplyPolynomials(vector<ll>& a, vector<ll>& b) {
    int n = 1;
    while (n < a.size() + b.size()) n <<= 1;
    vector<cd> fa(a.begin(), a.end()), fb(b.begin(), b.end());
    fa.resize(n); fb.resize(n);
    fft(fa, false); fft(fb, false);
    for (int i = 0; i < n; i++) fa[i] *= fb[i];
    fft(fa, true);
    vector<ll> res(n);
    for (int i = 0; i < n; i++) res[i] = round(fa[i].real());
    return res;
}

// 187. Mo’s Algorithm with Updates
// Time: O(n*sqrt(n) + q*sqrt(n)), Space: O(n)
// Purpose: Handle range queries with updates (same as 216, included for completeness)
struct Query { int l, r, t, idx; };
struct Update { int idx, oldVal, newVal; };
void moWithUpdates(vector<int>& arr, vector<Query>& queries, vector<Update>& updates) {
    int n = arr.size(), block = pow(n, 2.0 / 3);
    sort(queries.begin(), queries.end(), [&](Query a, Query b) {
        if (a.l / block != b.l / block) return a.l < b.l;
        if (a.r / block != b.r / block) return a.r < b.r;
        return a.t < b.t;
    });
    int currL = 0, currR = -1, currT = 0;
    for (auto& q : queries) {
        while (currT < q.t) {
            auto& u = updates[currT++];
            if (u.idx >= currL && u.idx <= currR) { remove(arr[u.idx]); add(u.newVal); }
            arr[u.idx] = u.newVal;
        }
        while (currT > q.t) {
            auto& u = updates[--currT];
            if (u.idx >= currL && u.idx <= currR) { remove(arr[u.idx]); add(u.oldVal); }
            arr[u.idx] = u.oldVal;
        }
        while (currL > q.l) add(arr[--currL]);
        while (currR < q.r) add(arr[++currR]);
        while (currL < q.l) remove(arr[currL++]);
        while (currR > q.r) remove(arr[currR--]);
        // Store result for q.idx
    }
};
void add(int val) {} // Placeholder
void remove(int val) {} // Placeholder

// 188. Persistent Segment Tree (Range Sum)
// Time: O(log n) per update/query, Space: O(n log n)
// Purpose: Maintain history of range sum updates (variation from 203: point update)
struct PersistentSegmentTreeSum {
    struct Node { ll val; Node *left, *right; Node(ll v = 0) : val(v), left(nullptr), right(nullptr) {} };
    vector<Node*> roots;
    int n;
    PersistentSegmentTreeSum(int n) : n(n) { roots.push_back(new Node()); }
    Node* update(Node* node, int start, int end, int l, int r, ll val) {
        Node* newNode = new Node(*node);
        newNode->val += val * (min(end, r) - max(start, l) + 1);
        if (start == l && end == r) return newNode;
        int mid = (start + end) / 2;
        if (r <= mid) newNode->left = update(node->left ? node->left : new Node(), start, mid, l, r, val);
        else if (l > mid) newNode->right = update(node->right ? node->right : new Node(), mid + 1, end, l, r, val);
        else {
            newNode->left = update(node->left ? node->left : new Node(), start, mid, l, mid, val);
            newNode->right = update(node->right ? node->right : new Node(), mid + 1, end, mid + 1, r, val);
        }
        return newNode;
    }
    ll query(Node* node, int start, int end, int l, int r) {
        if (!node || r < start || l > end) return 0;
        if (l <= start && end <= r) return node->val;
        int mid = (start + end) / 2;
        return query(node->left, start, mid, l, r) + query(node->right, mid + 1, end, l, r);
    }
    void update(int l, int r, ll val) { roots.push_back(update(roots.back(), 0, n - 1, l, r, val)); }
    ll query(int version, int l, int r) { return query(roots[version], 0, n - 1, l, r); }
};

// 189. Treap (With Split/Merge)
// Time: O(log n) expected, Space: O(n)
// Purpose: Randomized BST with split/merge (variation from 204: basic insert)
struct TreapSplitMerge {
    struct Node { int key, priority, size; Node *left, *right; Node(int k) : key(k), priority(rand()), size(1), left(nullptr), right(nullptr) {} };
    Node* root = nullptr;
    int getSize(Node* node) { return node ? node->size : 0; }
    void updateSize(Node* node) { if (node) node->size = getSize(node->left) + getSize(node->right) + 1; }
    Node* merge(Node* l, Node* r) {
        if (!l || !r) return l ? l : r;
        if (l->priority > r->priority) { l->right = merge(l->right, r); updateSize(l); return l; }
        r->left = merge(l, r->left); updateSize(r); return r;
    }
    pair<Node*, Node*> split(Node* node, int key) {
        if (!node) return {nullptr, nullptr};
        if (node->key <= key) {
            auto [l, r] = split(node->right, key);
            node->right = l; updateSize(node);
            return {node, r};
        }
        auto [l, r] = split(node->left, key);
        node->left = r; updateSize(node);
        return {l, node};
    }
    void insert(int key) {
        auto [l, r] = split(root, key);
        root = merge(merge(l, new Node(key)), r);
    }
};

// 190. Splay Tree (With Search)
// Time: O(log n) amortized, Space: O(n)
// Purpose: Self-adjusting BST with search focus (variation from 217)
struct SplayTreeSearch {
    struct Node { int key, size; Node *left, *right, *parent; Node(int k) : key(k), size(1), left(nullptr), right(nullptr), parent(nullptr) {} };
    Node* root = nullptr;
    void update(Node* node) { node->size = 1 + (node->left ? node->left->size : 0) + (node->right ? node->right->size : 0); }
    void rotate(Node* x) {
        Node *p = x->parent, *g = p->parent;
        if (x == p->left) {
            p->left = x->right; if (x->right) x->right->parent = p;
            x->right = p; p->parent = x;
        } else {
            p->right = x->left; if (x->left) x->left->parent = p;
            x->left = p; p->parent = x;
        }
        x->parent = g;
        if (g) (p == g->left ? g->left : g->right) = x;
        update(p); update(x);
    }
    void splay(Node* x) {
        while (x->parent) {
            Node *p = x->parent, *g = p->parent;
            if (g) rotate((x == p->left) == (p == g->left) ? p : x);
            rotate(x);
        }
        root = x;
    }
    Node* search(int key) {
        Node* node = root;
        while (node && node->key != key) node = key < node->key ? node->left : node->right;
        if (node) splay(node);
        return node;
    }
};

// 191. Aho-Corasick (With Count)
// Time: O(n + m + z), Space: O(n)
// Purpose: Multi-pattern matching with occurrence count (variation from 205)
struct AhoCorasickCount {
    struct Node { Node* children[26], *fail; int count; Node() : fail(nullptr), count(0) { fill(children, children + 26, nullptr); } };
    Node* root = new Node();
    void insert(string& s, int idx) {
        Node* node = root;
        for (char c : s) {
            int i = c - 'a';
            if (!node->children[i]) node->children[i] = new Node();
            node = node->children[i];
        }
        node->count++;
    }
    void build() {
        queue<Node*> q;
        root->fail = root;
        for (int i = 0; i < 26; i++) {
            if (root->children[i]) {
                root->children[i]->fail = root;
                q.push(root->children[i]);
            } else root->children[i] = root;
        }
        while (!q.empty()) {
            Node* node = q.front(); q.pop();
            for (int i = 0; i < 26; i++) {
                if (node->children[i]) {
                    node->children[i]->fail = node->fail->children[i];
                    node->children[i]->count += node->fail->children[i]->count;
                    q.push(node->children[i]);
                } else node->children[i] = node->fail->children[i];
            }
        }
    }
    int search(string& text) {
        Node* node = root;
        int total = 0;
        for (char c : text) {
            int i = c - 'a';
            node = node->children[i];
            total += node->count;
        }
        return total;
    }
};

// 192. Convex Hull (Andrew’s Monotone Chain)
// Time: O(n log n), Space: O(n)
// Purpose: Find convex hull using monotone chain (variation from 206: Graham Scan)
struct Point { ll x, y; };
vector<Point> convexHullAndrew(vector<Point>& points) {
    int n = points.size();
    sort(points.begin(), points.end(), [](Point& a, Point& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
    vector<Point> hull;
    for (int i = 0; i < n; i++) {
        while (hull.size() >= 2 && (hull[hull.size() - 2].x - hull.back().x) * (points[i].y - hull.back().y) <= 
               (hull[hull.size() - 2].y - hull.back().y) * (points[i].x - hull.back().x)) hull.pop_back();
        hull.push_back(points[i]);
    }
    int lower = hull.size();
    for (int i = n - 2; i >= 0; i--) {
        while (hull.size() > lower && (hull[hull.size() - 2].x - hull.back().x) * (points[i].y - hull.back().y) <= 
               (hull[hull.size() - 2].y - hull.back().y) * (points[i].x - hull.back().x)) hull.pop_back();
        hull.push_back(points[i]);
    }
    if (hull.size() > 1) hull.pop_back();
    return hull;
}

// 193. Line Sweep (With Intersections)
// Time: O(n log n), Space: O(n)
// Purpose: Count segment intersections (variation from 207)
struct Event { ll x; int type, idx; bool operator<(const Event& other) const { return x < other.x || (x == other.x && type < other.type); } };
int lineSweepIntersections(vector<pair<ll, ll>>& segments) {
    vector<Event> events;
    for (int i = 0; i < segments.size(); i++) {
        events.push_back({segments[i].first, 0, i});
        events.push_back({segments[i].second, 1, i});
    }
    sort(events.begin(), events.end());
    set<int> active;
    int intersections = 0;
    for (auto& e : events) {
        if (e.type == 0) {
            intersections += active.size(); // Count intersections with active segments
            active.insert(e.idx);
        } else active.erase(e.idx);
    }
    return intersections;
}

// 194. Closest Pair of Points (Divide and Conquer)
// Time: O(n log n), Space: O(n)
// Purpose: Find closest pair in 2D (same as 208, included for completeness)
double closestPairDC(vector<Point>& points) {
    sort(points.begin(), points.end(), [](Point& a, Point& b) { return a.x < b.x; });
    auto dist = [](Point& a, Point& b) {
        return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
    };
    double minDist = LINF;
    set<Point, decltype([](Point& a, Point& b) { return a.y < b.y; })> s;
    int j = 0;
    for (int i = 0; i < points.size(); i++) {
        while (j < i && points[i].x - points[j].x > minDist) s.erase(points[j++]);
        auto it = s.lower_bound({0, points[i].y - minDist});
        while (it != s.end() && it->y <= points[i].y + minDist) {
            minDist = min(minDist, dist(points[i], *it));
            it++;
        }
        s.insert(points[i]);
    }
    return minDist;
}

// 195. K-D Tree (Range Search)
// Time: O(n log n) build, O(sqrt(n)) range query expected, Space: O(n)
// Purpose: Range search in k-dimensional space (variation from 209: nearest neighbor)
struct KDNodeRange {
    Point p; KDNodeRange *left, *right;
    KDNodeRange(Point _p) : p(_p), left(nullptr), right(nullptr) {}
};
struct KDTreeRange {
    KDNodeRange* root = nullptr;
    void insert(Point p, KDNodeRange*& node, int depth) {
        if (!node) { node = new KDNodeRange(p); return; }
        int cd = depth % 2;
        if (cd == 0 ? p.x < node->p.x : p.y < node->p.y) insert(p, node->left, depth + 1);
        else insert(p, node->right, depth + 1);
    }
    void rangeSearch(KDNodeRange* node, int depth, ll x1, ll y1, ll x2, ll y2, vector<Point>& res) {
        if (!node) return;
        if (x1 <= node->p.x && node->p.x <= x2 && y1 <= node->p.y && node->p.y <= y2) res.push_back(node->p);
        int cd = depth % 2;
        if (cd == 0) {
            if (x1 <= node->p.x) rangeSearch(node->left, depth + 1, x1, y1, x2, y2, res);
            if (x2 >= node->p.x) rangeSearch(node->right, depth + 1, x1, y1, x2, y2, res);
        } else {
            if (y1 <= node->p.y) rangeSearch(node->left, depth + 1, x1, y1, x2, y2, res);
            if (y2 >= node->p.y) rangeSearch(node->right, depth + 1, x1, y1, x2, y2, res);
        }
    }
    void insert(Point p) { insert(p, root, 0); }
    vector<Point> rangeSearch(ll x1, ll y1, ll x2, ll y2) {
        vector<Point> res;
        rangeSearch(root, 0, x1, y1, x2, y2, res);
        return res;
    }
};

// 196. Heavy Path Decomposition (With Segment Tree)
// Time: O(n log n) preprocess, O(log n) query, Space: O(n log n)
// Purpose: Path sum queries on tree (variation from 210)
struct HLDWithSegTree {
    vector<int> par, heavy, head, pos;
    vector<vector<int>> adj;
    vector<ll> segTree;
    int n, curPos = 0;
    HLDWithSegTree(vector<vector<int>>& _adj, int _n) : adj(_adj), n(_n), par(n, -1), heavy(n, -1), head(n), pos(n), segTree(4 * n) {
        dfsSize(0); dfsHLD(0);
    }
    int dfsSize(int u) {
        int size = 1, maxChildSize = 0;
        for (int v : adj[u]) {
            if (v != par[u]) {
                par[v] = u;
                int childSize = dfsSize(v);
                if (childSize > maxChildSize) { heavy[u] = v; maxChildSize = childSize; }
                size += childSize;
            }
        }
        return size;
    }
    void dfsHLD(int u) {
        pos[u] = curPos++;
        head[u] = heavy[par[u]] == u ? head[par[u]] : u;
        if (heavy[u] != -1) dfsHLD(heavy[u]);
        for (int v : adj[u]) if (v != par[u] && v != heavy[u]) dfsHLD(v);
    }
    void updateSegTree(int node, int start, int end, int idx, ll val) {
        if (start == end) { segTree[node] = val; return; }
        int mid = (start + end) / 2;
        if (idx <= mid) updateSegTree(2 * node + 1, start, mid, idx, val);
        else updateSegTree(2 * node + 2, mid + 1, end, idx, val);
        segTree[node] = segTree[2 * node + 1] + segTree[2 * node + 2];
    }
    ll querySegTree(int node, int start, int end, int l, int r) {
        if (r < start || l > end) return 0;
        if (l <= start && end <= r) return segTree[node];
        int mid = (start + end) / 2;
        return querySegTree(2 * node + 1, start, mid, l, r) + querySegTree(2 * node + 2, mid + 1, end, l, r);
    }
    ll queryPath(int u, int v) {
        ll res = 0;
        while (head[u] != head[v]) {
            if (pos[head[u]] > pos[head[v]]) swap(u, v);
            res += querySegTree(0, 0, n - 1, pos[head[v]], pos[v]);
            v = par[head[v]];
        }
        if (pos[u] > pos[v]) swap(u, v);
        res += querySegTree(0, 0, n - 1, pos[u], pos[v]);
        return res;
    }
};

// 197. Centroid Decomposition (With Distance Queries)
// Time: O(n log n), Space: O(n)
// Purpose: Handle distance queries in tree (variation from 211)
struct CentroidDecompositionDist {
    vector<vector<int>> adj;
    vector<int> size, par, removed;
    vector<ll> dist;
    CentroidDecompositionDist(vector<vector<int>>& _adj) : adj(_adj), size(_adj.size()), par(_adj.size(), -1), removed(_adj.size(), 0), dist(_adj.size()) {}
    int getSize(int u, int p) {
        size[u] = 1;
        for (int v : adj[u]) if (v != p && !removed[v]) size[u] += getSize(v, u);
        return size[u];
    }
    int findCentroid(int u, int p, int total) {
        for (int v : adj[u]) {
            if (v != p && !removed[v] && size[v] > total / 2) return findCentroid(v, u, total);
        }
        return u;
    }
    void computeDist(int u, int p, ll d, int centroid) {
        dist[u] = d;
        for (int v : adj[u]) if (v != p && !removed[v]) computeDist(v, u, d + 1, centroid);
    }
    void decompose(int u, int p) {
        int total = getSize(u, -1);
        u = findCentroid(u, -1, total);
        computeDist(u, -1, 0, u);
        par[u] = p;
        removed[u] = 1;
        for (int v : adj[u]) if (!removed[v]) decompose(v, u);
    }
    ll queryDist(int u, int v) {
        // Use dist array to compute distance via LCA or centroid paths
        return dist[u] + dist[v]; // Simplified for example
    }
};

// 198. Bitonic Sort
// Time: O(log^2 n), Space: O(1)
// Purpose: Sort bitonic sequence (ascending then descending)
void bitonicSort(vector<int>& arr, int l, int r, bool asc) {
    if (r <= l) return;
    int k = r - l + 1;
    for (int i = k / 2; i >= 1; i /= 2) {
        for (int j = l; j <= r - i; j++) {
            if ((arr[j] > arr[j + i]) == asc) swap(arr[j], arr[j + i]);
        }
    }
}
void sortBitonic(vector<int>& arr) {
    int n = arr.size();
    for (int k = 2; k <= n; k *= 2) {
        for (int j = 0; j < n; j += k) {
            bitonicSort(arr, j, j + k - 1, j % (2 * k) == 0);
        }
    }
}

// 199. Cyclic Shift Search
// Time: O(log n), Space: O(1)
// Purpose: Search in cyclically shifted sorted array
int cyclicShiftSearch(vector<int>& nums, int target) {
    int l = 0, r = nums.size() - 1;
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (nums[m] == target) return m;
        if (nums[l] <= nums[m]) {
            if (nums[l] <= target && target < nums[m]) r = m - 1;
            else l = m + 1;
        } else {
            if (nums[m] < target && target <= nums[r]) l = m + 1;
            else r = m - 1;
        }
    }
    return -1;
}

// 200. Modular Inverse
// Time: O(log n), Space: O(1)
// Purpose: Compute modular inverse using extended GCD
ll modInverse(ll a, ll m) {
    ll m0 = m, t, q;
    ll x0 = 0, x1 = 1;
    if (m == 1) return 0;
    while (a > 1) {
        q = a / m;
        t = m; m = a % m; a = t;
        t = x0; x0 = x1 - q * x0; x1 = t;
    }
    if (x1 < 0) x1 += m0;
    return x1;
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr); cout.tie(nullptr);
    // Example usage of templates
    return 0;
}
