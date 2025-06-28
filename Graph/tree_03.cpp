class Tree {
public:
    int V;
    vector<vector<int>> adj;

    Tree(int vertices) : V(vertices) {
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // 51. Tree Diameter (DFS)
    pair<int,int> dfsDiameter(int u, int parent, vector<int>& dist) {
        pair<int,int> maxDist = {0, u};
        for (int v : adj[u]) {
            if (v != parent) {
                auto [d, node] = dfsDiameter(v, u, dist);
                if (d + 1 > maxDist.first) {
                    maxDist = {d + 1, node};
                }
                dist[u] = max(dist[u], dist[v] + 1);
            }
        }
        return maxDist;
    }

    int treeDiameterDFS() {
        vector<int> dist(V, 0);
        auto [d1, n1] = dfsDiameter(0, -1, dist);
        auto [d2, n2] = dfsDiameter(n1, -1, dist);
        return d2;
    }

    // 52. Tree Diameter (BFS)
    int treeDiameterBFS() {
        queue<int> q;
        vector<int> dist(V, -1);
        q.push(0);
        dist[0] = 0;
        int last = 0;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            last = u;
            for (int v : adj[u]) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }
        fill(dist.begin(), dist.end(), -1);
        q.push(last);
        dist[last] = 0;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            last = u;
            for (int v : adj[u]) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }
        return dist[last];
    }

    // 53. Tree Center
    vector<int> treeCenter() {
        vector<int> degree(V, 0), leaves;
        for (int i = 0; i < V; i++) {
            degree[i] = adj[i].size();
            if (degree[i] <= 1) leaves.push_back(i);
        }
        int remaining = V;
        while (remaining > 2) {
            vector<int> new_leaves;
            for (int u : leaves) {
                remaining--;
                for (int v : adj[u]) {
                    if (--degree[v] == 1) {
                        new_leaves.push_back(v);
                    }
                }
            }
            leaves = new_leaves;
        }
        return leaves;
    }

    // 54. Binary Lifting (Ancestor)
    vector<vector<int>> binaryLifting(int root) {
        int logV = ceil(log2(V));
        vector<vector<int>> up(V, vector<int>(logV + 1, -1));
        vector<int> depth(V, 0);
        function<void(int, int)> dfs = [&](int u, int p) {
            up[u][0] = p;
            for (int i = 1; i <= logV; i++) {
                if (up[u][i-1] != -1) {
                    up[u][i] = up[up[u][i-1]][i-1];
                }
            }
            for (int v : adj[u]) {
                if (v != p) {
                    depth[v] = depth[u] + 1;
                    dfs(v, u);
                }
            }
        };
        dfs(root, -1);
        return up;
    }

    int getKthAncestor(int node, int k, vector<vector<int>>& up) {
        int logV = up[0].size() - 1;
        for (int i = logV; i >= 0; i--) {
            if (k & (1 << i)) {
                node = up[node][i];
                if (node == -1) break;
            }
        }
        return node;
    }

    // 55. Tree Path Sum
    int treePathSum(int u, int parent, vector<int>& values) {
        int sum = values[u];
        for (int v : adj[u]) {
            if (v != parent) {
                sum += treePathSum(v, u, values);
            }
        }
        return sum;
    }

    // 56. Subtree Size
    vector<int> subtreeSize(int u, int parent) {
        vector<int> sizes(V, 1);
        for (int v : adj[u]) {
            if (v != parent) {
                auto sub = subtreeSize(v, u);
                for (int i = 0; i < V; i++) {
                    sizes[i] += sub[i];
                }
            }
        }
        return sizes;
    }

    // 57. Heavy-Light Decomposition
    vector<int> hld(int root) {
        vector<int> parent(V, -1), size(V, 1), heavy(V, -1), head(V), pos(V);
        int cur_pos = 0;
        function<void(int, int)> dfs = [&](int u, int p) {
            parent[u] = p;
            for (int v : adj[u]) {
                if (v != p) {
                    dfs(v, u);
                    size[u] += size[v];
                    if (heavy[u] == -1 || size[v] > size[heavy[u]]) {
                        heavy[u] = v;
                    }
                }
            }
        };
        function<void(int, int, int)> hld_dfs = [&](int u, int h, int p) {
            head[u] = h;
            pos[u] = cur_pos++;
            if (heavy[u] != -1) {
                hld_dfs(heavy[u], h, u);
            }
            for (int v : adj[u]) {
                if (v != p && v != heavy[u]) {
                    hld_dfs(v, v, u);
                }
            }
        };
        dfs(root, -1);
        hld_dfs(root, root, -1);
        return pos;
    }

    // 58. Lowest Common Ancestor (Euler Tour)
    pair<vector<int>, vector<int>> eulerTour(int root) {
        vector<int> tour, first(V, -1);
        int idx = 0;
        function<void(int, int)> dfs = [&](int u, int p) {
            first[u] = idx;
            tour.push_back(u);
            idx++;
            for (int v : adj[u]) {
                if (v != p) {
                    dfs(v, u);
                    tour.push_back(u);
                    idx++;
                }
            }
        };
        dfs(root, -1);
        return {tour, first};
    }

    // 59. Tree Flattening (Euler Tour)
    vector<int> flattenTree(int root) {
        auto [tour, _] = eulerTour(root);
        return tour;
    }

    // 60. Centroid Decomposition
    void centroidDecomposition(int u, int parent, vector<bool>& processed, 
                             vector<int>& size, vector<vector<int>>& centroids) {
        size[u] = 1;
        bool is_centroid = true;
        for (int v : adj[u]) {
            if (v != parent && !processed[v]) {
                centroidDecomposition(v, u, processed, size, centroids);
                size[u] += size[v];
                if (size[v] > V / 2) is_centroid = false;
            }
        }
        if (is_centroid && V - size[u] <= V / 2) {
            processed[u] = true;
            centroids.push_back({u});
            for (int v : adj[u]) {
                if (!processed[v]) {
                    centroidDecomposition(v, u, processed, size, centroids);
                }
            }
        }
    }

    vector<vector<int>> getCentroids() {
        vector<bool> processed(V, false);
        vector<int> size(V, 0);
        vector<vector<int>> centroids;
        centroidDecomposition(0, -1, processed, size, centroids);
        return centroids;
    }

    // 61. Tree Isomorphism
    bool areIsomorphic(Tree& other, int root1, int root2) {
        vector<vector<int>> t1 = adj, t2 = other.adj;
        function<string(int, int, vector<vector<int>>&)> dfs = 
            [&](int u, int p, vector<vector<int>>& adj) {
                vector<string> children;
                for (int v : adj[u]) {
                    if (v != p) {
                        children.push_back(dfs(v, u, adj));
                    }
                }
                sort(children.begin(), children.end());
                string result = "(";
                for (string& s : children) result += s;
                result += ")";
                return result;
            };
        return dfs(root1, -1, t1) == dfs(root2, -1, t2);
    }

    // 62. Min Vertex Cover in Tree
    pair<int, vector<int>> minVertexCover(int u, int parent, bool taken) {
        int include = 1, exclude = 0;
        vector<int> inc_nodes = {u}, exc_nodes;
        for (int v : adj[u]) {
            if (v != parent) {
                auto [inc, inc_v] = minVertexCover(v, u, true);
                auto [exc, exc_v] = minVertexCover(v, u, false);
                if (taken) {
                    include += min(inc, exc);
                    inc_nodes.insert(inc_nodes.end(), 
                                   (inc < exc ? inc_v : exc_v).begin(), 
                                   (inc < exc ? inc_v : exc_v).end());
                } else {
                    include += inc;
                    inc_nodes.insert(inc_nodes.end(), inc_v.begin(), inc_v.end());
                    exclude += min(inc, exc);
                    exc_nodes.insert(exc_nodes.end(), 
                                   (inc < exc ? inc_v : exc_v).begin(), 
                                   (inc < exc ? inc_v : exc_v).end());
                }
            }
        }
        return taken ? make_pair(include, inc_nodes) : 
                       make_pair(exclude, exc_nodes);
    }

    // 63. Max Independent Set in Tree
    pair<int, vector<int>> maxIndependentSet(int u, int parent, bool taken) {
        int include = 1, exclude = 0;
        vector<int> inc_nodes = {u}, exc_nodes;
        for (int v : adj[u]) {
            if (v != parent) {
                auto [inc, inc_v] = maxIndependentSet(v, u, false);
                auto [exc, exc_v] = maxIndependentSet(v, u, true);
                if (taken) {
                    exclude += max(inc, exc);
                    exc_nodes.insert(exc_nodes.end(), 
                                   (inc > exc ? inc_v : exc_v).begin(), 
                                   (inc > exc ? inc_v : exc_v).end());
                } else {
                    include += max(inc, exc);
                    inc_nodes.insert(inc_nodes.end(), 
                                   (inc > exc ? inc_v : exc_v).begin(), 
                                   (inc > exc ? inc_v : exc_v).end());
                    exclude += exc;
                    exc_nodes.insert(exc_nodes.end(), exc_v.begin(), exc_v.end());
                }
            }
        }
        return taken ? make_pair(exclude, exc_nodes) : 
                       make_pair(include, inc_nodes);
    }

    // 64. Tree Path Queries
    void pathQuery(int u, int v, vector<int>& pos, vector<int>& values, 
                  vector<int>& segTree) {
        // Use HLD positions and segment tree for path queries
        // Implementation omitted for brevity
    }

    // 65. Subtree Queries
    void subtreeQuery(int u, vector<int>& pos, vector<int>& values, 
                    vector<int>& segTree) {
        // Use segment tree for subtree queries
        // Implementation omitted for brevity
    }

    // 66. Tree Pruning
    vector<int> pruneLeaves() {
        vector<int> degree(V, 0), leaves;
        for (int i = 0; i < V; i++) {
            degree[i] = adj[i].size();
            if (degree[i] == 1) leaves.push_back(i);
        }
        while (!leaves.empty()) {
            vector<int> new_leaves;
            for (int u : leaves) {
                for (int v : adj[u]) {
                    if (--degree[v] == 1) {
                        new_leaves.push_back(v);
                    }
                }
            }
            leaves = new_leaves;
        }
        return leaves;
    }

    // 67. Binary Tree Preorder
    vector<int> preorder(int u, int parent) {
        vector<int> result = {u};
        for (int v : adj[u]) {
            if (v != parent) {
                auto sub = preorder(v, u);
                result.insert(result.end(), sub.begin(), sub.end());
            }
        }
        return result;
    }

    // 68. Binary Tree Inorder
    vector<int> inorder(int u, int parent) {
        vector<int> result;
        bool first = true;
        for (int v : adj[u]) {
            if (v != parent) {
                if (first) {
                    auto left = inorder(v, u);
                    result.insert(result.end(), left.begin(), left.end());
                    result.push_back(u);
                    first = false;
                } else {
                    auto right = inorder(v, u);
                    result.insert(result.end(), right.begin(), right.end());
                }
            }
        }
        if (first) result.push_back(u);
        return result;
    }

    // 69. Binary Tree Postorder
    vector<int> postorder(int u, int parent) {
        vector<int> result;
        for (int v : adj[u]) {
            if (v != parent) {
                auto sub = postorder(v, u);
                result.insert(result.end(), sub.begin(), sub.end());
            }
        }
        result.push_back(u);
        return result;
    }

    // 70. Tree Height
    int treeHeight(int u, int parent) {
        int maxHeight = 0;
        for (int v : adj[u]) {
            if (v != parent) {
                maxHeight = max(maxHeight, treeHeight(v, u) + 1);
            }
        }
        return maxHeight;
    }
};

int main() {
    // Example usage
    Graph g(5);
    g.addEdge(0, 1);
    g.addEdge(1, 2);
    g.addEdge(2, 3);
    g.addEdge(3, 4);
    cout << "Is Bipartite (BFS): " << g.isBipartiteBFS() << endl;

    Tree t(5);
    t.addEdge(0, 1);
    t.addEdge(0, 2);
    t.addEdge(2, 3);
    t.addEdge(2, 4);
    cout << "Tree Diameter (DFS): " << t.treeDiameterDFS() << endl;
    return 0;
}
