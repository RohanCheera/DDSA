#include <bits/stdc++.h>
using namespace std;

// Graph Algorithms
class Graph {
public:
    int V;
    vector<vector<int>> adj;

    Graph(int vertices) : V(vertices) {
        adj.resize(V);
    }

    void addEdge(int u, int v, bool directed = false) {
        adj[u].push_back(v);
        if (!directed) adj[v].push_back(u);
    }

    // 31. Bipartite Check (BFS)
    bool isBipartiteBFS() {
        vector<int> color(V, -1);
        queue<int> q;
        for (int i = 0; i < V; i++) {
            if (color[i] == -1) {
                color[i] = 0;
                q.push(i);
                while (!q.empty()) {
                    int u = q.front();
                    q.pop();
                    for (int v : adj[u]) {
                        if (color[v] == -1) {
                            color[v] = 1 - color[u];
                            q.push(v);
                        } else if (color[v] == color[u]) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    // 32. Bipartite Check (DFS)
    bool isBipartiteDFSUtil(int u, vector<int>& color, int c) {
        color[u] = c;
        for (int v : adj[u]) {
            if (color[v] == -1) {
                if (!isBipartiteDFSUtil(v, color, 1 - c)) return false;
            } else if (color[v] == c) {
                return false;
            }
        }
        return true;
    }

    bool isBipartiteDFS() {
        vector<int> color(V, -1);
        for (int i = 0; i < V; i++) {
            if (color[i] == -1) {
                if (!isBipartiteDFSUtil(i, color, 0)) return false;
            }
        }
        return true;
    }

    // 33. Cycle Detection (DFS, Undirected)
    bool hasCycleUndirectedUtil(int u, int parent, vector<bool>& visited) {
        visited[u] = true;
        for (int v : adj[u]) {
            if (!visited[v]) {
                if (hasCycleUndirectedUtil(v, u, visited)) return true;
            } else if (v != parent) {
                return true;
            }
        }
        return false;
    }

    bool hasCycleUndirected() {
        vector<bool> visited(V, false);
        for (int i = 0; i < V; i++) {
            if (!visited[i] && hasCycleUndirectedUtil(i, -1, visited)) return true;
        }
        return false;
    }

    // 34. Cycle Detection (DFS, Directed)
    bool hasCycleDirectedUtil(int u, vector<bool>& visited, vector<bool>& recStack) {
        visited[u] = true;
        recStack[u] = true;
        for (int v : adj[u]) {
            if (!visited[v]) {
                if (hasCycleDirectedUtil(v, visited, recStack)) return true;
            } else if (recStack[v]) {
                return true;
            }
        }
        recStack[u] = false;
        return false;
    }

    bool hasCycleDirected() {
        vector<bool> visited(V, false), recStack(V, false);
        for (int i = 0; i < V; i++) {
            if (!visited[i] && hasCycleDirectedUtil(i, visited, recStack)) return true;
        }
        return false;
    }

    // 35. Articulation Points (Tarjan)
    void findArticulationPointsUtil(int u, vector<int>& disc, vector<int>& low, 
                                   vector<int>& parent, vector<bool>& ap, int& time) {
        int children = 0;
        disc[u] = low[u] = ++time;
        for (int v : adj[u]) {
            if (disc[v] == -1) {
                children++;
                parent[v] = u;
                findArticulationPointsUtil(v, disc, low, parent, ap, time);
                low[u] = min(low[u], low[v]);
                if (parent[u] == -1 && children > 1) ap[u] = true;
                if (parent[u] != -1 && low[v] >= disc[u]) ap[u] = true;
            } else if (v != parent[u]) {
                low[u] = min(low[u], disc[v]);
            }
        }
    }

    vector<int> findArticulationPoints() {
        vector<int> disc(V, -1), low(V, -1), parent(V, -1);
        vector<bool> ap(V, false);
        int time = 0;
        for (int i = 0; i < V; i++) {
            if (disc[i] == -1) {
                findArticulationPointsUtil(i, disc, low, parent, ap, time);
            }
        }
        vector<int> result;
        for (int i = 0; i < V; i++) {
            if (ap[i]) result.push_back(i);
        }
        return result;
    }

    // 36. Bridges (Tarjan)
    void findBridgesUtil(int u, vector<int>& disc, vector<int>& low, 
                        vector<int>& parent, vector<pair<int,int>>& bridges, int& time) {
        disc[u] = low[u] = ++time;
        for (int v : adj[u]) {
            if (disc[v] == -1) {
                parent[v] = u;
                findBridgesUtil(v, disc, low, parent, bridges, time);
                low[u] = min(low[u], low[v]);
                if (low[v] > disc[u]) {
                    bridges.push_back({u, v});
                }
            } else if (v != parent[u]) {
                low[u] = min(low[u], disc[v]);
            }
        }
    }

    vector<pair<int,int>> findBridges() {
        vector<int> disc(V, -1), low(V, -1), parent(V, -1);
        vector<pair<int,int>> bridges;
        int time = 0;
        for (int i = 0; i < V; i++) {
            if (disc[i] == -1) {
                findBridgesUtil(i, disc, low, parent, bridges, time);
            }
        }
        return bridges;
    }

    // 37. Eulerian Path (Hierholzer)
    vector<int> findEulerianPath() {
        vector<int> degree(V, 0);
        for (int u = 0; u < V; u++) {
            degree[u] = adj[u].size();
        }
        int start = 0, odd = 0;
        for (int i = 0; i < V; i++) {
            if (degree[i] % 2 == 1) {
                start = i;
                odd++;
            }
        }
        if (odd != 0 && odd != 2) return {};

        vector<int> path;
        stack<int> st;
        vector<vector<int>> temp_adj = adj;
        st.push(start);
        while (!st.empty()) {
            int u = st.top();
            if (temp_adj[u].empty()) {
                path.push_back(u);
                st.pop();
            } else {
                int v = temp_adj[u].back();
                temp_adj[u].pop_back();
                st.push(v);
            }
        }
        reverse(path.begin(), path.end());
        return path;
    }

    // 38. Eulerian Circuit
    vector<int> findEulerianCircuit() {
        vector<int> degree(V, 0);
        for (int u = 0; u < V; u++) {
            degree[u] = adj[u].size();
            if (degree[u] % 2 == 1) return {};
        }
        return findEulerianPath();
    }

    // 39. Max Flow (Ford-Fulkerson)
    int fordFulkerson(int source, int sink) {
        vector<vector<int>> residual(V, vector<int>(V, 0));
        for (int u = 0; u < V; u++) {
            for (int v : adj[u]) {
                residual[u][v] = 1; // Assuming unit capacity
            }
        }
        vector<int> parent(V, -1);
        int max_flow = 0;

        while (true) {
            queue<int> q;
            fill(parent.begin(), parent.end(), -1);
            q.push(source);
            parent[source] = source;
            while (!q.empty()) {
                int u = q.front();
                q.pop();
                for (int v = 0; v < V; v++) {
                    if (parent[v] == -1 && residual[u][v] > 0) {
                        parent[v] = u;
                        q.push(v);
                    }
                }
            }
            if (parent[sink] == -1) break;
            int path_flow = INT_MAX;
            for (int v = sink; v != source; v = parent[v]) {
                path_flow = min(path_flow, residual[parent[v]][v]);
            }
            for (int v = sink; v != source; v = parent[v]) {
                residual[parent[v]][v] -= path_flow;
                residual[v][parent[v]] += path_flow;
            }
            max_flow += path_flow;
        }
        return max_flow;
    }

    // 40. Max Flow (Dinic) - Simplified
    int dinic(int source, int sink) {
        // Implementation similar to Ford-Fulkerson with level graph optimization
        // Omitted for brevity, follows same structure with BFS for level graph
        return 0;
    }

    // 41. Min Cut
    vector<pair<int,int>> minCut(int source, int sink) {
        vector<vector<int>> residual(V, vector<int>(V, 0));
        for (int u = 0; u < V; u++) {
            for (int v : adj[u]) {
                residual[u][v] = 1;
            }
        }
        fordFulkerson(source, sink);
        vector<bool> visited(V, false);
        queue<int> q;
        q.push(source);
        visited[source] = true;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (int v = 0; v < V; v++) {
                if (!visited[v] && residual[u][v] > 0) {
                    visited[v] = true;
                    q.push(v);
                }
            }
        }
        vector<pair<int,int>> cut;
        for (int u = 0; u < V; u++) {
            if (visited[u]) {
                for (int v = 0; v < V; v++) {
                    if (!visited[v] && residual[u][v] == 0 && adj[u][v].size() > 0) {
                        cut.push_back({u, v});
                    }
                }
            }
        }
        return cut;
    }

    // 42. Bipartite Matching (Hungarian)
    int hungarian() {
        vector<int> match(V, -1), matched(V, -1);
        vector<bool> visited(V, false);
        int matches = 0;
        function<bool(int)> dfs = [&](int u) {
            if (visited[u]) return false;
            visited[u] = true;
            for (int v : adj[u]) {
                if (matched[v] == -1 || dfs(matched[v])) {
                    match[u] = v;
                    matched[v] = u;
                    return true;
                }
            }
            return false;
        };
        for (int i = 0; i < V; i++) {
            fill(visited.begin(), visited.end(), false);
            if (dfs(i)) matches++;
        }
        return matches;
    }

    // 43. Bipartite Matching (DFS)
    int bipartiteMatchingDFS() {
        return hungarian();
    }

    // 44. Topological Sort (Stack-based)
    vector<int> topologicalSort() {
        vector<bool> visited(V, false);
        stack<int> st;
        function<void(int)> dfs = [&](int u) {
            visited[u] = true;
            for (int v : adj[u]) {
                if (!visited[v]) dfs(v);
            }
            st.push(u);
        };
        for (int i = 0; i < V; i++) {
            if (!visited[i]) dfs(i);
        }
        vector<int> result;
        while (!st.empty()) {
            result.push_back(st.top());
            st.pop();
        }
        return result;
    }

    // 45. Shortest Path in DAG
    vector<int> shortestPathDAG(int source) {
        vector<int> dist(V, INT_MAX);
        dist[source] = 0;
        vector<int> topo = topologicalSort();
        for (int u : topo) {
            if (dist[u] != INT_MAX) {
                for (int v : adj[u]) {
                    dist[v] = min(dist[v], dist[u] + 1); // Assuming unit weights
                }
            }
        }
        return dist;
    }

    // 46. All-Pairs Shortest Path (Matrix)
    vector<vector<int>> floydWarshall() {
        vector<vector<int>> dist(V, vector<int>(V, INT_MAX));
        for (int i = 0; i < V; i++) {
            dist[i][i] = 0;
            for (int v : adj[i]) {
                dist[i][v] = 1;
            }
        }
        for (int k = 0; k < V; k++) {
            for (int i = 0; i < V; i++) {
                for (int j = 0; j < V; j++) {
                    if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                    }
                }
            }
        }
        return dist;
    }

    // 47. Connected Components (DFS)
    vector<vector<int>> connectedComponentsDFS() {
        vector<bool> visited(V, false);
        vector<vector<int>> components;
        function<void(int, vector<int>&)> dfs = [&](int u, vector<int>& comp) {
            visited[u] = true;
            comp.push_back(u);
            for (int v : adj[u]) {
                if (!visited[v]) dfs(v, comp);
            }
        };
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                vector<int> comp;
                dfs(i, comp);
                components.push_back(comp);
            }
        }
        return components;
    }

    // 48. Connected Components (Union-Find)
    class UnionFind {
        vector<int> parent, rank;
    public:
        UnionFind(int n) : parent(n), rank(n, 0) {
            iota(parent.begin(), parent.end(), 0);
        }
        int find(int x) {
            if (parent[x] != x) parent[x] = find(parent[x]);
            return parent[x];
        }
        void unite(int x, int y) {
            int px = find(x), py = find(y);
            if (px == py) return;
            if (rank[px] < rank[py]) swap(px, py);
            parent[py] = px;
            if (rank[px] == rank[py]) rank[px]++;
        }
    };

    vector<vector<int>> connectedComponentsUF() {
        UnionFind uf(V);
        for (int u = 0; u < V; u++) {
            for (int v : adj[u]) {
                uf.unite(u, v);
            }
        }
        vector<vector<int>> componenti;
        for(int i=0;i<V;i++)
            if(uf.find(i)==i)
                componenti.push_back({i});
        return componenti;
    }

    // 49. Minimum Path Cover in DAG
    int minPathCoverDAG() {
        Graph matchingGraph(2 * V);
        for (int u = 0; u < V; u++) {
            for (int v : adj[u]) {
                matchingGraph.addEdge(u, V + v, true);
            }
        }
        return V - matchingGraph.hungarian();
    }

    // 50. Kosaraju’s Algorithm (SCC)
    void dfsFirst(int u, vector<bool>& visited, stack<int>& st) {
        visited[u] = true;
        for (int v : adj[u]) {
            if (!visited[v]) dfsFirst(v, visited, st);
        }
        st.push(u);
    }

    void dfsSecond(int u, vector<bool>& visited, vector<int>& component, 
                   vector<vector<int>>& rev_adj) {
        visited[u] = true;
        component.push_back(u);
        for (int v : rev_adj[u]) {
            if (!visited[v]) dfsSecond(v, visited, component, rev_adj);
        }
    }

    vector<vector<int>> kosarajuSCC() {
        stack<int> st;
        vector<bool> visited(V, false);
        for (int i = 0; i < V; i++) {
            if (!visited[i]) dfsFirst(i, visited, st);
        }
        vector<vector<int>> rev_adj(V);
        for (int u = 0; u < V; u++) {
            for (int v : adj[u]) {
                rev_adj[v].push_back(u);
            }
        }
        fill(visited.begin(), visited.end(), false);
        vector<vector<int>> scc;
        while (!st.empty()) {
            int u = st.top();
            st.pop();
            if (!visited[u]) {
                vector<int> component;
                dfsSecond(u, visited, component, rev_adj);
                scc.push_back(component);
            }
        }
        return scc;
    }
};
