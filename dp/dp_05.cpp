#include <bits/stdc++.h>
using namespace std;

// Dynamic Programming Algorithms
class DP {
public:
    // 71. Knapsack 0/1
    int knapsack01(int W, vector<int>& weights, vector<int>& values) {
        int n = weights.size();
        vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));
        for (int i = 1; i <= n; i++) {
            for (int w = 0; w <= W; w++) {
                if (weights[i-1] <= w) {
                    dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1]);
                } else {
                    dp[i][w] = dp[i-1][w];
                }
            }
        }
        return dp[n][W];
    }

    // 72. Knapsack Unbounded
    int knapsackUnbounded(int W, vector<int>& weights, vector<int>& values) {
        int n = weights.size();
        vector<int> dp(W + 1, 0);
        for (int w = 0; w <= W; w++) {
            for (int i = 0; i < n; i++) {
                if (weights[i] <= w) {
                    dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
                }
            }
        }
        return dp[W];
    }

    // 73. Longest Increasing Subsequence
    int longestIncreasingSubsequence(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n, 1);
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
        }
        return *max_element(dp.begin(), dp.end());
    }

    // 74. Longest Common Subsequence
    int longestCommonSubsequence(string s1, string s2) {
        int n = s1.size(), m = s2.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (s1[i-1] == s2[j-1]) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                } else {
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        return dp[n][m];
    }

    // 75. Edit Distance
    int editDistance(string s1, string s2) {
        int n = s1.size(), m = s2.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
        for (int i = 0; i <= n; i++) dp[i][0] = i;
        for (int j = 0; j <= m; j++) dp[0][j] = j;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (s1[i-1] == s2[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                } else {
                    dp[i][j] = min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]}) + 1;
                }
            }
        }
        return dp[n][m];
    }

    // 76. Coin Change (Min Coins)
    int coinChangeMinCoins(vector<int>& coins, int amount) {
        vector<int> dp(amount + 1, INT_MAX);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i && dp[i - coin] != INT_MAX) {
                    dp[i] = min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount] == INT_MAX ? -1 : dp[amount];
    }

    // 77. Coin Change (Count Ways)
    int coinChangeCountWays(vector<int>& coins, int amount) {
        vector<int> dp(amount + 1, 0);
        dp[0] = 1;
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] += dp[i - coin];
            }
        }
        return dp[amount];
    }

    // 78. Subset Sum
    bool subsetSum(vector<int>& nums, int sum) {
        int n = nums.size();
        vector<vector<bool>> dp(n + 1, vector<bool>(sum + 1, false));
        for (int i = 0; i <= n; i++) dp[i][0] = true;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= sum; j++) {
                if (nums[i-1] <= j) {
                    dp[i][j] = dp[i-1][j] || dp[i-1][j - nums[i-1]];
                } else {
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        return dp[n][sum];
    }

    // 79. Longest Palindromic Subsequence
    int longestPalindromicSubsequence(string s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        for (int i = 0; i < n; i++) dp[i][i] = 1;
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                if (s[i] == s[j] && len == 2) {
                    dp[i][j] = 2;
                } else if (s[i] == s[j]) {
                    dp[i][j] = dp[i+1][j-1] + 2;
                } else {
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1]);
                }
            }
        }
        return dp[0][n-1];
    }

    // 80. Matrix Chain Multiplication
    int matrixChainMultiplication(vector<int>& dims) {
        int n = dims.size() - 1;
        vector<vector<int>> dp(n, vector<int>(n, INT_MAX));
        for (int i = 0; i < n; i++) dp[i][i] = 0;
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                for (int k = i; k < j; k++) {
                    dp[i][j] = min(dp[i][j], 
                        dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]);
                }
            }
        }
        return dp[0][n-1];
    }

    // 81. Palindrome Partitioning
    int palindromePartitioning(string s) {
        int n = s.size();
        vector<vector<bool>> isPal(n, vector<bool>(n, false));
        vector<int> dp(n + 1, INT_MAX);
        for (int i = 0; i < n; i++) isPal[i][i] = true;
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                isPal[i][j] = (s[i] == s[j] && (len == 2 || isPal[i+1][j-1]));
            }
        }
        dp[0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                if (isPal[j][i-1]) {
                    dp[i] = min(dp[i], dp[j] + 1);
                }
            }
        }
        return dp[n] - 1;
    }

    // 82. Longest Valid Parentheses
    int longestValidParentheses(string s) {
        int n = s.size(), maxLen = 0;
        vector<int> dp(n, 0);
        for (int i = 1; i < n; i++) {
            if (s[i] == ')') {
                if (s[i-1] == '(') {
                    dp[i] = (i >= 2 ? dp[i-2] : 0) + 2;
                } else if (i - dp[i-1] - 1 >= 0 && s[i - dp[i-1] - 1] == '(') {
                    dp[i] = dp[i-1] + 2 + (i - dp[i-1] - 2 >= 0 ? dp[i - dp[i-1] - 2] : 0);
                }
                maxLen = max(maxLen, dp[i]);
            }
        }
        return maxLen;
    }

    // 83. Word Break
    bool wordBreak(string s, vector<string>& wordDict) {
        int n = s.size();
        vector<bool> dp(n + 1, false);
        dp[0] = true;
        for (int i = 1; i <= n; i++) {
            for (const string& word : wordDict) {
                int len = word.size();
                if (i >= len && s.substr(i - len, len) == word) {
                    dp[i] = dp[i] || dp[i - len];
                }
            }
        }
        return dp[n];
    }

    // 84. Decode Ways
    int decodeWays(string s) {
        int n = s.size();
        vector<int> dp(n + 1, 0);
        dp[0] = 1;
        dp[1] = s[0] == '0' ? 0 : 1;
        for (int i = 2; i <= n; i++) {
            if (s[i-1] != '0') {
                dp[i] += dp[i-1];
            }
            int twoDigit = stoi(s.substr(i-2, 2));
            if (twoDigit >= 10 && twoDigit <= 26) {
                dp[i] += dp[i-2];
            }
        }
        return dp[n];
    }

    // 85. Unique Paths
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n, 0));
        dp[0][0] = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i > 0) dp[i][j] += dp[i-1][j];
                if (j > 0) dp[i][j] += dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }

    // 86. Minimum Path Sum
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<int>> dp(m, vector<int>(n, 0));
        dp[0][0] = grid[0][0];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i > 0 || j > 0) dp[i][j] = INT_MAX;
                if (i > 0) dp[i][j] = min(dp[i][j], dp[i-1][j] + grid[i][j]);
                if (j > 0) dp[i][j] = min(dp[i][j], dp[i][j-1] + grid[i][j]);
            }
        }
        return dp[m-1][n-1];
    }

    // 87. House Robber
    int houseRobber(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) return 0;
        int prev2 = 0, prev1 = nums[0];
        for (int i = 1; i < n; i++) {
            int curr = max(prev1, prev2 + nums[i]);
            prev2 = prev1;
            prev1 = curr;
        }
        return prev1;
    }

    // 88. House Robber II
    int houseRobberII(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) return 0;
        if (n == 1) return nums[0];
        vector<int> nums1(nums.begin(), nums.end() - 1);
        vector<int> nums2(nums.begin() + 1, nums.end());
        return max(houseRobber(nums1), houseRobber(nums2));
    }

    // 89. Max Product Subarray
    int maxProductSubarray(vector<int>& nums) {
        int maxProd = nums[0], minProd = nums[0], result = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            int temp = maxProd;
            maxProd = max({nums[i], nums[i] * maxProd, nums[i] * minProd});
            minProd = min({nums[i], nums[i] * temp, nums[i] * minProd});
            result = max(result, maxProd);
        }
        return result;
    }

    // 90. Regular Expression Matching
    bool regularExpressionMatching(string s, string p) {
        int n = s.size(), m = p.size();
        vector<vector<bool>> dp(n + 1, vector<bool>(m + 1, false));
        dp[0][0] = true;
        for (int j = 2; j <= m; j++) {
            if (p[j-1] == '*') dp[0][j] = dp[0][j-2];
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (p[j-1] == '*') {
                    dp[i][j] = dp[i][j-2] || 
                              (dp[i-1][j] && (s[i-1] == p[j-2] || p[j-2] == '.'));
                } else if (p[j-1] == '.' || s[i-1] == p[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                }
            }
        }
        return dp[n][m];
    }

    // 91. Wildcard Matching
    bool wildcardMatching(string s, string p) {
        int n = s.size(), m = p.size();
        vector<vector<bool>> dp(n + 1, vector<bool>(m + 1, false));
        dp[0][0] = true;
        for (int j = 1; j <= m; j++) {
            if (p[j-1] == '*') dp[0][j] = dp[0][j-1];
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (p[j-1] == '*') {
                    dp[i][j] = dp[i][j-1] || dp[i-1][j];
                } else if (p[j-1] == '?' || s[i-1] == p[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                }
            }
        }
        return dp[n][m];
    }

    // 92. Burst Balloons
    int burstBalloons(vector<int>& nums) {
        nums.insert(nums.begin(), 1);
        nums.push_back(1);
        int n = nums.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        for (int len = 2; len < n; len++) {
            for (int i = 0; i < n - len; i++) {
                int j = i + len;
                for (int k = i + 1; k < j; k++) {
                    dp[i][j] = max(dp[i][j], 
                        dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j]);
                }
            }
        }
        return dp[0][n-1];
    }

    // 93. Digit DP
    int digitDP(string num, int k) {
        int n = num.size();
        vector<vector<vector<int>>> dp(n + 1, 
            vector<vector<int>>(k + 1, vector<int>(2, -1)));
        function<int(int, int, bool)> solve = [&](int pos, int sum, bool tight) {
            if (pos == n) return sum == k ? 1 : 0;
            if (dp[pos][sum][tight] != -1) return dp[pos][sum][tight];
            int ans = 0, limit = tight ? num[pos] - '0' : 9;
            for (int d = 0; d <= limit; d++) {
                ans += solve(pos + 1, sum + d, tight && d == limit);
            }
            return dp[pos][sum][tight] = ans;
        };
        return solve(0, 0, true);
    }

    // 94. Bitmask DP
    int bitmaskDP(vector<vector<int>>& cost) {
        int n = cost.size();
        vector<int> dp(1 << n, INT_MAX);
        dp[0] = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            for (int i = 0; i < n; i++) {
                if (!(mask & (1 << i))) {
                    dp[mask | (1 << i)] = min(dp[mask | (1 << i)], 
                                            dp[mask] + cost[mask][i]);
                }
            }
        }
        return dp[(1 << n) - 1];
    }

    // 95. Interval DP
    int intervalDP(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                for (int k = i; k < j; k++) {
                    dp[i][j] = max(dp[i][j], dp[i][k] + dp[k+1][j]);
                }
            }
        }
        return dp[0][n-1];
    }

    // 96. Tree DP
    int treeDP(vector<vector<int>>& adj, vector<int>& values, int u, int parent) {
        int sum = values[u];
        for (int v : adj[u]) {
            if (v != parent) {
                sum += treeDP(adj, values, v, u);
            }
        }
        return sum;
    }

    // 97. Knapsack with Multiple Constraints
    int knapsackMultipleConstraints(int W1, int W2, vector<int>& weights1, 
                                  vector<int>& weights2, vector<int>& values) {
        int n = weights1.size();
        vector<vector<vector<int>>> dp(n + 1, 
            vector<vector<int>>(W1 + 1, vector<int>(W2 + 1, 0)));
        for (int i = 1; i <= n; i++) {
            for (int w1 = 0; w1 <= W1; w1++) {
                for (int w2 = 0; w2 <= W2; w2++) {
                    dp[i][w1][w2] = dp[i-1][w1][w2];
                    if (w1 >= weights1[i-1] && w2 >= weights2[i-1]) {
                        dp[i][w1][w2] = max(dp[i][w1][w2], 
                            dp[i-1][w1-weights1[i-1]][w2-weights2[i-1]] + values[i-1]);
                    }
                }
            }
        }
        return dp[n][W1][W2];
    }

    // 98. Longest Zigzag Subsequence
    int longestZigzagSubsequence(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> dp(n, vector<int>(2, 1));
        int maxLen = 1;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i][0] = max(dp[i][0], dp[j][1] + 1);
                } else if (nums[i] < nums[j]) {
                    dp[i][1] = max(dp[i][1], dp[j][0] + 1);
                }
                maxLen = max({maxLen, dp[i][0], dp[i][1]});
            }
        }
        return maxLen;
    }

    // 99. Min Cost to Climb Stairs
    int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();
        int prev2 = 0, prev1 = 0;
        for (int i = 2; i <= n; i++) {
            int curr = min(prev1 + cost[i-1], prev2 + cost[i-2]);
            prev2 = prev1;
            prev1 = curr;
        }
        return prev1;
    }

    // 100. Max Sum After Partitioning
    int maxSumAfterPartitioning(vector<int>& arr, int k) {
        int n = arr.size();
        vector<int> dp(n + 1, 0);
        for (int i = 1; i <= n; i++) {
            int currMax = 0;
            for (int j = 1; j <= min(k, i); j++) {
                currMax = max(currMax, arr[i-j]);
                dp[i] = max(dp[i], dp[i-j] + currMax * j);
            }
        }
        return dp[n];
    }
};

