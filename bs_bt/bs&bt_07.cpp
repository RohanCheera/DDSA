#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int MOD = 1e9 + 7;
const int INF = 1e9;
const ll LINF = 1e18;

// 121. Search in Rotated Sorted Array
// Time: O(log n), Space: O(1)
// Purpose: Find target in rotated sorted array
int searchRotated(vector<int>& nums, int target) {
    int l = 0, r = nums.size() - 1;
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (nums[m] == target) return m;
        if (nums[l] <= nums[m]) { // Left half is sorted
            if (nums[l] <= target && target < nums[m]) r = m - 1;
            else l = m + 1;
        } else { // Right half is sorted
            if (nums[m] < target && target <= nums[r]) l = m + 1;
            else r = m - 1;
        }
    }
    return -1;
}

// 122. Find Minimum in Rotated Sorted Array
// Time: O(log n), Space: O(1)
// Purpose: Find minimum element in rotated sorted array
int findMinRotated(vector<int>& nums) {
    int l = 0, r = nums.size() - 1;
    while (l < r) {
        int m = l + (r - l) / 2;
        if (nums[m] > nums[r]) l = m + 1;
        else r = m;
    }
    return nums[l];
}

// 123. Median of Two Sorted Arrays
// Time: O(log(min(n, m))), Space: O(1)
// Purpose: Find median of two sorted arrays
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    if (nums1.size() > nums2.size()) swap(nums1, nums2);
    int n1 = nums1.size(), n2 = nums2.size();
    int l = 0, r = n1;
    while (l <= r) {
        int cut1 = (l + r) / 2;
        int cut2 = (n1 + n2 + 1) / 2 - cut1;
        int left1 = cut1 == 0 ? INT_MIN : nums1[cut1 - 1];
        int left2 = cut2 == 0 ? INT_MIN : nums2[cut2 - 1];
        int right1 = cut1 == n1 ? INT_MAX : nums1[cut1];
        int right2 = cut2 == n2 ? INT_MAX : nums2[cut2];
        if (left1 <= right2 && left2 <= right1) {
            if ((n1 + n2) % 2 == 0) return (max(left1, left2) + min(right1, right2)) / 2.0;
            return max(left1, left2);
        }
        if (left1 > right2) r = cut1 - 1;
        else l = cut1 + 1;
    }
    return 0.0; // Should not reach here
}

// 124. Capacity to Ship Packages
// Time: O(n log sum), Space: O(1)
// Purpose: Find minimum capacity to ship packages in D days
int shipWithinDays(vector<int>& weights, int days) {
    int l = *max_element(weights.begin(), weights.end());
    int r = accumulate(weights.begin(), weights.end(), 0);
    while (l < r) {
        int m = l + (r - l) / 2;
        int currDays = 1, currSum = 0;
        for (int w : weights) {
            if (currSum + w > m) { currDays++; currSum = w; }
            else currSum += w;
        }
        if (currDays <= days) r = m;
        else l = m + 1;
    }
    return l;
}

// 125. Split Array Largest Sum
// Time: O(n log sum), Space: O(1)
// Purpose: Minimize largest sum of subarrays after splitting into m parts
int splitArray(vector<int>& nums, int m) {
    ll l = *max_element(nums.begin(), nums.end());
    ll r = accumulate(nums.begin(), nums.end(), 0LL);
    while (l < r) {
        ll mid = l + (r - l) / 2;
        int count = 1;
        ll sum = 0;
        for (int x : nums) {
            if (sum + x > mid) { count++; sum = x; }
            else sum += x;
        }
        if (count <= m) r = mid;
        else l = mid + 1;
    }
    return l;
}

// 126. Find First and Last Position
// Time: O(log n), Space: O(1)
// Purpose: Find first and last occurrence of target
vector<int> searchRange(vector<int>& nums, int target) {
    auto binarySearch = [&](bool findFirst) {
        int l = 0, r = nums.size() - 1, res = -1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (nums[m] == target) {
                res = m;
                if (findFirst) r = m - 1;
                else l = m + 1;
            } else if (nums[m] < target) l = m + 1;
            else r = m - 1;
        }
        return res;
    };
    return {binarySearch(true), binarySearch(false)};
}

// 127. Sqrt(x)
// Time: O(log n), Space: O(1)
// Purpose: Compute integer square root
int mySqrt(int x) {
    if (x == 0) return 0;
    int l = 1, r = x;
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (m <= x / m && (m + 1) > x / (m + 1)) return m;
        if (m > x / m) r = m - 1;
        else l = m + 1;
    }
    return l;
}

// 128. Find Peak Element
// Time: O(log n), Space: O(1)
// Purpose: Find a peak element in array
int findPeakElement(vector<int>& nums) {
    int l = 0, r = nums.size() - 1;
    while (l < r) {
        int m = l + (r - l) / 2;
        if (nums[m] < nums[m + 1]) l = m + 1;
        else r = m;
    }
    return l;
}

// 129. Minimum Time to Complete Trips
// Time: O(n log time), Space: O(1)
// Purpose: Find minimum time for trips
ll minimumTime(vector<int>& time, int totalTrips) {
    ll l = 1, r = 1LL * *min_element(time.begin(), time.end()) * totalTrips;
    while (l < r) {
        ll m = l + (r - l) / 2;
        ll trips = 0;
        for (int t : time) trips += m / t;
        if (trips >= totalTrips) r = m;
        else l = m + 1;
} else l = m + 1;
    }
    return l;
}

// 130. Single Number
// Time: O(n), Space: O(1)
// Purpose: Find number appearing once using XOR
int singleNumber(vector<int>& nums) {
    int res = 0;
    for (int x : nums) res ^= x;
    return res;
}

// 131. Number of 1 Bits
// Time: O(1), Space: O(1)
// Purpose: Count set bits in 32-bit integer
int hammingWeight(uint32_t n) {
    return __builtin_popcount(n);
}

// 132. Reverse Bits
// Time: O(1), Space: O(1)
// Purpose: Reverse bits of 32-bit integer
uint32_t reverseBits(uint32_t n) {
    uint32_t res = 0;
    for (int i = 0; i < 32; i++) {
        res = (res << 1) | (n & 1);
        n >>= 1;
    }
    return res;
}

// 133. Sum of Two Integers
// Time: O(1), Space: O(1)
// Purpose: Add two integers without + operator
int getSum(int a, int b) {
    while (b) {
        int carry = ((unsigned int)(a & b)) << 1;
        a ^= b;
        b = carry;
    }
    return a;
}

// 134. Missing Number
// Time: O(n), Space: O(1)
// Purpose: Find missing number using XOR
int missingNumber(vector<int>& nums) {
    int res = nums.size();
    for (int i = 0; i < nums.size(); i++) res ^= i ^ nums[i];
    return res;
}

// 135. Subset XOR Sum
// Time: O(n), Space: O(1)
// Purpose: Compute XOR sum of all subsets
int subsetXORSum(vector<int>& nums) {
    int res = 0;
    for (int x : nums) res |= x;
    return res << (nums.size() - 1);
}

// 136. Maximum XOR of Two Numbers
// Time: O(n * 32), Space: O(n)
// Purpose: Find maximum XOR using trie
struct TrieNode {
    TrieNode* children[2];
    TrieNode() { children[0] = children[1] = nullptr; }
};
int maxXOR(vector<int>& nums) {
    TrieNode* root = new TrieNode();
    auto insert = [&](int num) {
        TrieNode* node = root;
        for (int i = 31; i >= 0; i--) {
            int bit = (num >> i) & 1;
            if (!node->children[bit]) node->children[bit] = new TrieNode();
            node = node->children[bit];
        }
    };
    auto query = [&](int num) {
        TrieNode* node = root;
        int res = 0;
        for (int i = 31; i >= 0; i--) {
            int bit = (num >> i) & 1;
            if (node->children[!bit]) {
                res |= (1 << i);
                node = node->children[!bit];
            } else node = node->children[bit];
        }
        return res;
    };
    int ans = 0;
    for (int x : nums) {
        insert(x);
        ans = max(ans, query(x));
    }
    return ans;
}

// 137. Bitwise AND of Numbers Range
// Time: O(1), Space: O(1)
// Purpose: Find common prefix of bits
int rangeBitwiseAnd(int left, int right) {
    int shift = 0;
    while (left != right) {
        left >>= 1;
        right >>= 1;
        shift++;
    }
    return left << shift;
}

// 138. Divide Two Integers
// Time: O(log n), Space: O(1)
// Purpose: Divide integers using bit manipulation
int divide(int dividend, int divisor) {
    if (dividend == INT_MIN && divisor == -1) return INT_MAX;
    ll a = abs((ll)dividend), b = abs((ll)divisor), res = 0;
    while (a >= b) {
        ll temp = b, m = 1;
        while (a >= (temp << 1)) {
            temp <<= 1;
            m <<= 1;
        }
        a -= temp;
        res += m;
    }
    return (dividend > 0) == (divisor > 0) ? res : -res;
}

// 139. Gray Code
// Time: O(2^n), Space: O(1)
// Purpose: Generate Gray code sequence
vector<int> grayCode(int n) {
    vector<int> res;
    for (int i = 0; i < (1 << n); i++) res.push_back(i ^ (i >> 1));
    return res;
}

// 140. Count Bits
// Time: O(n log n), Space: O(n)
// Purpose: Count set bits for 0 to n
vector<int> countBits(int n) {
    vector<int> res(n + 1);
    for (int i = 0; i <= n; i++) res[i] = __builtin_popcount(i);
    return res;
}

// 141. Jump Game
// Time: O(n), Space: O(1)
// Purpose: Check if end is reachable
bool canJump(vector<int>& nums) {
    int maxReach = 0;
    for (int i = 0; i < nums.size() && i <= maxReach; i++) {
        maxReach = max(maxReach, i + nums[i]);
        if (maxReach >= nums.size() - 1) return true;
    }
    return false;
}

// 142. Jump Game II
// Time: O(n), Space: O(1)
// Purpose: Find minimum jumps to reach end
int jump(vector<int>& nums) {
    int jumps = 0, currEnd = 0, currFarthest = 0;
    for (int i = 0; i < nums.size() - 1; i++) {
        currFarthest = max(currFarthest, i + nums[i]);
        if (i == currEnd) {
            jumps++;
            currEnd = currFarthest;
        }
    }
    return jumps;
}

// 143. Gas Station
// Time: O(n), Space: O(1)
// Purpose: Find starting point for circular tour
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
    int total = 0, curr = 0, start = 0;
    for (int i = 0; i < gas.size(); i++) {
        total += gas[i] - cost[i];
        curr += gas[i] - cost[i];
        if (curr < 0) { curr = 0; start = i + 1; }
    }
    return total >= 0 ? start : -1;
}

// 144. Minimum Number of Arrows
// Time: O(n log n), Space: O(1)
// Purpose: Find minimum arrows to burst balloons
int findMinArrowShots(vector<vector<int>>& points) {
    sort(points.begin(), points.end(), [](auto& a, auto& b) { return a[1] < b[1]; });
    int arrows = 1, end = points[0][1];
    for (int i = 1; i < points.size(); i++) {
        if (points[i][0] > end) { arrows++; end = points[i][1]; }
    }
    return arrows;
}

// 145. Partition Labels
// Time: O(n), Space: O(1)
// Purpose: Partition string by last occurrence
vector<int> partitionLabels(string s) {
    vector<int> last(26, -1);
    for (int i = 0; i < s.size(); i++) last[s[i] - 'a'] = i;
    vector<int> res;
    int start = 0, end = 0;
    for (int i = 0; i < s.size(); i++) {
        end = max(end, last[s[i] - 'a']);
        if (i == end) { res.push_back(end - start + 1); start = i + 1; }
    }
    return res;
}

// 146. Non-overlapping Intervals
// Time: O(n log n), Space: O(1)
// Purpose: Remove minimum intervals to make non-overlapping
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end(), [](auto& a, auto& b) { return a[1] < b[1]; });
    int count = 0, end = -INF;
    for (auto& interval : intervals) {
        if (interval[0] >= end) end = interval[1];
        else count++;
    }
    return count;
}

// 147. Meeting Rooms II
// Time: O(n log n), Space: O(n)
// Purpose: Find minimum rooms for meetings
int minMeetingRooms(vector<vector<int>>& intervals) {
    vector<int> start, end;
    for (auto& interval : intervals) {
        start.push_back(interval[0]);
        end.push_back(interval[1]);
    }
    sort(start.begin(), start.end());
    sort(end.begin(), end.end());
    int rooms = 0, maxRooms = 0, i = 0, j = 0;
    while (i < intervals.size()) {
        if (start[i] < end[j]) { rooms++; maxRooms = max(maxRooms, rooms); i++; }
        else { rooms--; j++; }
    }
    return maxRooms;
}

// 148. Assign Cookies
// Time: O(n log n), Space: O(1)
// Purpose: Assign cookies to children greedily
int findContentChildren(vector<int>& g, vector<int>& s) {
    sort(g.begin(), g.end());
    sort(s.begin(), s.end());
    int i = 0, j = 0;
    while (i < g.size() && j < s.size()) {
        if (s[j] >= g[i]) i++;
        j++;
    }
    return i;
}

// 149. Lemonade Change
// Time: O(n), Space: O(1)
// Purpose: Handle change for lemonade stand
bool lemonadeChange(vector<int>& bills) {
    int five = 0, ten = 0;
    for (int bill : bills) {
        if (bill == 5) five++;
        else if (bill == 10) { five--; ten++; }
        else if (ten > 0) { ten--; five--; }
        else five -= 3;
        if (five < 0) return false;
    }
    return true;
}

// 150. Candy
// Time: O(n), Space: O(n)
// Purpose: Distribute candies based on ratings
int candy(vector<int>& ratings) {
    int n = ratings.size();
    vector<int> candies(n, 1);
    for (int i = 1; i < n; i++) if (ratings[i] > ratings[i - 1]) candies[i] = candies[i - 1] + 1;
    for (int i = n - 2; i >= 0; i--) if (ratings[i] > ratings[i + 1]) candies[i] = max(candies[i], candies[i + 1] + 1);
    return accumulate(candies.begin(), candies.end(), 0);
}

// 151. Task Scheduler
// Time: O(n), Space: O(1)
// Purpose: Schedule tasks with cooldown
int leastInterval(vector<char>& tasks, int n) {
    vector<int> freq(26);
    for (char c : tasks) freq[c - 'A']++;
    int maxFreq = *max_element(freq.begin(), freq.end());
    int maxCount = count(freq.begin(), freq.end(), maxFreq);
    return max((int)tasks.size(), (maxFreq - 1) * (n + 1) + maxCount);
}

// 152. Minimum Cost to Hire K Workers
// Time: O(n log n), Space: O(n)
// Purpose: Minimize cost to hire k workers
double mincostToHireWorkers(vector<int>& quality, vector<int>& wage, int k) {
    int n = quality.size();
    vector<pair<double, int>> ratio;
    for (int i = 0; i < n; i++) ratio.push_back({(double)wage[i] / quality[i], quality[i]});
    sort(ratio.begin(), ratio.end());
    priority_queue<int> pq;
    ll totalQuality = 0;
    double minCost = LINF;
    for (auto [r, q] : ratio) {
        pq.push(q);
        totalQuality += q;
        if (pq.size() > k) { totalQuality -= pq.top(); pq.pop(); }
        if (pq.size() == k) minCost = min(minCost, r * totalQuality);
    }
    return minCost;
}

// 153. Reorganize String
// Time: O(n log n), Space: O(n)
// Purpose: Rearrange string to avoid adjacent duplicates
string reorganizeString(string s) {
    vector<int> freq(26);
    for (char c : s) freq[c - 'a']++;
    priority_queue<pair<int, char>> pq;
    for (int i = 0; i < 26; i++) if (freq[i]) pq.push({freq[i], 'a' + i});
    string res;
    while (!pq.empty()) {
        auto [f1, c1] = pq.top(); pq.pop();
        if (res.empty() || res.back() != c1) {
            res += c1;
            if (f1 > 1) pq.push({f1 - 1, c1});
        } else if (!pq.empty()) {
            auto [f2, c2] = pq.top(); pq.pop();
            res += c2;
            if (f2 > 1) pq.push({f2 - 1, c2});
            pq.push({f1, c1});
        } else return "";
    }
    return res;
}

// 154. Maximum Subarray (Kadane’s Algorithm)
// Time: O(n), Space: O(1)
// Purpose: Find maximum sum subarray
int maxSubArray(vector<int>& nums) {
    int maxSum = nums[0], currSum = nums[0];
    for (int i = 1; i < nums.size(); i++) {
        currSum = max(nums[i], currSum + nums[i]);
        maxSum = max(maxSum, currSum);
    }
    return maxSum;
}

// 155. Minimum Platforms
// Time: O(n log n), Space: O(1)
// Purpose: Find minimum platforms for train schedules
int findPlatform(vector<int>& arr, vector<int>& dep) {
    sort(arr.begin(), arr.end());
    sort(dep.begin(), dep.end());
    int platforms = 0, maxPlatforms = 0, i = 0, j = 0;
    while (i < arr.size()) {
        if (arr[i] <= dep[j]) { platforms++; maxPlatforms = max(maxPlatforms, platforms); i++; }
        else { platforms--; j++; }
    }
    return maxPlatforms;
}

// 156. Fractional Knapsack
// Time: O(n log n), Space: O(1)
// Purpose: Maximize value in fractional knapsack
double fractionalKnapsack(vector<int>& values, vector<int>& weights, int capacity) {
    vector<pair<double, int>> ratio;
    for (int i = 0; i < values.size(); i++) ratio.push_back({(double)values[i] / weights[i], weights[i]});
    sort(ratio.rbegin(), ratio.rend());
    double totalValue = 0;
    int currWeight = 0;
    for (auto [r, w] : ratio) {
        if (currWeight + w <= capacity) {
            currWeight += w;
            totalValue += r * w;
        } else {
            totalValue += r * (capacity - currWeight);
            break;
        }
    }
    return totalValue;
}

// 157. Job Sequencing
// Time: O(n log n), Space: O(n)
// Purpose: Maximize profit by scheduling jobs
struct Job { int id, deadline, profit; };
int jobScheduling(vector<Job>& jobs) {
    sort(jobs.begin(), jobs.end(), [](Job& a, Job& b) { return a.profit > b.profit; });
    int maxDeadline = 0;
    for (auto& job : jobs) maxDeadline = max(maxDeadline, job.deadline);
    vector<int> slots(maxDeadline + 1, -1);
    int totalProfit = 0;
    for (auto& job : jobs) {
        for (int j = min(maxDeadline, job.deadline); j > 0; j--) {
            if (slots[j] == -1) {
                slots[j] = job.id;
                totalProfit += job.profit;
                break;
            }
        }
    }
    return totalProfit;
}

// 158. Maximize Sum After K Negations
// Time: O(n log n), Space: O(n)
// Purpose: Maximize sum by negating k elements
ll largestSumAfterKNegations(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<>> pq(nums.begin(), nums.end());
    while (k--) {
        int x = pq.top(); pq.pop();
        pq.push(-x);
    }
    ll sum = 0;
    while (!pq.empty()) { sum += pq.top(); pq.pop(); }
    return sum;
}

// 159. Minimum Cost to Cut Board
// Time: O(n log n), Space: O(n)
// Purpose: Minimize cost of cutting board
ll minimumCostToCutBoard(vector<int>& cuts) {
    sort(cuts.begin(), cuts.end());
    cuts.insert(cuts.begin(), 0);
    cuts.push_back(cuts.back() + 1);
    int n = cuts.size();
    vector<vector<ll>> dp(n, vector<ll>(n, LINF));
    for (int len = 2; len < n; len++) {
        for (int i = 0; i + len < n; i++) {
            int j = i + len;
            for (int k = i + 1; k < j; k++) {
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j] + cuts[j] - cuts[i]);
            }
        }
    }
    return dp[0][n - 1];
}

// 160. Split Array into Consecutive Subsequences
// Time: O(n), Space: O(n)
// Purpose: Check if array can be split into consecutive subsequences
bool isPossible(vector<int>& nums) {
    unordered_map<int, int> count, need;
    for (int x : nums) count[x]++;
    for (int x : nums) {
        if (count[x] == 0) continue;
        count[x]--;
        if (need[x] > 0) {
            need[x]--;
            need[x + 1]++;
        } else if (count[x + 1] > 0 && count[x + 2] > 0) {
            count[x + 1]--;
            count[x + 2]--;
            need[x + 3]++;
        } else return false;
    }
    return true;
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr); cout.tie(nullptr);
    // Example usage of templates
    return 0;
}
