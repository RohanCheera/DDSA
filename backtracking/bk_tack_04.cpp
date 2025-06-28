#include <bits/stdc++.h>
using namespace std;

class Backtracking {
public:
    // 161. N-Queens
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> result;
        vector<string> board(n, string(n, '.'));
        vector<int> row(n, 0), diag1(2*n-1, 0), diag2(2*n-1, 0);
        solveNQueensUtil(0, n, board, result, row, diag1, diag2);
        return result;
    }
    
private:
    void solveNQueensUtil(int col, int n, vector<string>& board, vector<vector<string>>& result,
                         vector<int>& row, vector<int>& diag1, vector<int>& diag2) {
        if (col == n) {
            result.push_back(board);
            return;
        }
        for (int i = 0; i < n; i++) {
            if (!row[i] && !diag1[i+col] && !diag2[i-col+n-1]) {
                board[i][col] = 'Q';
                row[i] = diag1[i+col] = diag2[i-col+n-1] = 1;
                solveNQueensUtil(col+1, n, board, result, row, diag1, diag2);
                board[i][col] = '.';
                row[i] = diag1[i+col] = diag2[i-col+n-1] = 0;
            }
        }
    }

public:
    // 162. Sudoku Solver
    void solveSudoku(vector<vector<char>>& board) {
        solveSudokuUtil(board);
    }
    
private:
    bool solveSudokuUtil(vector<vector<char>>& board) {
        int n = 9;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == '.') {
                    for (char c = '1'; c <= '9'; c++) {
                        if (isValidSudoku(board, i, j, c)) {
                            board[i][j] = c;
                            if (solveSudokuUtil(board)) return true;
                            board[i][j] = '.';
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }
    
    bool isValidSudoku(vector<vector<char>>& board, int row, int col, char c) {
        for (int i = 0; i < 9; i++) {
            if (board[row][i] == c || board[i][col] == c || 
                board[3*(row/3) + i/3][3*(col/3) + i%3] == c) {
                return false;
            }
        }
        return true;
    }

public:
    // 163. Combination Sum
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> result;
        vector<int> curr;
        sort(candidates.begin(), candidates.end());
        combinationSumUtil(candidates, target, 0, curr, result);
        return result;
    }
    
private:
    void combinationSumUtil(vector<int>& candidates, int target, int idx, 
                           vector<int>& curr, vector<vector<int>>& result) {
        if (target == 0) {
            result.push_back(curr);
            return;
        }
        for (int i = idx; i < candidates.size(); i++) {
            if (candidates[i] > target) break;
            curr.push_back(candidates[i]);
            combinationSumUtil(candidates, target - candidates[i], i, curr, result);
            curr.pop_back();
        }
    }

public:
    // 164. Permutations
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> result;
        permuteUtil(nums, 0, result);
        return result;
    }
    
private:
    void permuteUtil(vector<int>& nums, int idx, vector<vector<int>>& result) {
        if (idx == nums.size()) {
            result.push_back(nums);
            return;
        }
        for (int i = idx; i < nums.size(); i++) {
            swap(nums[idx], nums[i]);
            permuteUtil(nums, idx + 1, result);
            swap(nums[idx], nums[i]);
        }
    }

public:
    // 165. Subsets
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> result;
        vector<int> curr;
        subsetsUtil(nums, 0, curr, result);
        return result;
    }
    
private:
    void subsetsUtil(vector<int>& nums, int idx, vector<int>& curr, 
                    vector<vector<int>>& result) {
        result.push_back(curr);
        for (int i = idx; i < nums.size(); i++) {
            curr.push_back(nums[i]);
            subsetsUtil(nums, i + 1, curr, result);
            curr.pop_back();
        }
    }

public:
    // 166. Word Search
    bool exist(vector<vector<char>>& board, string word) {
        int m = board.size(), n = board[0].size();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (wordSearchUtil(board, word, 0, i, j)) return true;
            }
        }
        return false;
    }
    
private:
    bool wordSearchUtil(vector<vector<char>>& board, string& word, int idx, int i, int j) {
        if (idx == word.size()) return true;
        if (i < 0 || i >= board.size() || j < 0 || j >= board[0].size() || 
            board[i][j] != word[idx]) return false;
        
        char temp = board[i][j];
        board[i][j] = '#';
        bool found = wordSearchUtil(board, word, idx + 1, i + 1, j) ||
                     wordSearchUtil(board, word, idx + 1, i - 1, j) ||
                     wordSearchUtil(board, word, idx + 1, i, j + 1) ||
                     wordSearchUtil(board, word, idx + 1, i, j - 1);
        board[i][j] = temp;
        return found;
    }

public:
    // 167. Letter Combinations of Phone Number
    vector<string> letterCombinations(string digits) {
        vector<string> result;
        if (digits.empty()) return result;
        vector<string> mapping = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        string curr;
        letterCombinationsUtil(digits, 0, curr, mapping, result);
        return result;
    }
    
private:
    void letterCombinationsUtil(string& digits, int idx, string& curr, 
                              vector<string>& mapping, vector<string>& result) {
        if (idx == digits.size()) {
            result.push_back(curr);
            return;
        }
        string letters = mapping[digits[idx] - '0'];
        for (char c : letters) {
            curr.push_back(c);
            letterCombinationsUtil(digits, idx + 1, curr, mapping, result);
            curr.pop_back();
        }
    }

public:
    // 168. Generate Parentheses
    vector<string> generateParenthesis(int n) {
        vector<string> result;
        string curr;
        generateParenthesisUtil(n, 0, 0, curr, result);
        return result;
    }
    
private:
    void generateParenthesisUtil(int n, int open, int close, string& curr, 
                                vector<string>& result) {
        if (curr.size() == 2 * n) {
            result.push_back(curr);
            return;
        }
        if (open < n) {
            curr.push_back('(');
            generateParenthesisUtil(n, open + 1, close, curr, result);
            curr.pop_back();
        }
        if (close < open) {
            curr.push_back(')');
            generateParenthesisUtil(n, open, close + 1, curr, result);
            curr.pop_back();
        }
    }

public:
    // 169. Palindrome Partitioning
    vector<vector<string>> partition(string s) {
        vector<vector<string>> result;
        vector<string> curr;
        partitionUtil(s, 0, curr, result);
        return result;
    }
    
private:
    void partitionUtil(string& s, int idx, vector<string>& curr, 
                      vector<vector<string>>& result) {
        if (idx == s.size()) {
            result.push_back(curr);
            return;
        }
        for (int i = idx; i < s.size(); i++) {
            string sub = s.substr(idx, i - idx + 1);
            if (isPalindrome(sub)) {
                curr.push_back(sub);
                partitionUtil(s, i + 1, curr, result);
                curr.pop_back();
            }
        }
    }
    
    bool isPalindrome(string& s) {
        int left = 0, right = s.size() - 1;
        while (left < right) {
            if (s[left++] != s[right--]) return false;
        }
        return true;
    }

public:
    // 170. Regular Expression Matching (Backtracking)
    bool isMatch(string s, string p) {
        return isMatchUtil(s, p, 0, 0);
    }
    
private:
    bool isMatchUtil(string& s, string& p, int i, int j) {
        if (j == p.size()) return i == s.size();
        bool firstMatch = i < s.size() && (p[j] == s[i] || p[j] == '.');
        if (j + 1 < p.size() && p[j + 1] == '*') {
            return isMatchUtil(s, p, i, j + 2) || 
                   (firstMatch && isMatchUtil(s, p, i + 1, j));
        }
        return firstMatch && isMatchUtil(s, p, i + 1, j + 1);
    }

public:
    // 171. Combination Sum II
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<vector<int>> result;
        vector<int> curr;
        sort(candidates.begin(), candidates.end());
        combinationSum2Util(candidates, target, 0, curr, result);
        return result;
    }
    
private:
    void combinationSum2Util(vector<int>& candidates, int target, int idx, 
                            vector<int>& curr, vector<vector<int>>& result) {
        if (target == 0) {
            result.push_back(curr);
            return;
        }
        for (int i = idx; i < candidates.size(); i++) {
            if (i > idx && candidates[i] == candidates[i-1]) continue;
            if (candidates[i] > target) break;
            curr.push_back(candidates[i]);
            combinationSum2Util(candidates, target - candidates[i], i + 1, curr, result);
            curr.pop_back();
        }
    }

public:
    // 172. Permutations II
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> result;
        sort(nums.begin(), nums.end());
        permuteUniqueUtil(nums, 0, result);
        return result;
    }
    
private:
    void permuteUniqueUtil(vector<int>& nums, int idx, vector<vector<int>>& result) {
        if (idx == nums.size()) {
            result.push_back(nums);
            return;
        }
        unordered_set<int> used;
        for (int i = idx; i < nums.size(); i++) {
            if (used.find(nums[i]) != used.end()) continue;
            used.insert(nums[i]);
            swap(nums[idx], nums[i]);
            permuteUniqueUtil(nums, idx + 1, result);
            swap(nums[idx], nums[i]);
        }
    }

public:
    // 173. Subsets II
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vector<vector<int>> result;
        vector<int> curr;
        sort(nums.begin(), nums.end());
        subsetsWithDupUtil(nums, 0, curr, result);
        return result;
    }
    
private:
    void subsetsWithDupUtil(vector<int>& nums, int idx, vector<int>& curr, 
                           vector<vector<int>>& result) {
        result.push_back(curr);
        for (int i = idx; i < nums.size(); i++) {
            if (i > idx && nums[i] == nums[i-1]) continue;
            curr.push_back(nums[i]);
            subsetsWithDupUtil(nums, i + 1, curr, result);
            curr.pop_back();
        }
    }

public:
    // 174. Word Search II
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        vector<string> result;
        for (string& word : words) {
            if (exist(board, word)) {
                result.push_back(word);
            }
        }
        return result;
    }

public:
    // 175. N-Queens II
    int totalNQueens(int n) {
        vector<int> row(n, 0), diag1(2*n-1, 0), diag2(2*n-1, 0);
        return totalNQueensUtil(0, n, row, diag1, diag2);
    }
    
private:
    int totalNQueensUtil(int col, int n, vector<int>& row, vector<int>& diag1, 
                        vector<int>& diag2) {
        if (col == n) return 1;
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (!row[i] && !diag1[i+col] && !diag2[i-col+n-1]) {
                row[i] = diag1[i+col] = diag2[i-col+n-1] = 1;
                count += totalNQueensUtil(col + 1, n, row, diag1, diag2);
                row[i] = diag1[i+col] = diag2[i-col+n-1] = 0;
            }
        }
        return count;
    }

public:
    // 176. Partition to K Equal Sum Subsets
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if (sum % k != 0) return false;
        int target = sum / k;
        vector<bool> used(nums.size(), false);
        return canPartitionKSubsetsUtil(nums, k, 0, 0, target, used);
    }
    
private:
    bool canPartitionKSubsetsUtil(vector<int>& nums, int k, int start, int currSum, 
                                 int target, vector<bool>& used) {
        if (k == 0) return true;
        if (currSum == target) {
            return canPartitionKSubsetsUtil(nums, k - 1, 0, 0, target, used);
        }
        for (int i = start; i < nums.size(); i++) {
            if (!used[i] && currSum + nums[i] <= target) {
                used[i] = true;
                if (canPartitionKSubsetsUtil(nums, k, i + 1, currSum + nums[i], target, used)) {
                    return true;
                }
                used[i] = false;
            }
        }
        return false;
    }

public:
    // 177. Beautiful Arrangement
    int countArrangement(int n) {
        vector<int> nums(n);
        iota(nums.begin(), nums.end(), 1);
        return countArrangementUtil(nums, 0);
    }
    
private:
    int countArrangementUtil(vector<int>& nums, int idx) {
        if (idx == nums.size()) return 1;
        int count = 0;
        for (int i = idx; i < nums.size(); i++) {
            if ((nums[i] % (idx + 1) == 0) || ((idx + 1) % nums[i] == 0)) {
                swap(nums[idx], nums[i]);
                count += countArrangementUtil(nums, idx + 1);
                swap(nums[idx], nums[i]);
            }
        }
        return count;
    }

public:
    // 178. Path with Maximum Gold
    int getMaximumGold(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size(), maxGold = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] != 0) {
                    maxGold = max(maxGold, getMaximumGoldUtil(grid, i, j, m, n));
                }
            }
        }
        return maxGold;
    }
    
private:
    int getMaximumGoldUtil(vector<vector<int>>& grid, int i, int j, int m, int n) {
        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == 0) return 0;
        int temp = grid[i][j];
        grid[i][j] = 0;
        int maxGold = 0;
        int dirs[4][2] = {{0,1}, {1,0}, {0,-1}, {-1,0}};
        for (auto& dir : dirs) {
            maxGold = max(maxGold, getMaximumGoldUtil(grid, i + dir[0], j + dir[1], m, n));
        }
        grid[i][j] = temp;
        return maxGold + temp;
    }

public:
    // 179. Additive Number
    bool isAdditiveNumber(string num) {
        for (int i = 1; i <= num.size()/2; i++) {
            for (int j = 1; j <= (num.size()-i)/2; j++) {
                string first = num.substr(0, i);
                string second = num.substr(i, j);
                if ((first.size() > 1 && first[0] == '0') || 
                    (second.size() > 1 && second[0] == '0')) continue;
                if (isAdditiveNumberUtil(num, i + j, stoll(first), stoll(second))) {
                    return true;
                }
            }
        }
        return false;
    }
    
private:
    bool isAdditiveNumberUtil(string& num, int idx, long long first, long long second) {
        if (idx == num.size()) return true;
        long long sum = first + second;
        string sumStr = to_string(sum);
        if (idx + sumStr.size() > num.size() || 
            num.substr(idx, sumStr.size()) != sumStr) return false;
        return isAdditiveNumberUtil(num, idx + sumStr.size(), second, sum);
    }

public:
    // 180. Matchsticks to Square
    bool makesquare(vector<int>& matchsticks) {
        int sum = accumulate(matchsticks.begin(), matchsticks.end(), 0);
        if (sum % 4 != 0) return false;
        int side = sum / 4;
        sort(matchsticks.rbegin(), matchsticks.rend());
        if (matchsticks[0] > side) return false;
        vector<int> sides(4, 0);
        return makesquareUtil(matchsticks, 0, side, sides);
    }
    
private:
    bool makesquareUtil(vector<int>& matchsticks, int idx, int target, vector<int>& sides) {
        if (idx == matchsticks.size()) {
            return sides[0] == sides[1] && sides[1] == sides[2] && sides[2] == sides[3];
        }
        for (int i = 0; i < 4; i++) {
            if (sides[i] + matchsticks[idx] <= target) {
                sides[i] += matchsticks[idx];
                if (makesquareUtil(matchsticks, idx + 1, target, sides)) return true;
                sides[i] -= matchsticks[idx];
            }
            if (sides[i] == 0) break;
        }
        return false;
    }
};

int main() {
    Backtracking bt;
    // Example usage
    int n = 4;
    auto nQueens = bt.solveNQueens(n);
    cout << "N-Queens solutions for n=" << n << ": " << nQueens.size() << endl;

    vector<int> candidates = {2, 3, 6, 7};
    int target = 7;
    auto combSum = bt.combinationSum(candidates, target);
    cout << "Combination Sum for target=" << target << ": " << combSum.size() << endl;

    return 0;
}
