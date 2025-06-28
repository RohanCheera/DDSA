// String Algorithms
class StringAlgorithms {
public:
    // 101. Z Algorithm
    vector<int> zAlgorithm(string s) {
        int n = s.size();
        vector<int> z(n, 0);
        int l = 0, r = 0;
        for (int i = 1; i < n; i++) {
            if (i <= r) {
                z[i] = min(r - i + 1, z[i - l]);
            }
            while (i + z[i] < n && s[z[i]] == s[i + z[i]]) {
                z[i]++;
            }
            if (i + z[i] - 1 > r) {
                l = i;
                r = i + z[i] - 1;
            }
        }
        return z;
    }

    // 102. Suffix Array
    vector<int> suffixArray(string s) {
        s += '$';
        int n = s.size();
        vector<int> sa(n), rank(n), tmp(n);
        for (int i = 0; i < n; i++) {
            sa[i] = i;
            rank[i] = s[i];
        }
        for (int k = 1; k < n; k *= 2) {
            auto cmp = [&](int i, int j) {
                if (rank[i] != rank[j]) return rank[i] < rank[j];
                int ri = i + k < n ? rank[i + k] : -1;
                int rj = j + k < n ? rank[j + k] : -1;
                return ri < rj;
            };
            sort(sa.begin(), sa.end(), cmp);
            tmp[sa[0]] = 0;
            for (int i = 1; i < n; i++) {
                tmp[sa[i]] = tmp[sa[i-1]] + (cmp(sa[i-1], sa[i]) ? 1 : 0);
            }
            rank = tmp;
        }
        return vector<int>(sa.begin() + 1, sa.end());
    }

    // 103. Longest Common Prefix (Trie)
    struct TrieNode {
        TrieNode* children[26];
        bool isEnd;
        TrieNode() : isEnd(false) {
            fill(children, children + 26, nullptr);
        }
    };

    string longestCommonPrefixTrie(vector<string>& strs) {
        if (strs.empty()) return "";
        TrieNode* root = new TrieNode();
        for (string& s : strs) {
            TrieNode* curr = root;
            for (char c : s) {
                int idx = c - 'a';
                if (!curr->children[idx]) curr->children[idx] = new TrieNode();
                curr = curr->children[idx];
            }
            curr->isEnd = true;
        }
        string result;
        TrieNode* curr = root;
        while (curr) {
            int count = 0, idx = -1;
            for (int i = 0; i < 26; i++) {
                if (curr->children[i]) {
                    count++;
                    idx = i;
                }
            }
            if (count != 1 || curr->isEnd) break;
            result += (char)('a' + idx);
            curr = curr->children[idx];
        }
        return result;
    }

    // 104. Longest Common Prefix (Divide-Conquer)
    string longestCommonPrefixDC(vector<string>& strs, int l, int r) {
        if (l == r) return strs[l];
        if (l < r) {
            int mid = (l + r) / 2;
            string left = longestCommonPrefixDC(strs, l, mid);
            string right = longestCommonPrefixDC(strs, mid + 1, r);
            int minLen = min(left.size(), right.size());
            string result;
            for (int i = 0; i < minLen; i++) {
                if (left[i] != right[i]) break;
                result += left[i];
            }
            return result;
        }
        return "";
    }

    // 105. Valid Palindrome
    bool isValidPalindrome(string s) {
        string filtered;
        for (char c : s) {
            if (isalnum(c)) filtered += tolower(c);
        }
        int left = 0, right = filtered.size() - 1;
        while (left < right) {
            if (filtered[left++] != filtered[right--]) return false;
        }
        return true;
    }

    // 106. Group Anagrams
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> mp;
        for (string& s : strs) {
            string key = s;
            sort(key.begin(), key.end());
            mp[key].push_back(s);
        }
        vector<vector<string>> result;
        for (auto& p : mp) {
            result.push_back(p.second);
        }
        return result;
    }

    // 107. Minimum Window Substring
    string minWindowSubstring(string s, string t) {
        unordered_map<char, int> mp;
        for (char c : t) mp[c]++;
        int required = mp.size(), formed = 0, left = 0, minLen = INT_MAX, minLeft = 0;
        unordered_map<char, int> window;
        for (int right = 0; right < s.size(); right++) {
            window[s[right]]++;
            if (mp.count(s[right]) && window[s[right]] == mp[s[right]]) {
                formed++;
            }
            while (left <= right && formed == required) {
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    minLeft = left;
                }
                window[s[left]]--;
                if (mp.count(s[left]) && window[s[left]] < mp[s[left]]) {
                    formed--;
                }
                left++;
            }
        }
        return minLen == INT_MAX ? "" : s.substr(minLeft, minLen);
    }

    // 108. Longest Repeating Character Replacement
    int longestRepeatingCharReplacement(string s, int k) {
        vector<int> count(26, 0);
        int maxCount = 0, maxLen = 0, left = 0;
        for (int right = 0; right < s.size(); right++) {
            maxCount = max(maxCount, ++count[s[right] - 'A']);
            while (right - left + 1 - maxCount > k) {
                count[s[left] - 'A']--;
                left++;
            }
            maxLen = max(maxLen, right - left + 1);
        }
        return maxLen;
    }

    // 109. String to Integer (atoi)
    int stringToInteger(string s) {
        int i = 0, n = s.size(), sign = 1;
        long result = 0;
        while (i < n && s[i] == ' ') i++;
        if (i < n && (s[i] == '+' || s[i] == '-')) {
            sign = s[i] == '+' ? 1 : -1;
            i++;
        }
        while (i < n && isdigit(s[i])) {
            result = result * 10 + (s[i] - '0');
            if (result * sign > INT_MAX) return INT_MAX;
            if (result * sign < INT_MIN) return INT_MIN;
            i++;
        }
        return result * sign;
    }

    // 110. Text Justification
    vector<string> textJustification(vector<string>& words, int maxWidth) {
        vector<string> result;
        int i = 0, n = words.size();
        while (i < n) {
            int count = words[i].size(), last = i + 1;
            while (last < n && count + words[last].size() + (last - i) <= maxWidth) {
                count += words[last].size();
                last++;
            }
            string line;
            int gaps = last - i - 1;
            if (last == n || gaps == 0) {
                for (int j = i; j < last; j++) {
                    line += words[j];
                    if (j < last - 1) line += " ";
                }
                line += string(maxWidth - line.size(), ' ');
            } else {
                int spaces = (maxWidth - count) / gaps;
                int extra = (maxWidth - count) % gaps;
                for (int j = i; j < last; j++) {
                    line += words[j];
                    if (j < last - 1) {
                        line += string(spaces + (j - i < extra ? 1 : 0), ' ');
                    }
                }
            }
            result.push_back(line);
            i = last;
        }
        return result;
    }

    // 111. Longest Valid Parentheses (Stack)
    int longestValidParenthesesStack(string s) {
        stack<int> st;
        st.push(-1);
        int maxLen = 0;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(') {
                st.push(i);
            } else {
                st.pop();
                if (st.empty()) {
                    st.push(i);
                } else {
                    maxLen = max(maxLen, i - st.top());
                }
            }
        }
        return maxLen;
    }

    // 112. Distinct Subsequences
    int distinctSubsequences(string s, string t) {
        int n = s.size(), m = t.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
        for (int i = 0; i <= n; i++) dp[i][0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                dp[i][j] = dp[i-1][j];
                if (s[i-1] == t[j-1]) {
                    dp[i][j] += dp[i-1][j-1];
                }
            }
        }
        return dp[n][m];
    }

    // 113. Minimum Add to Make Parentheses Valid
    int minAddToMakeValid(string s) {
        int open = 0, need = 0;
        for (char c : s) {
            if (c == '(') {
                open++;
            } else {
                if (open == 0) need++;
                else open--;
            }
        }
        return open + need;
    }

    // 114. Shortest Palindrome
    string shortestPalindrome(string s) {
        string rev = s;
        reverse(rev.begin(), rev.end());
        string combined = s + "#" + rev;
        vector<int> kmp = zAlgorithm(combined);
        int i = kmp[combined.size() - 1];
        return rev.substr(0, s.size() - i) + s;
    }

    // 115. Valid Anagram
    bool isValidAnagram(string s, string t) {
        if (s.size() != t.size()) return false;
        vector<int> count(26, 0);
        for (int i = 0; i < s.size(); i++) {
            count[s[i] - 'a']++;
            count[t[i] - 'a']--;
        }
        for (int c : count) {
            if (c != 0) return false;
        }
        return true;
    }

    // 116. Find All Anagrams in String
    vector<int> findAnagrams(string s, string p) {
        vector<int> result;
        if (s.size() < p.size()) return result;
        vector<int> pCount(26, 0), sCount(26, 0);
        for (char c : p) pCount[c - 'a']++;
        for (int i = 0; i < p.size(); i++) sCount[s[i] - 'a']++;
        if (pCount == sCount) result.push_back(0);
        for (int i = p.size(); i < s.size(); i++) {
            sCount[s[i] - 'a']++;
            sCount[s[i - p.size()] - 'a']--;
            if (pCount == sCount) result.push_back(i - p.size() + 1);
        }
        return result;
    }

    // 117. Longest Substring Without Repeating
    int longestSubstringWithoutRepeating(string s) {
        unordered_map<char, int> mp;
        int maxLen = 0, left = 0;
        for (int right = 0; right < s.size(); right++) {
            if (mp.count(s[right]) && mp[s[right]] >= left) {
                left = mp[s[right]] + 1;
            }
            mp[s[right]] = right;
            maxLen = max(maxLen, right - left + 1);
        }
        return maxLen;
    }

    // 118. Longest Palindromic Substring (DP)
    string longestPalindromicSubstringDP(string s) {
        int n = s.size(), start = 0, maxLen = 1;
        vector<vector<bool>> dp(n, vector<bool>(n, false));
        for (int i = 0; i < n; i++) dp[i][i] = true;
        for (int i = 0; i < n - 1; i++) {
            if (s[i] == s[i + 1]) {
                dp[i][i + 1] = true;
                start = i;
                maxLen = 2;
            }
        }
        for (int len = 3; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                if (s[i] == s[j] && dp[i + 1][j - 1]) {
                    dp[i][j] = true;
                    if (len > maxLen) {
                        start = i;
                        maxLen = len;
                    }
                }
            }
        }
        return s.substr(start, maxLen);
    }

    // 119. String Compression
    int stringCompression(vector<char>& chars) {
        int write = 0, count = 1;
        for (int read = 1; read <= chars.size(); read++) {
            if (read == chars.size() || chars[read] != chars[read-1]) {
                chars[write++] = chars[read-1];
                if (count > 1) {
                    for (char c : to_string(count)) {
                        chars[write++] = c;
                    }
                }
                count = 1;
            } else {
                count++;
            }
        }
        return write;
    }

    // 120. Rolling Hash
    vector<int> rollingHash(string s, string pattern) {
        vector<int> result;
        long p = 31, mod = 1e9 + 9, patternHash = 0, currHash = 0, powP = 1;
        for (int i = 0; i < pattern.size(); i++) {
            patternHash = (patternHash * p + pattern[i]) % mod;
            currHash = (currHash * p + s[i]) % mod;
            if (i > 0) powP = (powP * p) % mod;
        }
        if (patternHash == currHash && s.substr(0, pattern.size()) == pattern) {
            result.push_back(0);
        }
        for (int i = pattern.size(); i < s.size(); i++) {
            currHash = (currHash * p + s[i] - powP * s[i - pattern.size()] % mod + mod) % mod;
            if (currHash == patternHash && s.substr(i - pattern.size() + 1, pattern.size()) == pattern) {
                result.push_back(i - pattern.size() + 1);
            }
        }
        return result;
    }
};

