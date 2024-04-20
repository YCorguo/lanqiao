# Competitive Programming

# AtCoder
## ABC350
### A
#### 题意
给你包含6个字符的字符串，前三个字符是ABC，后三个字符是数字，问是否在ABC001, ABC002, ..., ABC314, ABC315, ABC317, ABC318, ..., ABC349中。若是，输出Yes，否则输出No。
#### 样例输入1
```
ABC349
```
#### 样例输出1
```
Yes
```
#### 样例输入2
```
ABC350
```
#### 样例输出2
```
No
```
#### 样例输入3
```
ABC316
```
#### 样例输出3
```
No
```
#### 思路
判断。
#### c++代码
```c++
#include <bits/stdc++.h>
int main()
{
    std::string s;
    std::cin >> s;
    int num = std::stoi(s.substr(3));
    if (1 <= num && num <= 349 && num != 316) std::cout << "Yes";
    else std::cout << "No";
}
```
#### python代码
```python
num = int(input()[3:6])
if 1 <= num <= 349 and num != 316: print('Yes')
else: print('No')
```

### B
#### 题意
给定N颗牙，Q次治疗，每次针对一颗牙展开治疗，如果有牙就敲掉，否则就补上，问最后有几只牙。
#### 样例输入1
```
30 6
2 9 18 27 18 9
```
#### 样例输出1
```
28
```
#### 样例输入2
```
1 7
1 1 1 1 1 1 1
```
#### 样例输出2
```
0
```
#### 样例输入3
```
9 20
9 5 1 2 2 2 8 9 2 1 6 2 6 5 8 7 8 5 9 8
```
#### 样例输出3
```
5
```
#### 思路
异或。
#### c++代码
```c++
#include <bits/stdc++.h>
int main()
{
    int n, q; std::cin >> n >> q;
    std::vector<int> st(n, 1);
    while (q --) {
        int x;
        std::cin >> x;
        st[x-1] ^= 1;
    }
    std::cout << std::accumulate(st.begin(), st.end(), 0);
}
```
#### python代码
```python
n, q = map(int, input().split())
st = [1] * n
for x in list(map(int, input().split())):
    st[int(x)-1] ^= 1
print(sum(st))
```

### C
#### 题意
给定N和长为N的数组a，是1-N的一个排列，要求通过不超过N-1次交换使之升序，输出交换过程。
#### 样例输入1
```
5
3 4 1 2 5
```
#### 样例输出1
```
2
1 3
2 4
```
#### 样例输入2
```
4
1 2 3 4
```
#### 样例输出2
```
0
```
#### 样例输入3
```
3
3 1 2
```
#### 样例输出3
```
2
1 2
2 3
```
#### 思路
哈希。
#### c++代码
```c++
#include <bits/stdc++.h>
int main()
{
    int n; std::cin >> n;
    std::vector<int> a(n + 1), p(n + 1);
    std::vector<std::pair<int, int>> res;
    for (int i = 1; i <= n; i ++) std::cin >> a[i], p[a[i]] = i;
    for (int i = 1; i <= n; i ++)
        if (i != a[i]) {
            int u = p[i];
            p[a[i]] = u;
            std::swap(a[i], a[u]);
            res.push_back({i, u});
        }
    std::cout << res.size() << '\n';
    for (auto & [x, y] : res) std::cout << x << ' ' << y << std::endl;
}
```
#### python代码
```python
n = int(input())
a = [0] + list(map(int, input().split()))
p = {a[i] : i for i in range(1, n + 1)}
res = []
for i in range(1, n + 1):
    if a[i] != i:
        u = p[i]
        res.append([i, u])
        p[a[i]], a[i], a[u] = u, a[u], a[i]
print(len(res))
for r in res:
    print(*r)
```

### D
#### 题意
给定N为点数，M为边数，以及M条边，问把每个子块连通还需要加几条边。
#### 样例输入1
```
5
3 4 1 2 5
```
#### 样例输出1
```
2
1 3
2 4
```
#### 样例输入2
```
4
1 2 3 4
```
#### 样例输出2
```
0
```
#### 样例输入3
```
3
3 1 2
```
#### 样例输出3
```
2
1 2
2 3
```
#### 思路
并查集。
#### c++代码
```c++
#include <bits/stdc++.h>
int main()
{
    int n, m; std::cin >> n >> m;
    std::vector<int> f(n), num(n, 1);
    std::function<int(int)> find = [&](int x) { return f[x] == x ? x : f[x] = find(f[x]); };
    for(int i = 0; i < n; i++) f[i] = i;
    for (int i = 0; i < m; i ++) {
        int u, v; std::cin >> u >> v;
        int pu = find(u-1), pv = find(v-1);
        if (pu != pv) {
            f[pu] = pv;
            num[pv] += num[pu];
        }
    }
    long long res = 0;
    for (int i = 0; i < n; i ++)
        if (f[i] == i) res += (long long)num[i] * (num[i] - 1) / 2;
    std::cout << res - m;
}
```
#### python代码
```python
n, m = map(int, input().split())
f, num = [i for i in range(n)], [1] * n
def find(x):
    if f[x] != x: f[x] = find(f[x])
    return f[x]
for _ in range(m):
    u, v = map(int, input().split())
    pu, pv = find(u-1), find(v-1)
    if pu != pv:
        f[pu] = pv
        num[pv] += num[pu]
res = 0
for i in range(n):
    if i == f[i]:
        res += num[i] * (num[i]-1) // 2
print(res - m)
```
