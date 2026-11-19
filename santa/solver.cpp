// Santa 2026 - Elf Calculator solver (v1)
// Greedy per-target search over the current register file:
//   exact search up to ~5 instructions via meet-in-the-middle,
//   then recursive "nearest reachable value + residual" fallback.
// Build: g++ -O2 -march=native -fopenmp -o solver solver.cpp
// Usage: ./solver puzzles.csv out.csv [first_n]
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64;
typedef __int128 i128;
typedef unsigned int u32;

static const int NREG = 48;
static int g_branch = 1, g_branch_depth = 0;

// ---------------------------------------------------------------- parsing
struct Puzzle {
    string id;
    vector<u64> start, targets;
    u64 limit;
};

static vector<u64> parse_list(const string& s) {
    vector<u64> v;
    u64 cur = 0;
    bool in = false;
    for (char c : s) {
        if (isdigit((unsigned char)c)) { cur = cur * 10 + (c - '0'); in = true; }
        else if (in) { v.push_back(cur); cur = 0; in = false; }
    }
    if (in) v.push_back(cur);
    return v;
}

static vector<Puzzle> load(const string& path) {
    ifstream f(path);
    string line;
    getline(f, line);  // header
    vector<Puzzle> out;
    while (getline(f, line)) {
        if (line.empty()) continue;
        // id,"[..]","[..]",limit,registers
        Puzzle p;
        size_t c1 = line.find(',');
        p.id = line.substr(0, c1);
        size_t b1 = line.find('[', c1), e1 = line.find(']', b1);
        p.start = parse_list(line.substr(b1, e1 - b1 + 1));
        size_t b2 = line.find('[', e1), e2 = line.find(']', b2);
        p.targets = parse_list(line.substr(b2, e2 - b2 + 1));
        size_t c2 = line.find(',', e2 + 1);
        size_t c3 = line.find(',', c2 + 1);
        p.limit = stoull(line.substr(c2 + 1, c3 - c2 - 1));
        out.push_back(p);
    }
    return out;
}

// ---------------------------------------------------------------- arithmetic
// op codes: 0 '+', 1 '-', 2 '*', 3 '/'
static inline bool apply(int op, u64 a, u64 b, u64 lim, u64& r) {
    switch (op) {
        case 0: r = a + b; return r <= lim && r >= a;
        case 1: if (a <= b) return false; r = a - b; return true;
        case 2: { i128 x = (i128)a * b; if (x > (i128)lim) return false; r = (u64)x; return true; }
        case 3: if (a % b) return false; r = a / b; return true;
    }
    return false;
}
static const char OPC[4] = {'+', '-', '*', '/'};

// ---------------------------------------------------------------- hash map u64 -> u64
struct HMap {
    vector<u64> keys, vals;
    u64 mask = 0;
    size_t n = 0;
    void init(size_t cap) {
        size_t sz = 16;
        while (sz < cap * 2) sz <<= 1;
        keys.assign(sz, 0); vals.assign(sz, 0); mask = sz - 1; n = 0;
    }
    static inline u64 h(u64 k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; return k; }
    // insert if absent; returns true if inserted
    inline bool put(u64 k, u64 v) {
        u64 i = h(k) & mask;
        while (keys[i]) { if (keys[i] == k) return false; i = (i + 1) & mask; }
        keys[i] = k; vals[i] = v; n++;
        return true;
    }
    inline const u64* get(u64 k) const {
        u64 i = h(k) & mask;
        while (keys[i]) { if (keys[i] == k) return &vals[i]; i = (i + 1) & mask; }
        return nullptr;
    }
};

// ---------------------------------------------------------------- expression DAG
// A plan is a small DAG of nodes; leaves reference values already in registers.
struct Node { u64 v; int op; int l, r; };  // op=-1 leaf
struct Plan {
    vector<Node> nodes;
    int leaf(u64 v) {
        for (int i = 0; i < (int)nodes.size(); i++) if (nodes[i].op == -1 && nodes[i].v == v) return i;
        nodes.push_back({v, -1, -1, -1});
        return nodes.size() - 1;
    }
    int mk(int op, int l, int r, u64 v) { nodes.push_back({v, op, l, r}); return nodes.size() - 1; }
};

// ---------------------------------------------------------------- solver state
struct State {
    u64 lim;
    int reg_owner_time[NREG];
    u64 regv[NREG];
    bool def[NREG];
    long long clock = 0;
    vector<array<int, 4>> prog;
    unordered_set<u64> need;

    int find_reg(u64 v) const {
        for (int i = 0; i < NREG; i++) if (def[i] && regv[i] == v) return i;
        return -1;
    }
    vector<u64> values() const {
        vector<u64> v;
        for (int i = 0; i < NREG; i++) if (def[i]) v.push_back(regv[i]);
        sort(v.begin(), v.end());
        v.erase(unique(v.begin(), v.end()), v.end());
        return v;
    }
};

// Level sets built from current values.
// S0: register values. S1: a op b. S2: s1 op s0 (or s0 op s1).
// Encodings for S1: (i<<16)|(j<<8)|op... use indices into S0.
struct Levels {
    vector<u64> S0;
    HMap s0;             // value -> idx
    vector<u64> S1v;     // values
    vector<u32> S1enc;   // i | j<<8 | op<<16
    HMap s1;             // value -> idx into S1v
    HMap s2;             // value -> enc: s1idx | s0idx<<24 | op<<32 | dir<<34
    vector<u64> S2sorted;
};

static void build_levels(const vector<u64>& S0, u64 lim, Levels& L, bool want_s2) {
    L.S0 = S0;
    int n = S0.size();
    L.s0.init(n + 1);
    for (int i = 0; i < n; i++) L.s0.put(S0[i], i);
    L.S1v.clear(); L.S1enc.clear();
    L.s1.init(4 * n * n + 1);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int op = 0; op < 4; op++) {
                if ((op == 0 || op == 2) && j < i) continue;
                u64 r;
                if (!apply(op, S0[i], S0[j], lim, r)) continue;
                if (L.s0.get(r)) continue;
                if (L.s1.put(r, L.S1v.size())) {
                    L.S1v.push_back(r);
                    L.S1enc.push_back(i | (j << 8) | (op << 16));
                }
            }
    if (!want_s2) return;
    size_t m = L.S1v.size();
    L.s2.init(m * n * 6 + 1);
    L.S2sorted.clear();
    for (size_t a = 0; a < m; a++) {
        u64 x = L.S1v[a];
        for (int j = 0; j < n; j++) {
            u64 y = S0[j];
            for (int op = 0; op < 4; op++) {
                for (int dir = 0; dir < 2; dir++) {
                    if (dir == 1 && (op == 0 || op == 2)) continue;
                    u64 r;
                    bool ok = dir == 0 ? apply(op, x, y, lim, r) : apply(op, y, x, lim, r);
                    if (!ok) continue;
                    if (L.s0.get(r) || L.s1.get(r)) continue;
                    if (L.s2.put(r, (u64)a | ((u64)j << 24) | ((u64)op << 32) | ((u64)dir << 34)))
                        L.S2sorted.push_back(r);
                }
            }
        }
    }
    for (size_t a = 0; a < m; a++) {
        u64 x = L.S1v[a];
        for (int op : {0, 2}) {
            u64 r;
            if (!apply(op, x, x, lim, r)) continue;
            if (L.s0.get(r) || L.s1.get(r)) continue;
            if (L.s2.put(r, (u64)a | ((u64)255 << 24) | ((u64)op << 32)))
                L.S2sorted.push_back(r);
        }
    }
    sort(L.S2sorted.begin(), L.S2sorted.end());
}

// add a value at level 0/1/2 into a plan, return node id
static int plan_value(const Levels& L, Plan& P, u64 v) {
    if (L.s0.get(v)) return P.leaf(v);
    if (auto e = L.s1.get(v)) {
        u32 enc = L.S1enc[*e];
        int i = enc & 255, j = (enc >> 8) & 255, op = enc >> 16;
        int a = P.leaf(L.S0[i]), b = P.leaf(L.S0[j]);
        return P.mk(op, a, b, v);
    }
    if (auto e = L.s2.get(v)) {
        u64 enc = *e;
        int a = enc & 0xFFFFFF, j = (enc >> 24) & 255, op = (enc >> 32) & 3, dir = (enc >> 34) & 1;
        int x = plan_value(L, P, L.S1v[a]);
        int y = j == 255 ? x : P.leaf(L.S0[j]);
        return dir == 0 ? P.mk(op, x, y, v) : P.mk(op, y, x, v);
    }
    return -1;
}
static int level_of(const Levels& L, u64 v) {
    if (L.s0.get(v)) return 0;
    if (L.s1.get(v)) return 1;
    if (L.s2.get(v)) return 2;
    return 99;
}

// For target t and op, list candidate partner pairs: t = x op y.
// Given y, the needed x: '+': t-y, '-': t+y (x-y=t), '*': t/y, '/': t*y.
// Also y op x forms: '-': y - x = t -> x = y - t; '/': y / x = t -> x = y / t.
static inline bool need_x(int op, int dir, u64 t, u64 y, u64 lim, u64& x) {
    if (dir == 0) {  // x op y = t
        switch (op) {
            case 0: if (y >= t) return false; x = t - y; return true;
            case 1: { if (t + y > lim || t + y < t) return false; x = t + y; return true; }
            case 2: if (y < 2 || t % y) return false; x = t / y; return true;
            case 3: { i128 z = (i128)t * y; if (y < 2 || z > (i128)lim) return false; x = (u64)z; return true; }
        }
    } else {  // y op x = t
        switch (op) {
            case 1: if (y <= t) return false; x = y - t; return true;
            case 3: if (y % t || y == t) return false; x = y / t; return true;
        }
    }
    return false;
}

// exact search: returns cost (instructions) and fills plan; cost limited to <=5.
static int exact(const Levels& L, u64 t, u64 lim, Plan& P, int maxc) {
    int lv = level_of(L, t);
    if (lv <= 2) { if (lv > maxc) return -1; plan_value(L, P, t); return lv; }
    // cost 3: x op y with x in S2, y in S0 -> covered by... (S2 op S0 is level3)
    //         or x,y in S1
    struct Cand { int cost; u64 x, y; int op, dir; };
    Cand best{99, 0, 0, 0, 0};
    auto consider = [&](u64 x, u64 y, int op, int dir, int cost) {
        if (cost < best.cost) best = {cost, x, y, op, dir};
    };
    if (maxc >= 3) {
        for (u64 y : L.S0)
            for (int op = 0; op < 4; op++)
                for (int dir = 0; dir < 2; dir++) {
                    u64 x;
                    if (need_x(op, dir, t, y, lim, x) && L.s2.get(x)) { consider(x, y, op, dir, 3); goto done3; }
                }
        for (u64 y : L.S1v)
            for (int op = 0; op < 4; op++)
                for (int dir = 0; dir < 2; dir++) {
                    u64 x;
                    if (need_x(op, dir, t, y, lim, x) && L.s1.get(x)) { consider(x, y, op, dir, 3); goto done3; }
                }
    }
done3:
    if (best.cost > 3 && maxc >= 4) {
        // cost 4: x in S2, y in S1
        for (u64 y : L.S1v) {
            for (int op = 0; op < 4; op++)
                for (int dir = 0; dir < 2; dir++) {
                    u64 x;
                    if (need_x(op, dir, t, y, lim, x) && L.s2.get(x)) { consider(x, y, op, dir, 4); goto done4; }
                }
        }
    }
done4:
    if (best.cost > 4 && maxc >= 5) {
        // cost 5: x in S2, y in S2
        for (u64 y : L.S2sorted) {
            for (int op = 0; op < 4; op++)
                for (int dir = 0; dir < 2; dir++) {
                    u64 x;
                    if (need_x(op, dir, t, y, lim, x) && L.s2.get(x)) { consider(x, y, op, dir, 5); goto done5; }
                }
        }
    }
done5:
    if (best.cost > maxc) return -1;
    int a = plan_value(L, P, best.x), b = plan_value(L, P, best.y);
    if (best.dir == 0) P.mk(best.op, a, b, t); else P.mk(best.op, b, a, t);
    // actual node count may be less due to sharing; count internal nodes
    int c = 0;
    for (auto& n : P.nodes) c += n.op >= 0;
    return c;
}

// ---------------------------------------------------------------- recursive fallback
static int count_ops(const Plan& P) { int c = 0; for (auto& n : P.nodes) c += n.op >= 0; return c; }
static int find_node(const Plan& P, u64 v) { for (int i = 0; i < (int)P.nodes.size(); i++) if (P.nodes[i].v == v) return i; return -1; }

static int est_cost(const Levels& L, u64 v) {
    int lv = level_of(L, v);
    if (lv <= 2) return lv;
    int bits = 64 - __builtin_clzll(v);
    return 3 + bits / 9;
}

static inline void nearest_sorted(const vector<u64>& S, u64 q, u64& lo, u64& hi) {
    lo = 0; hi = 0;
    auto it = upper_bound(S.begin(), S.end(), q);
    if (it != S.begin()) lo = *(it - 1);
    if (it != S.end()) hi = *it;
}

static bool solve_rec(const Levels& L, u64 t, u64 lim, Plan& P, int depth, bool top) {
    if (find_node(P, t) >= 0) return true;
    {
        Plan Q = P;
        if (exact(L, t, lim, Q, 5) >= 0) { P = Q; return true; }
    }
    // window search: t = X*Y +/- Z, Y in S0uS1, X in S2 (scan near t/Y), Z in S0uS1uS2
    {
        int bc = INT_MAX; u64 bX = 0, bY = 0, bZ = 0; int bs = 0;
        auto tryY = [&](u64 Y) {
            if (Y < 2) return;
            u64 q = t / Y;
            if (!q) return;
            int ly = level_of(L, Y);
            auto& S2 = L.S2sorted;
            auto it = lower_bound(S2.begin(), S2.end(), q);
            auto check = [&](u64 X) {
                i128 pr = (i128)X * Y;
                if (pr > (i128)lim) return;
                u64 p = (u64)pr;
                u64 Z; int sg;
                if (p == t) { Z = 0; sg = 0; }
                else if (p < t) { Z = t - p; sg = 1; }
                else { Z = p - t; sg = -1; }
                int lz = sg == 0 ? -1 : level_of(L, Z);
                if (lz > 2) return;
                int c = ly + level_of(L, X) + 1 + (sg ? lz + 1 : 0);
                if (c < bc) { bc = c; bX = X; bY = Y; bZ = Z; bs = sg; }
            };
            auto jt = it;
            for (int k = 0; k < 1024 && jt != S2.end(); k++, ++jt) check(*jt);
            jt = it;
            for (int k = 0; k < 1024 && jt != S2.begin(); k++) { --jt; check(*jt); }
        };
        for (u64 y : L.S0) tryY(y);
        for (u64 y : L.S1v) tryY(y);
        if (bc <= 7) {
            Plan Q = P;
            int ny = find_node(Q, bY); if (ny < 0) ny = plan_value(L, Q, bY);
            int nx = find_node(Q, bX); if (nx < 0) nx = plan_value(L, Q, bX);
            ny = find_node(Q, bY);
            u64 p = bX * bY;
            int np = find_node(Q, p); if (np < 0) np = Q.mk(2, nx, ny, p);
            if (bs) {
                int nz = find_node(Q, bZ); if (nz < 0) nz = plan_value(L, Q, bZ);
                np = find_node(Q, p);
                Q.mk(bs > 0 ? 0 : 1, np, nz, t);
            }
            P = Q;
            return true;
        }
    }
    if (depth > 200) return false;
    // candidates: kind 0 = approx (t = x*y +/- r, x,y from levels), kind 1 = horner (t = q*y +/- r, q recursive)
    struct C { int cost; int kind; u64 x, y, r; int sign; };
    vector<C> cands;
    auto consider = [&](u64 x, u64 y) {
        if (!x || !y) return;
        i128 pr = (i128)x * y;
        if (pr > (i128)lim) return;
        u64 p = (u64)pr;
        if (y > 1 && x == 1) return;
        if (p == t) { cands.push_back({level_of(L, x) + (y > 1 ? level_of(L, y) + 1 : 0), 0, x, y, 0, 0}); return; }
        int sign = p < t ? 1 : -1;
        u64 r = p < t ? t - p : p - t;
        if (r > t / 2) return;
        int c = level_of(L, x) + (y > 1 ? level_of(L, y) + 1 : 0) + 1 + est_cost(L, r);
        cands.push_back({c, 0, x, y, r, sign});
    };
    {
        u64 lo, hi;
        nearest_sorted(L.S2sorted, t, lo, hi);
        consider(lo, 1); consider(hi, 1);
        for (u64 x : L.S1v) consider(x, 1);
        for (u64 x : L.S0) consider(x, 1);
    }
    auto mult_with = [&](u64 y) {
        if (y < 2) return;
        u64 q = t / y;
        if (!q) return;
        u64 lo, hi;
        nearest_sorted(L.S2sorted, q, lo, hi);
        consider(lo, y); consider(hi, y);
        if (L.s1.get(q)) consider(q, y);
        if (L.s1.get(q + 1)) consider(q + 1, y);
        if (L.s0.get(q)) consider(q, y);
        if (L.s0.get(q + 1)) consider(q + 1, y);
    };
    for (u64 y : L.S0) mult_with(y);
    for (u64 y : L.S1v) mult_with(y);
    auto horner = [&](u64 y) {
        if (y < 2 || y > t) return;
        u64 q = t / y, r = t % y;
        cands.push_back({level_of(L, y) + est_cost(L, q) + 1 + (r ? 1 + est_cost(L, r) : 0), 1, q, y, r, 1});
        i128 pr = (i128)y * (q + 1);
        if (r && pr <= (i128)lim)
            cands.push_back({level_of(L, y) + est_cost(L, q + 1) + 2 + est_cost(L, (u64)(pr - t)), 1, q + 1, y, (u64)(pr - t), -1});
    };
    for (u64 y : L.S0) horner(y);
    for (u64 y : L.S1v) horner(y);
    for (u64 y : L.S2sorted) { if (y > t || y > 65536) break; horner(y); }
    if (cands.empty()) return false;
    int K = depth < g_branch_depth ? g_branch : 1;
    if ((int)cands.size() > K) {
        nth_element(cands.begin(), cands.begin() + K, cands.end(), [](const C& a, const C& b) { return a.cost < b.cost; });
        cands.resize(K);
    }
    sort(cands.begin(), cands.end(), [](const C& a, const C& b) { return a.cost < b.cost; });
    Plan bestP; int bestOps = INT_MAX;
    for (auto& c : cands) {
        Plan Q = P;
        bool ok = true;
        u64 p;
        if (c.kind == 0) {
            int nx = find_node(Q, c.x); if (nx < 0) nx = plan_value(L, Q, c.x);
            p = c.x;
            if (c.y > 1) {
                int ny = find_node(Q, c.y); if (ny < 0) ny = plan_value(L, Q, c.y);
                nx = find_node(Q, c.x);
                p = c.x * c.y;
                if (find_node(Q, p) < 0) Q.mk(2, nx, ny, p);
            }
        } else {
            if (!solve_rec(L, c.x, lim, Q, depth + 1, false)) { ok = false; }
            if (ok) {
                int ny = find_node(Q, c.y); if (ny < 0) ny = plan_value(L, Q, c.y);
                int nq = find_node(Q, c.x);
                p = c.x * c.y;
                if (find_node(Q, p) < 0) Q.mk(2, nq, ny, p);
            }
        }
        if (ok && c.r) {
            if (!solve_rec(L, c.r, lim, Q, depth + 1, false)) ok = false;
            else Q.mk(c.sign > 0 ? 0 : 1, find_node(Q, p), find_node(Q, c.r), t);
        }
        if (!ok) continue;
        int ops = count_ops(Q);
        if (ops < bestOps) { bestOps = ops; bestP = Q; }
    }
    if (bestOps == INT_MAX) return false;
    P = bestP;
    return true;
}

// ---------------------------------------------------------------- emission
static void emit(State& S, const Plan& P, const unordered_set<u64>& keep_hint) {
    int N = P.nodes.size();
    vector<int> reg_of(N, -1), last(N, -1);
    for (int i = 0; i < N; i++)
        if (P.nodes[i].op >= 0) { last[P.nodes[i].l] = max(last[P.nodes[i].l], i); last[P.nodes[i].r] = max(last[P.nodes[i].r], i); }
    vector<int> pin(NREG, 0);  // number of pending uses pinning a register
    for (int i = 0; i < N; i++)
        if (P.nodes[i].op == -1) {
            reg_of[i] = S.find_reg(P.nodes[i].v);
            if (reg_of[i] < 0) { fprintf(stderr, "leaf not in regs\n"); exit(1); }
        }
    auto pinned = [&](int k, int now) {
        for (int i = 0; i < N; i++) if (reg_of[i] == k && last[i] > now) return true;
        return false;
    };
    for (int i = 0; i < N; i++) {
        const Node& n = P.nodes[i];
        if (n.op < 0) continue;
        int ex = S.find_reg(n.v);
        if (ex >= 0) { reg_of[i] = ex; S.reg_owner_time[ex] = ++S.clock; continue; }
        int ra = reg_of[n.l], rb = reg_of[n.r];
        int d = -1;
        for (int k = 0; k < NREG; k++) if (!S.def[k]) { d = k; break; }
        if (d < 0) {
            long long bt = LLONG_MAX;
            for (int k = 0; k < NREG; k++) {
                if (k != ra && k != rb && pinned(k, i)) continue;
                if ((k == ra || k == rb) && pinned(k, i)) continue;
                long long tt = S.reg_owner_time[k];
                if (keep_hint.count(S.regv[k])) tt += (getenv("EVT") ? -1000000 : 1000000);
                if (tt < bt) { bt = tt; d = k; }
            }
        }
        if (d < 0) { fprintf(stderr, "out of registers\n"); exit(1); }
        // any node currently mapped to d is overwritten
        for (int j = 0; j < N; j++) if (reg_of[j] == d && j != n.l && j != n.r) reg_of[j] = -2;
        if (reg_of[n.l] == d && last[n.l] > i) { fprintf(stderr, "clobber\n"); exit(1); }
        S.prog.push_back({n.op, ra, rb, d});
        for (int j = 0; j < N; j++) if (reg_of[j] == d) reg_of[j] = -2;
        S.regv[d] = n.v; S.def[d] = true; S.reg_owner_time[d] = ++S.clock;
        S.need.erase(n.v);
        reg_of[i] = d;
    }
}

static vector<array<int, 4>> solve_puzzle(const Puzzle& pz) {
    State S;
    S.lim = pz.limit;
    for (int i = 0; i < NREG; i++) { S.def[i] = false; S.regv[i] = 0; S.reg_owner_time[i] = 0; }
    for (size_t i = 0; i < pz.start.size(); i++) { S.regv[i] = pz.start[i]; S.def[i] = true; }
    for (u64 t : pz.targets) S.need.insert(t);
    for (u64 v : pz.start) S.need.erase(v);
    unordered_set<u64> hint;
    for (u64 t : pz.targets) {
        if (!S.need.count(t)) continue;
        Levels L;
        build_levels(S.values(), pz.limit, L, true);
        Plan P;
        bool ok = solve_rec(L, t, pz.limit, P, 0, true);
        if (!ok) {
            fprintf(stderr, "%s: failed target %llu\n", pz.id.c_str(), t);
            exit(1);
        }
        size_t before = S.prog.size();
        emit(S, P, hint);
        if (getenv("KEEP") && string(getenv("KEEP")) == "targets") hint.insert(t);
        if (getenv("STATS")) fprintf(stderr, "STAT %s %d %zu %zu\n", pz.id.c_str(), 64 - __builtin_clzll(t), S.prog.size() - before, S.values().size());
        if (S.need.count(t)) { fprintf(stderr, "%s: target not produced %llu\n", pz.id.c_str(), t); exit(1); }
    }
    return S.prog;
}

static void analyze(const vector<Puzzle>& pzs) {
    // For each target: min cost (<=3, else 9) using start + all OTHER targets as free values.
    map<int,int> hist; long long tot=0, cnt=0;
    for (auto& pz : pzs) {
        for (u64 t : pz.targets) {
            vector<u64> S0 = pz.start;
            for (u64 u : pz.targets) if (u != t) S0.push_back(u);
            sort(S0.begin(), S0.end()); S0.erase(unique(S0.begin(), S0.end()), S0.end());
            S0.erase(remove(S0.begin(), S0.end(), t), S0.end());
            Levels L; build_levels(S0, pz.limit, L, false);
            int c = 9;
            if (L.s0.get(t)) c = 0; else if (L.s1.get(t)) c = 1;
            else {
                // cost2: x in S1, y in S0
                for (u64 y : S0) for (int op=0;op<4;op++) for (int dir=0;dir<2;dir++){u64 x; if(need_x(op,dir,t,y,pz.limit,x)&&L.s1.get(x)){c=2;goto d;}}
                for (u64 y : L.S1v) for (int op=0;op<4;op++) for (int dir=0;dir<2;dir++){u64 x; if(need_x(op,dir,t,y,pz.limit,x)&&L.s1.get(x)){c=3;goto d;}}
            }
            d: hist[c]++; tot+=c; cnt++;
        }
    }
    for (auto& [k,v]: hist) fprintf(stderr, "cost %d (9=more than 3): %d\n", k, v);
}

int main(int argc, char** argv) {
    auto pzs = load(argv[1]);
    if (getenv("BR")) g_branch = atoi(getenv("BR"));
    if (getenv("BRD")) g_branch_depth = atoi(getenv("BRD"));
    if (argc > 2 && string(argv[2]) == "analyze") { analyze(pzs); return 0; }
    if (argc > 2 && string(argv[2]) == "fromstart") {
        map<int,int> hist;
        for (auto& pz : pzs) {
            vector<u64> S0 = pz.start; sort(S0.begin(),S0.end()); S0.erase(unique(S0.begin(),S0.end()),S0.end());
            Levels L; build_levels(S0, pz.limit, L, true);
            for (u64 t : pz.targets) { Plan P; int c = exact(L, t, pz.limit, P, 5); hist[c]++; }
        }
        for (auto& [k,v]: hist) fprintf(stderr, "cost %d: %d\n", k, v);
        return 0;
    }
    int n = argc > 3 ? atoi(argv[3]) : pzs.size();
    pzs.resize(n);
    vector<vector<array<int, 4>>> res(n);
    atomic<int> done{0};
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n; i++) {
        res[i] = solve_puzzle(pzs[i]);
        int d = ++done;
        if (d % 50 == 0) fprintf(stderr, "%d/%d\n", d, n);
    }
    ofstream o(argv[2]);
    o << "id,program\n";
    long long tot = 0;
    for (int i = 0; i < n; i++) {
        o << pzs[i].id << ",\"[";
        for (size_t k = 0; k < res[i].size(); k++) {
            auto& in = res[i][k];
            if (k) o << ",";
            o << "[\"\"" << OPC[in[0]] << "\"\"," << in[1] << "," << in[2] << "," << in[3] << "]";
        }
        o << "]\"\n";
        tot += res[i].size();
    }
    fprintf(stderr, "total %lld\n", tot);
}
