// Production adapter of 027 generator.cpp / 028 mode 8 (2026-10-22).
// Reuses their event-driven ready queues, receiver-frontier supply dates,
// 95% release priority, capacity counters and simultaneous-event draining.
// Graph/owners/stable IDs come from dspp_plan.py, including deployment FIFO,
// per-chunk W constraints and inactive-H relays. No new split decision here.
#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <time.h>
#include <utility>
#include <vector>

namespace {
using std::vector;
struct Node {
    int kind, rank, stable, relay, forward;
    double cost, priority, slack, reference_end, tokens;
    vector<int> succ, pred, born, users;
    vector<std::pair<int,int>> messages;
};
struct Owner { double bytes; vector<int> born, users; };
double monotonic_seconds() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
}

// Return 0: complete, 1: capacity deadlock, 2: deadline, -1: invalid payload.
extern "C" int dspp_construct(const char* data, int* output, int* sizes,
                               double deadline) {
    try {
        std::istringstream in(data);
        int n, pp, no, supply, feedback;
        in >> n >> pp >> no >> supply >> feedback;
        if (n <= 0 || pp <= 0 || no < 0) return -1;
        vector<Node> nodes(n); vector<Owner> owners(no);
        vector<double> cap(pp), limit(pp);
        for (int r=0;r<pp;r++) in >> cap[r] >> limit[r];
        auto list = [&](vector<int>& out) {
            int count; in >> count;
            if (!in || count < 0) throw std::runtime_error("invalid list");
            out.resize(count);
            for (int& v:out) { in >> v; if (v<0 || v>=n) throw std::runtime_error("invalid node"); }
        };
        for (int v=0;v<n;v++) {
            auto& x=nodes[v];
            in >> x.kind >> x.rank >> x.stable >> x.cost >> x.priority >> x.slack
               >> x.reference_end >> x.relay >> x.forward >> x.tokens;
            if (x.kind<0 || x.kind>5 || x.rank<0 || x.rank>=pp) return -1;
            list(x.succ);
            int nm; in >> nm;
            if (nm<0) return -1;
            for (int j=0;j<nm;j++) {
                int m,t; in >> m >> t;
                if (m<0 || m>=n || t<0 || t>=n) return -1;
                x.messages.emplace_back(m,t);
            }
        }
        for (int v=0;v<n;v++) for (int t:nodes[v].succ) nodes[t].pred.push_back(v);
        for (int i=0;i<no;i++) {
            in >> owners[i].bytes; list(owners[i].born); list(owners[i].users);
            for (int v:owners[i].born) nodes[v].born.push_back(i);
            for (int v:owners[i].users) nodes[v].users.push_back(i);
        }
        if (!in) return -1;
        vector<int> degree(n), remaining(no), allocated(no), known(n), busy(pp);
        vector<double> end(n), frontier(pp), drift(pp), live(pp), tlive(pp);
        vector<vector<int>> ready(pp), orders(pp);
        for (int v=0;v<n;v++) degree[v]=nodes[v].pred.size();
        for (int i=0;i<no;i++) remaining[i]=owners[i].users.size();
        // Same stable-ID event tie rule as the Python adapter.
        using Event=std::pair<double,std::pair<int,int>>;
        std::priority_queue<Event,vector<Event>,std::greater<Event>> events;
        double now=0.; int completed=0;
        auto activate = [&](int v) {
            auto& x=nodes[v];
            if (x.kind>=4) {
                known[v]=1; end[v]=now+(x.kind==5 ? 0. : x.cost);
                events.push({end[v],{x.stable,v}});
            } else ready[x.rank].push_back(v);
        };
        auto estimate_end = [&](int v) {
            if (known[v]) return end[v];
            int mapped=nodes[v].relay>=0 ? nodes[v].relay : v;
            return std::max(now,nodes[mapped].reference_end+drift[nodes[mapped].rank]);
        };
        for (int v=0;v<n;v++) if (!degree[v]) activate(v);
        while (completed<n) {
            if (deadline>0. && monotonic_seconds()>=deadline) return 2;
            for (int r=0;r<pp;r++) {
                if (busy[r]) continue;
                int best=-1; double best_date=0.; vector<int> best_alloc;
                for (int v:ready[r]) {
                    auto& x=nodes[v]; double extra=0.,freed=0.; vector<int> fresh;
                    for (int o:x.born) if (!allocated[o]) { fresh.push_back(o); extra+=owners[o].bytes; }
                    if (live[r]+extra>cap[r]+1e-8) continue;
                    if (x.kind==0 && tlive[r]+x.tokens>limit[r]+1e-8) continue;
                    for (int o:x.users) if (remaining[o]==1 && allocated[o]) freed+=owners[o].bytes;
                    double date=x.priority;
                    if (supply && (x.kind==0 || x.kind==1)) for (auto [m,t]:x.messages) {
                        double due=std::max(now,frontier[nodes[t].rank]);
                        for (int p:nodes[t].pred) if (p!=m) due=std::max(due,estimate_end(p));
                        date=std::min(date,due-nodes[m].cost-x.cost+x.slack);
                    }
                    if (freed && live[r]>.95*cap[r]) date=std::min(date,now-freed/std::max(1.,cap[r]));
                    if (best<0 || date<best_date-1e-8 ||
                        (std::abs(date-best_date)<1e-8 && x.stable<nodes[best].stable)) {
                        best=v; best_date=date; best_alloc=std::move(fresh);
                    }
                }
                if (best<0) continue;
                auto& x=nodes[best]; auto& queue=ready[r];
                queue.erase(std::find(queue.begin(),queue.end(),best));
                orders[r].push_back(best); busy[r]=1;
                for (int o:best_alloc) { allocated[o]=1; live[r]+=owners[o].bytes; }
                if (x.kind==0) tlive[r]+=x.tokens;
                known[best]=1; end[best]=frontier[r]=now+x.cost;
                events.push({end[best],{x.stable,best}});
            }
            if (events.empty()) return 1;
            now=events.top().first;
            while (!events.empty() && events.top().first<=now+1e-9) {
                int v=events.top().second.second; events.pop(); completed++;
                auto& x=nodes[v];
                if (x.kind<4) {
                    int r=x.rank; busy[r]=0;
                    if (!feedback || x.kind!=3) drift[r]=now-x.reference_end;
                    if (x.kind==1) tlive[r]-=nodes[x.forward].tokens;
                    for (int o:x.users) if (--remaining[o]==0 && allocated[o]) {
                        allocated[o]=0; live[r]-=owners[o].bytes;
                    }
                }
                for (int t:x.succ) if (--degree[t]==0) activate(t);
            }
        }
        int offset=0;
        for (int r=0;r<pp;r++) {
            sizes[r]=orders[r].size();
            for (int v:orders[r]) output[offset++]=v;
        }
        return 0;
    } catch (...) { return -1; }
}
