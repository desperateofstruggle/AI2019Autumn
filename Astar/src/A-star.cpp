#include "A-star.h"
using namespace std;
const int maxv = 1010;
const int maxe = 1e5 + 10;

int tot = 1, head[maxv], _tot = 1, _head[maxv];

edge e[maxe], _e[maxe];

inline void add_edge(int from, int to, int w)
{
  e[++tot] = edge(head[from], to, w), head[from] = tot;
  e[++tot] = edge(head[to], from, w), head[to] = tot;
  _e[++_tot] = edge(_head[to], from, w), _head[to] = _tot;
  _e[++_tot] = edge(_head[from], to, w), _head[from] = _tot;
}

int n, m, st, ed, d[maxv], appear_times[maxv], choice;
bool vis[maxv];

typedef pair<int, int> P;

void dij()
{
  priority_queue<P> q;
  memset(d, 0x3f, sizeof(d));
  d[ed] = 0;
  q.push(make_pair(0, ed));
  while (q.size())
  {
    int u = q.top().second;
    q.pop();
    if (vis[u])
      continue;
    vis[u] = 1;
    for (int i = _head[u]; i; i = _e[i].nxt)
    {
      int v = _e[i].to, w = _e[i].w;
      if (d[v] > d[u] + w)
      {
        d[v] = d[u] + w;
        q.push(make_pair(-d[v], v));
      }
    }
  }
}

void data_reads()
{
  printf("Please enter the number of the points(cities) and the numbers of the edges(roads):\n");
  n = read(), m = read();
  printf("Please enter the number of the edge information such as shown followed:  1 2 100  that means from 1 to 2 it is 100 far away:\n");
  for (int i = 1; i <= m; i++)
  {
    int x = read(), y = read(), z = read();
    add_edge(x, y, z);
  }
  printf("please enter the start points and the end points:");
  st = read(), ed = read();
  printf("please enter the methods you choosed(1 --- dij as the h*,  2 --- the straight distance as the h*):");
  choice = read();
  choice = (choice == 1)? 1 : 2;
  dij();
}

struct node
{
  int idx, now, fur;
  string tra = "";
  node(int x = 0, int y = 0, int z = 0, string str = "") : idx(x), now(y), fur(z) , tra(str) {}
  friend bool operator<(const node &a, const node &b)
  {
    return b.now + b.fur < a.now + a.fur;
  }
};

int solve()
{
  priority_queue<node> q;
  q.push(node(st, 0, d[st]));
  while(q.size()) {
    node u =q.top();
    q.pop();
    appear_times[u.idx]++;
    if(u.idx == ed){
      cout << u.idx << endl;
      return u.now;
    }
    if(appear_times[u.idx] > 1)continue;
    cout << u.idx << "->";
    for(int i=head[u.idx]; i; i = e[i].nxt) {
      int v =e[i].to, w = e[i].w;
      q.push(node(v, u.now + w, d[v]));
    }
  }
  return -1;
}

int solve1() {
  priority_queue<node> q;
  q.push(node(st, 0, d[st], string(Num2Name[st]) + "(" + to_string(st) + ")"));
  initial();
  while(q.size()){
    node u =q.top();
    q.pop();
    appear_times[u.idx]++;
    if(u.idx == ed){
      cout << "The routine is:";
      cout << u.tra << endl;
      return u.now;
    }
    if(appear_times[u.idx] > 1)continue;
      
    for(int i=head[u.idx]; i; i = e[i].nxt)
    {
      int v =e[i].to, w = e[i].w;
      q.push(node(v, u.now + w, h[v][ed], u.tra + "->" + string(Num2Name[v]) + "(" + to_string(v) + ")"));
    }
  }
  return -1;
}

int main()
{
  outputs();
  data_reads();
  if(choice == 1)
    printf("Cost:%d\n", solve());
  else
    printf("Cost:%d\n", solve1());

  return 0;
}