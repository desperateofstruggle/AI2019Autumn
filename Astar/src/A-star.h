#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory.h>
#include <queue>
#include <vector>
#include <string>

//int h[21] = {0,366,0,160,242,161,178,77,151,226,244,241,234,380,98,193,253,329,80,199,374};
int h[21][21];

void initial(){
    h[1][2] = h[2][1] = 366;
    h[2][2] = h[2][2] = 0;
    h[3][2] = h[2][3] = 160;
    h[4][2] = h[2][4] = 242;
    h[5][2] = h[2][5] = 161;
    h[6][2] = h[2][6] = 178;
    h[7][2] = h[2][7] = 77;
    h[8][2] = h[2][8] = 151;
    h[9][2] = h[2][9] = 226;
    h[10][2] = h[2][10] = 244;
    h[11][2] = h[2][11] = 241;
    h[12][2] = h[2][12] = 234;
    h[13][2] = h[2][13] = 380;
    h[14][2] = h[2][14] = 98;
    h[15][2] = h[2][15] = 193;
    h[16][2] = h[2][16] = 253;
    h[17][2] = h[2][17] = 329;
    h[18][2] = h[2][18] = 80;
    h[19][2] = h[2][19] = 199;
    h[20][2] = h[2][20] = 374;
    for(int i = 1; i <= 20; i++){
        for(int j = 1; j <= 20; j++){
            if(i == 2 || j == 2)continue;
            h[i][j] = abs(h[i][2] - h[j][2]);
        }
    }
}

char Num2Name[][20]={"null", "Arad", "Bucharest", "Craiova", "Dobreta", "Eforie", "Fagaras", "Giurgiu", "Hirsova", "Iasi", "Lugoj", "Mehadia", "Neamt", "Oradea", "Pitesti", "Rimnicu Vilcea", "Sibiu", "Timisoara", "Urzeceni", "Vaslui", "Zerind"};

struct edge
{
  int nxt, to, w;
  edge(int x = 0, int y = 0, int z = 0) : nxt(x), to(y), w(z) {}
};


inline int read()
{
  int x = 0, f = 1;
  char ch;
  do
  {
    ch = getchar();
    if (ch == '-')
      f = -1;
  } while (!isdigit(ch));
  do
  {
    x = x * 10 + ch - '0';
    ch = getchar();
  } while (isdigit(ch));
  return f * x;
}

char output0[] = "-------------------------------------------------------------------------------------------------------------------------------------------------------------\n";
char output1[] = "This program is to generate the shortest path based on the A* algorithm!\n";
char output2[] = "First, you have to know that the first Essentials of this program is to deal with the problem over the graph offered in the assignment.pdf by the TAs.\n";
char output3[] = "So the English name in the result output is correspondent to that graph, \n -------------------------------------------------------------------------------------------------------------------------------------------------------------\n";
char output4[] = "\n\n************************************************************************************************************************************************\n\n";
char output5[] = "And if you want to generate more than 20 points, you must !!! must have to change the codes in the file A-star.h(line 10-41) to ensure the program will not show runtime error!!!!!!!\n\n************************************************************************************************************************************************";

void outputs(){
    std::cout << output0 << output1 << output2 << output3 << output4 << output5 << std::endl;
    std::cout << "Now, begin:" << std::endl;
}

