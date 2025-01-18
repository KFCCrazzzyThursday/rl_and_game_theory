#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

const int ROWS = 8;
const int COLS = 8;
const double THETA = 1e-4; // threshold

char maze[ROWS][COLS] = {
    {'*', '*', '*', '*', '*', '*', '*', '*'},
    {'*', '0', '0', '0', '0', '0', '0', '*'},
    {'S', '0', '*', '*', '0', '*', '0', '*'},
    {'*', '0', '0', '*', '*', '0', '0', '*'},
    {'*', '*', '0', '0', '*', '0', '*', '*'},
    {'*', '0', '*', '0', '*', '0', '0', '*'},
    {'*', '0', '0', '0', '0', '*', '0', 'E'},
    {'*', '*', '*', '*', '*', '*', '*', '*'}
};

// MDP
std::vector<std::pair<int, int>> states; // S
// location / get S
int getStateIndex(int x, int y)
{
    for (size_t i = 0; i < states.size(); ++i) {
        if (states[i].first == x && states[i].second == y) {
            return i;
        }
    }
    return -1;
}
// A
int actions[4][2] = {
    {-1, 0}, // N
    {0, 1},  // E
    {1, 0},  // S
    {0, -1}  // W
};
// R
double reward = -1.0;
// gamma
const double GAMMA = 1.0; 

// helper func
bool isValid(int x, int y)
{
    return x >= 0 && x < ROWS && y >= 0 && y < COLS && maze[x][y] != '*';
}


int main()
{
    int startX, startY, endX, endY;
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            if (maze[i][j] == '0' || maze[i][j] == 'S' || maze[i][j] == 'E') {
                states.push_back({i, j});
                if (maze[i][j] == 'S') {
                    startX = i;
                    startY = j;
                }
                if (maze[i][j] == 'E') {
                    endX = i;
                    endY = j;
                }
            }
        }
    }

    int numStates = states.size();
    int numActions = 4;

    std::vector<double> V(numStates, 0.0);
    std::vector<int> policy(numStates, 0); 
    bool isValueStable = false;

    while (!isValueStable) {
        double delta = 0.0;
        for (int s = 0; s < numStates; ++s) {
            int x = states[s].first;
            int y = states[s].second;

            if (x == endX && y == endY) {
                continue;
            }

            double v = V[s];
            double maxActionValue = -std::numeric_limits<double>::infinity();

            for (int a = 0; a < numActions; ++a) {
                int newX = x + actions[a][0];
                int newY = y + actions[a][1];

                if (!isValid(newX, newY)) {
                    newX = x;
                    newY = y;
                }

                int sPrime = getStateIndex(newX, newY);
                double actionValue = reward + GAMMA * V[sPrime];

                if (actionValue > maxActionValue) {
                    maxActionValue = actionValue;
                    policy[s] = a;
                }
            }
            V[s] = maxActionValue;
            delta = std::max(delta, std::abs(v - V[s]));
        }

        if (delta < THETA) {
            isValueStable = true;
        }
    }

    std::cout << "Optimal Policy: " << std::endl;
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            if (maze[i][j] == '*') {
                std::cout << "* ";
            } else if (maze[i][j] == 'E') {
                std::cout << "E ";
            } else {
                int s = getStateIndex(i, j);
                if (s == -1 || (i == endX && j == endY)) {
                    std::cout << "  ";
                } else {
                    char dir;
                    switch (policy[s]) {
                    case 0:
                        dir = 'N';
                        break;
                    case 1:
                        dir = 'E';
                        break;
                    case 2:
                        dir = 'S';
                        break;
                    case 3:
                        dir = 'W';
                        break;
                    }
                    std::cout << dir << " ";
                }
            }
        }
        std::cout << std::endl;
    }

    return 0;
}
