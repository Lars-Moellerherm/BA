#include "RandomWalker.h"
#include "cmath"

void RandomWalker::run(){
  cout << "Hello" << endl;
}
void RandomWalker::init(){
  x = 0;
  y = 0;
  z = 0;
}
void RandomWalker::init(int a,int b,int c){
  x = a;
  y = b;
  z = c;
}
void RandomWalker::get_dist(){
  return sqrt(pow(x,2)+pow(y,2)+pow(z,2))
}
