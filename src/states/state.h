#ifndef STATE_H
#define STATE_H

class State {
 public:
  // No virtual destructor: the states are members, never deleted through a
  // State*.
  float InitProbability(int state) const { return 0.5; }
  int Next(int state, int bit) const { return 0; }
};

#endif

