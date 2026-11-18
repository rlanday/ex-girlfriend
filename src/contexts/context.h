#ifndef CONTEXT_H
#define CONTEXT_H

class Context {
 public:
  // No virtual member: every context lives by value in a typed SmallVector of
  // ContextManager and is never deleted through a Context*, and IsEqual()
  // compares two objects of the same class (typed argument), so the base
  // needs neither a vtable nor a class tag (BracketContext keeps a vtable,
  // see there).
  const unsigned long long& GetContext() const { return context_; }
  unsigned long long Size() const { return size_; }

 protected:
  void Update() {}
  unsigned long long context_, size_;
};

#endif

