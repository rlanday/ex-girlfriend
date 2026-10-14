#ifndef CONTEXT_H
#define CONTEXT_H

class Context {
 public:
  virtual ~Context() {}
  // Class tag: IsEqual compares the dynamic class through it instead of
  // dynamic_cast, so the program builds without RTTI (no type_info objects
  // that would tie the binary to the libstdc++ it was linked against).
  enum KindId { kOther = 0, kContextHash, kIndirectHash, kBracketContext, kSparse, kCombinedContext, kIntervalHash };
  virtual int Kind() const { return kOther; }
  template <class T> static T* As(Context* c) {
    return (c && c->Kind() == T::kKind) ? static_cast<T*>(c) : nullptr;
  }
  bool IsEqual(Context* c) { return false; }
  const unsigned long long& GetContext() const { return context_; }
  unsigned long long Size() const { return size_; }

 protected:
  void Update() {}
  unsigned long long context_, size_;
};

#endif

