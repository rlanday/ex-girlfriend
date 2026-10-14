#ifndef ALIAS_TITLE_SUBST_H_
#define ALIAS_TITLE_SUBST_H_

namespace alias_title_subst {

inline unsigned long long g_seiten = 0;        // title lines in total
inline unsigned long long g_ersetzt = 0;
inline unsigned long long g_titel_bytes = 0;   // bytes of the replaced titles
inline unsigned long long g_ersatz_bytes = 0;  // bytes of the replacement rows
inline unsigned long long g_block_gespart = 0;
inline int g_rest = 0;
inline void Beginn(int n) { g_rest = n; }
inline void Ende() { g_rest = 0; }
inline bool Laeuft() { return g_rest > 0; }
inline void Verbrauche() { if (g_rest > 0) --g_rest; }
}

#endif
