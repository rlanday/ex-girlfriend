#ifndef DICT_DERIVE_H
#define DICT_DERIVE_H

namespace dict_derive {

bool BuildDictFromInput(const char* enwik9_path, const char* hints_path,
                        const char* out_path);
}  // namespace dict_derive

#endif  // DICT_DERIVE_H
