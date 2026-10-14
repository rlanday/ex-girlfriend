#ifndef R1_REORDER_TRANSFORM_H
#define R1_REORDER_TRANSFORM_H

#include <string>

namespace r1_reorder {

// Inherited from cmix-lex: applies the validated payload_lex permutation to
// regime 1 of the already dictionary-encoded cmix stream, after the word
// transform (on .main_phda9prepr it would change the output of the word
// transform).  This entry does not call it.
bool ReorderEncodedTailFile(const std::string& path,
    const std::string& side_path);

bool ExtractSideFromFile(const std::string& path, const std::string& side_path);

// Restores the post-WRT tail permutation after arithmetic decode and before
// dictionary decode.
bool RestoreEncodedTailFile(const std::string& path,
    const std::string& side_path);

}  // namespace r1_reorder

#endif
