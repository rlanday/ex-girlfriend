# Error codes

A message prints its code `E<n>`; a few messages in hot code also print the values of their conversions (`%s`, `%d`, ...) in the order of the text.

| code | message |
|---|---|
| E3 | build check: MixerInput eps=%a (1-eps=%a), compiled eps=%a (max=%a) |
| E4 | build check: mixer with %u inputs, compiled N0=%d (layer 0), N1=%d (layer 1) |
| E5 | direct mmap |
| E6 | Dictionary too large: %d entries (max %zu) |
| E7 | ppmd page: mmap address space |
| E8 | ppmd page: open backing file |
| E9 | ppmd page: ftruncate |
| E10 | ppmd page: memfd_create |
| E11 | ppmd page: ftruncate memfd |
| E12 | ppmd page: mmap pool range |
| E13 | ppmd page: mmap pool |
| E14 | ppmd page: mmap mirror |
| E15 | ppmd page: /proc/self/statm unreadable, pool fixed at %llu MiB |
| E16 | ppmd page: no memory for the tag table |
| E17 | ppmd page: no memory for the dirty table |
| E18 | ppmd page: ring mirror not mapped |
| E19 | ppmd page: ring mirror not coherent |
| E20 | ppmd page dyn: pwrite |
| E21 | ppmd page dyn: ftruncate |
| E22 | ppmd page dyn: mmap space |
| E23 | ppmd page dyn: mmap pool |
| E24 | ppmd page dyn: mmap mirror |
| E25 | ppmd page dyn: out of memory (tags) |
| E26 | ppmd page dyn: out of memory (dirty) |
| E27 | ppmd page: pwrite |
| E28 | ppmd page: pread |
| E29 | ppmd page: release called twice - ignored |
| E30 | ppmd page: release pool |
| E31 | ppmd page: frame mapping wrong at page %llu (N=%llu, reciprocal=%llu, shift=%u) |
| E32 | ppmd page: text mapping wrong at page %llu |
| E33 | ppmd page: mirror not mapped |
| E34 | ppmd page: mirror not coherent |
| E35 | ppmd page: object spans a band boundary |
| E36 | ppmd page: setup failed (rc=%d) |
| E37 | adaptive depth: %u features, capacity %d |
| E40 | arena mmap |
| E41 | no author record in the block |
| E42 | transformer state |
| E44 | correction model: %s width %d, %d values registered |
| E53 | correction model: stage-1 row width %d, compiled %d (rebuild with %d) |
| E76 | id model: %lld does not fit %d digits |
| E77 | id model: out of memory |
| E78 | id model: value %lld outside [%lld,%lld] |
| E79 | id model: duplicate value %lld |
| E80 | id model: duplicate page id %lld |
| E103 | A or B line count differs from the page count |
| E104 | ppmd warm-up: .dict missing |
| E105 | grammar violation @%llu mask=%d ctx=%u forced=%d actual=%d (st=%d flag=%d wl=%d) |
| E106 | grammar: abort (archive would not decode) |
| E107 | build check: tail element %d is %d, not 0 |
| E112 | correction model: compiled transformer state width %d, transformer reports %d |
| E113 | id model: fxcm uses %u of %u columns, %u free ones needed at the end |
| E114 | dict_derive: %s missing |
| E115 | article order: reconstruction failed |
| E116 | cannot open %s |
| E121 | page form: out of memory |
| E138 | header sort: out of memory |
| E148 | user ids: out of memory |
| E150 | preprocess: out of memory |
| E163 | /proc/self/exe unreadable |
| E164 | self-extraction: cannot open %s |
| E165 | self-extraction: decoding the article order failed (status %d) |
| E166 | self-extraction: decoding the dictionary hints failed (status %d) |
| E167 | self-extraction: cannot read the archive trailer |
| E168 | self-extraction: cannot extract %s |
| E170 | can't open file for measuring its size |
| E171 | metablock codec: pack failed |
| E172 | ip pack: pack failed |
| E173 | ip pack: unpack failed |
| E174 | metablock codec: unpack failed |
| E175 | build check: translation units disagree on sizeof(Predictor) (%llu, %llu) |
| E176 | comment index: inverse transform failed |
| E177 | census: inverse transform failed |
| E178 | dictionary hints: derivation failed |
| E181 | census: preprocessing failed |
| E182 | comment index: preprocessing failed |
| E184 | config.ints mismatch |
| E185 | config.kimi mismatch |
| E187 | mmap failed |
| E194 | model_opt: row out of range (token %d, prev %d/%d, match %d/%d, embedding rows %d) |
| E195 | unsupported d_in |
| E196 | relu2q epilogue requires d_in==192 |
| E197 | relu2q epilogue requires d_out<=<TF2_DMLP_STR> |
| E198 | quant_bias epilogue requires d_in==192 |
| E199 | quant_bias epilogue requires d_out<=<TF2_DMLP_STR> |
| E206 | the stamped weight container %s is not readable |
| E207 | weight container %s does not match the stamped fingerprint (%016llx read, %016llx stamped) |
| E208 | reinjected alias byte 0x%02x is not in the frozen byte set |
| E209 | byte 0x%02x is not in the frozen byte set |
| E210 | transformer: pre-block of %d rows exceeds the reserve %d |
| E211 | pre- and post-block in one cycle (%d and %d tokens), post-block must come first |
| E213 | transformer loss: %.6f nats/token over %llu tokens |
| E214 | line after the metablock is empty (token %llu) |
| E215 | metablock not opened after %llu tokens |
| E216 | anchor at token %llu inside the metablock started at token %llu |
| E225 | prompt: section title at %lld overwrites a post-block (%d tokens) |
| E226 | link context: mmap of %zu B failed |
| E227 | link context: mremap to %zu B failed |
| E234 | unknown dtype code %u |
| E235 | %s: truncated file (need %zu more bytes) |
| E256 | run failed |
| E268 | %s: the tensor names do not match the stamped list |
| E404 | census block: unreadable |
| E406 | census block: preprocessing failed |
| E407 | census block: insertion failed |
| E408 | census block: extraction failed |
| E409 | census block: restore failed |
| E480 | comment numbers: out of memory |
| E483 | comment numbers: preprocessing failed |
| E484 | comment numbers: restore failed |
| E536 | text start: pack failed |
| E537 | text start: unpack failed |
| E542 | text start: forced bit differs from the stream at %llu (kind %d), the archive would not decode |
| E543 | repetition line outside the name grammar |
| E547 | dictionary article: slot file failed |
| E548 | dictionary article: slot 1 missing |
| E552 | computed values: %s missing |
| E553 | semantic cache: out of memory\n / or: rule stages: encoding failed |
| E554 | rule stages: decoding failed |
| E555 | rule stages: no ./.dict for the reader view |
| E556 | rule stages: the reader view failed |
| E557 | ppmd: heap exhausted (%d) |
