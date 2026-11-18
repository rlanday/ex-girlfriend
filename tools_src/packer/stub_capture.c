/* stub_capture.c - build check only, not part of comp9 or archive9.  LD_PRELOAD
 * helper for check (d) of the packer build: the packed binary is started for real; when
 * the image that its stub exec'd starts up, the dynamic loader runs this
 * constructor before anything of the image itself, which copies the running
 * image (/proc/self/exe, the memory file the stub wrote) to $STUB_CAPTURE and
 * exits.  So the build sees what the SHIPPED stub produced from the SHIPPED
 * stream, not only what the probe build produces from its own stream. */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
__attribute__((constructor)) static void stub_capture(void) {
  const char *o = getenv("STUB_CAPTURE");
  FILE *f, *g;
  char buf[65536];
  size_t n;
  if (!o) return;
  f = fopen("/proc/self/exe", "rb");
  g = fopen(o, "wb");
  if (!f || !g) _exit(91);
  while ((n = fread(buf, 1, sizeof buf, f)) > 0)
    if (fwrite(buf, 1, n, g) != n) _exit(92);
  if (fclose(g) != 0) _exit(93);
  fclose(f);
  _exit(0);
}
