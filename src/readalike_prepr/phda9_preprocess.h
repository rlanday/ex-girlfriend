#ifndef PREPR_H
#define PREPR_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef unsigned char  U8;
typedef unsigned short U16;
typedef unsigned int   U32;
typedef uint64_t       U64;

void setpos(FILE *file, U64 newpos) { fseeko(file, newpos, SEEK_SET); }
U64 curpos(FILE *file) { return ftello(file); }
U64 blockread(FILE *file,U8 *ptr, U64 count) {return fread(ptr,1,count,file);}
U64 blockwrite(FILE *file,U8 *ptr, U64 count) {return fwrite(ptr,1,count,file);}

// One loud guard for all corpus invariants of the phda9 stage.  Paths that
// are never reached on enwik9 are removed, and in their place stands an
// abort, never a silent fallback.  The guards sit on the encoder side, at the
// earliest point where the impossible input would appear; the decoder needs
// none, because it only sees streams this encoder produced.
static void phda9_check_failed(const char *what) {
  fprintf(stderr, "preprocess: %s (stream does not match this build)\n", what);
  exit(1);
}

int phda9_line_len(const char *p, const char *end) {
    const char *q = (const char*)memchr(p, '\n', end - p);
    return q ? int(q + 1 - p) : int(end - p);
}

int phda9_is_contributor_end_line(const char *p, int len) {
    return (len >= 13 && memcmp(p, "/contributor>", 13) == 0);
}

// The encoder tail is exactly "headersize\n" + "langsize\n" + header + lang,
// so after reading those two the read position is the end of the file; the
// inherited path for a separate id side stream is removed, and
// decode_txt_wit() stops if data are left.
FILE* tmpfile2(void){
    FILE *f;
#ifdef WINDOWS
    int i;
    char temppath[MAX_PATH];
    char filename[MAX_PATH];

    //i=GetTempPath(MAX_PATH,temppath);          //store temp file in system temp path
    i=GetModuleFileName(NULL,temppath,MAX_PATH); //store temp file in program folder
    if ((i==0) || (i>MAX_PATH)) return NULL;
    char *p=strrchr(temppath, '\\');
    if (p==0) return NULL;
    p++;*p=0;
    if (GetTempFileName(temppath,"tmp",0,filename)==0) return NULL;
    f=fopen(filename,"w+bTD");
    if (f==NULL) unlink(filename);
    return f;
#else
    f=tmpfile();  // temporary file
    if (!f) return NULL;
    return f;
#endif
}

// WIT
typedef enum {FDECOMPRESS, FCOMPARE, FDISCARD} FMode;
const char UTF8bytes[256] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    3,3,3,3,3,3,3,3,4,4,4,4,5,5,5,5
};

// utf8 len
int utf8len(char *s){
    return UTF8bytes[(U32)(U8)s[0]] + 1;
}

int wfgets(char *str, int count, FILE  *fp) {
    int c, i = 0;
    while (i<count-1 && ((c=getc(fp))!=EOF)) {
        str[i++]=c;
        if (c=='\n')
            break;
    }
    str[i]=0;
    return i;
}

void wfputs(const char *str,FILE *fp) {
    for (int i = 0; *str; str++){
        putc(*str,fp);
    }
}
// UTF8 to WC (USC) and reverse
int utf8towc(char *dest, U32 ch) {
    int val=0;
    if (ch==1) {
        val=(char)dest[0] ;
        return val;
    }
    if (ch==2) {
        val=dest[1]&0x3F;
        val=val|(dest[0]&0x1F)<<6;
        return val;
    }
    if (ch==3) {
        val|=(dest[0]&0x1F)<<12;
        val|=(dest[1]&0x3F)<<6;
        val=val|(dest[2]&0x3F);
        return val;
    }
    if (ch==4) {
        val|=(dest[0]&0xF)<<18;
        val|=(dest[1]&0x3F)<<12;
        val|=(dest[2]&0x3F)<<6;
        val=val|(dest[3]&0x3F);
        return val;
    }
    return 0;
}
int wctoutf8(char *dest, U32 ch){
    if (ch < 0x80) {
        dest[0] = (char)ch;
        return 1;
    }
    if (ch < 0x800) {
        dest[0] = (ch>>6) | 0xC0;
        dest[1] = (ch & 0x3F) | 0x80;
        return 2;
    }
    if (ch < 0x10000) {
        dest[0] = (ch>>12) | 0xE0;
        dest[1] = ((ch>>6) & 0x3F) | 0x80;
        dest[2] = (ch & 0x3F) | 0x80;
        return 3;
    }
    if (ch < 0x110000) {
        dest[0] = (ch>>18) | 0xF0;
        dest[1] = ((ch>>12) & 0x3F) | 0x80;
        dest[2] = ((ch>>6) & 0x3F) | 0x80;
        dest[3] = (ch & 0x3F) | 0x80;
        return 4;
    }
    return 0;
}

// get num string length terminated by ; or non number
int numlen(char *str) {
    int c, i = 0;
    for (i = 0; *str!=';'; str++){
        if (*str<'0' || *str>'9') return 0;
        i++;
    }
    return i;
}
// reduce
//re3
void hent1(char *in,char *out){
    int j, k;
    do {
    j=*in++; *out++=j;
    if (j=='&') {
        k=*in++;
        if (k=='&')  *(int*)out=0x3B706D61, out+=4;  else//';pma'
        if (k=='"')  *(int*)out=0x746F7571, out+=4, *out++=';';  else//'touq'
        if (k=='<')  *(int*)out=0x203B746C, out+=3;  else//' ;tl'
        if (k=='>')  *(int*)out=0x203B7467, out+=3;//' ;tg'
        else   *out++=k;
    }
  }
  while (j!=0);
}
// reverse html entity numbers
void hent6(char *in,char *out){
    int j, k;
    do {
    j=*in++; *out++=j;

    if (j=='&') {
        k=*in++;
        if (k==5){// escape char
            k=utf8len(in);
            int nu=utf8towc(in,k);
            *out='&';
            sprintf(out+1, "#%d;",   nu);
            int a=numlen(out+1+1);
            out=out+2+a+1;
            in=in+k;
        }
        else   *out++=k;
    }
  }
  while (j!=0);
}
//re4
void hent3(char *in,char *out) {
    int j, k;
    do {
    j=*in++; *out++=j;
    if (j==';' && *(int*)(in-5)==0x706D6126) { //'pma&'
      k = *in++;
      if (k==3); else
      if (k=='"')  *(int*)out=0x746F7571, out+=4, *out++=';';  else//'touq'
      if (k=='<')  *(int*)out=0x203B746C, out+=3;  else//' ;tl'
      if (k=='>')  *(int*)out=0x203B7467, out+=3;  else//' ;tg'
      if (k=='!')  *(int*)out=0x7073626E, out+=4, *out++=';';  else//'psbn'
      if (k=='*')  *(int*)out=0x7361646E, out+=4, *out++='h', *out++=';'; else//'sadn'
      if (k=='^')  *(int*)out=0x7361646D, out+=4, *out++='h', *out++=';'; else //'sadm'
      //if (U8(k)==0xc2 && U8(*(in))==0xae ) {  *(int*)out=0x3B676572, out+=4,in++;}else //'reg'
      if (U8(k)==0xc2 && U8(*(in))==0xb0 ) {  *(int*)out=0x3B676564, out+=4,in++;}else
      if (U8(k)==0x0b) {  *(int*)out=0x32707573, out+=4,*out++=';';}else //'sup2' (mark)
      //if (U8(k)==0xc2 && U8(*(in))==0xb3 ) {  *(int*)out=0x33707573, out+=4,*out++=';',in++;}else //'sup3'  // old
      //if (U8(k)==0xe2 && U8(*(in))==0x82&& U8(*(in+1))==0xac ) {  *(int*)out=0x6F727565, out+=4,*out++=';',in+=2;}else //'euro'
      if (U8(k)==0xc3 && U8(*(in))==0x97  ) {  *(int*)out=0x656D6974, out+=4,*out++='s',*out++=';',in++;}//else //'times' old
      //if (U8(k)==0xe2 && U8(*(in))==0x88&& U8(*(in+1))==0x88 ) {  *(int*)out=0x6E697369, out+=4,*out++=';',in+=2;}else //'isin'
      //if (U8(k)==0xe2 && U8(*(in))==0x86&& U8(*(in+1))==0x92 ) {  *(int*)out=0x72726172, out+=4,*out++=';',in+=2;}//else //'rarr'
      //if (U8(k)==0xe2 && U8(*(in))==0x88&& U8(*(in+1))==0x92 ) {  *(int*)out=0x756E696D, out+=4,*out++='s',*out++=';',in+=2;}  //'minus' old
      else *out++=k;
    }
  }
  while (j!=0);
}
//pre3
void hent(char *in,char *out){
    int j, k;
    do {
    j=*in++; *out++=j;

    if (j=='&') {
      k = *(int*)in;
      if (k==0x3B706D61) { //';pma'
       *out++='&', in+=4; k=*(in);
      if ((k=='"') || (k=='<') || (k=='>') || (k=='!') || (k=='*') || (k=='^') ||
      (U8(k)==0xc2 && U8(*(in+1))==0xb2 ) ||
      (U8(k)==0xc2 && U8(*(in+1))==0xb3 ) ||
      (U8(k)==0xc2 && U8(*(in+1))==0xae )  ||
      (U8(k)==0xc2 && U8(*(in+1))==0xb0 )  ||
      (U8(k)==0xe2 && U8(*(in+1))==0x82&& U8(*(in+1+1))==0xac )  ||
      (U8(k)==0xc3 && U8(*(in+1))==0x97  ) ||
      (U8(k)==0xe2 && U8(*(in+1))==0x88&& U8(*(in+1+1))==0x92 )  ||
      (U8(k)==0xe2 && U8(*(in+1))==0x88&& U8(*(in+1+1))==0x88 )  ||
      (U8(k)==0xe2 && U8(*(in+1))==0x86&& U8(*(in+1+1))==0x92 )
      || (U8(k)==0x0b)

      )    *out++=3; }  else  // sup2, escape if char present
      if (k==0x746F7571 && *(in+4)==';')  *out++='"', in+=5;  else//'touq'
      {
        k = k*256 + ' ';
        if (k==0x3B746C20) *out++='<', in+=3;  else//';tl '
        if (k==0x3B746720) *out++='>', in+=3;//';tg '
      }
    }
  }
  while (j!=0);
}
//pre4
void hent2(char *in,char *out){
    int j, k;
    char *start = in;
    do {
    j=*in++; *out++=j;

    if (j=='&' && (in - start > 1 && *(in-2)=='&')) {
        k=*(int*)in;
        if (k==0x746F7571 && *(in+4)==';')  *out++='"', in+=5;  else//'touq'
        if (k==0x7073626E && *(in+4)==';')  *out++='!', in+=5;  else//'psbn'
        if (k==0x7361646E && *(in+4)=='h' && *(in+5)==';')  *out++='*', in+=6;  else//'sadn'
        if (k==0x7361646D && *(in+4)=='h' && *(in+5)==';')  *out++='^', in+=6;  else//'sadm'
        //if (k==0x3B676572  ) { *out++=0xc2,*out++=0xae, in+=4;  }  else//'reg'
        if (k==0x3B676564  ) { *out++=0xc2,*out++=0xb0, in+=4;  }  else
        if (k==0x32707573  && *(in+4)==';' ) { *out++=0x0b, in+=5;  }  else//'sup2' (mark)
        //if (k==0x33707573  && *(in+4)==';' ) { *out++=0xc2,*out++=0xb3, in+=5;  }  else//'sup3'
        if (k==0x656D6974  && *(in+4)=='s'&& *(in+5)==';' ) { *out++=0xc3,*out++=0x97, in+=6;  }  else//'times'
        //if (k==0x6E697369  && *(in+4)==';' ) { *out++=0xe2,*out++=0x88,*out++=0x88, in+=5;  }  else//'isin'
        //if (k==0x72726172  && *(in+4)==';' ) { *out++=0xe2,*out++=0x86,*out++=0x92, in+=5;  }  else//'rarr'

        //if (k==0x756E696D  && *(in+4)=='s'&& *(in+5)==';' ) { *out++=0xe2,*out++=0x88,*out++=0x92, in+=6;  }  else//'minus'
        {
        k = k*256 + ' ';
        if (k==0x3B746C20)  *out++='<', in+=3;  else//';tl '
        if (k==0x3B746720)  *out++='>', in+=3;//';tg '
        }
    }
  }
  while (j!=0);
}
//Html Entity numbers
//&#
void hent5( char *in,char *out){
    int j, k;
    do {
        j=*in++; *out++=j;
        if (j=='&' && *(in)=='#' && *(in-2)=='&' && *(in+1)>'0'&& *(in+1)<='9') { //  &&#xxx; to &@UTF8
            int n=numlen(in+1);
            int d=atoi(&in[1]);
            if (d>255 && n /*&& n!=38*/){//>2 && n<6
           /* if (d==60) {
                printf("%s",out);
                printf("%s",in);
            }*/
                in++;
                *(out-1)=5;// escape char
                int e=wctoutf8(out,d);
                out=out+e;
                in=in+n+1;
                // printf("Numlen: %d value: %d utflen: %d\n",n,d,e);
            }
        }
  }
  while (j!=0);
}

void hent9( char *in,char *out){
    int j, k;
     int t=0;

#define PROCESS(sym, src, dst, CONDITION) \
  {\
     char *t,  *p = src,  *q = dst,  *end = p + strlen(src);\
    while ( (t=strchr(p, sym)) != 0) {   \
        memcpy(q, p, t-p);  q+=t-p;      \
        int count = 0;                   \
        while(*t++ == sym)  ++count;     \
        if ((CONDITION) && (count==1 || count==2))  count = 3-count;\
        memset(q, sym, count);  q+=count;\
        p = t-1;\
    }\
    memcpy(q, p, end-p);  q[end-p]=0; \
  }
   // breaks wordmodel
   PROCESS('{', in, out, 1)
   PROCESS('}', out, in, 1)
    PROCESS('[', in, out, 1)
    PROCESS(']', out, in, 1)
    PROCESS('&', in, out, 1)
    //PROCESS(0x27, out,in, 1)
   // PROCESS('\'', out, in, 1)
   do {
           j=*in++; *out++=j;
        }
        while (j!=0);

}
void skipline( char *in,char *out ){
    int j;
    do {
        j=*in++; *out++=j;
    }
    while (j!=0);
}
// &" -> "   for now
void removeamp( char *in,char *out ,int skip){
    int j;
    for (int i=0;i<skip;i++) {j=*in++; *out++=j;
    }
    char *outo=out;
    bool amp=false;
     do {
        j=*in++;
        if (j=='"' || j=='<' || j=='>')  { /*assert(p[-1]=='&');*/  --out; amp=true; }
        *out++=j;
      }
      while (j!=0);
      //if (amp==true) printf("%s",outo);
}
void restoreamp( char *in,char *out,int skip ){
    int j;
    for (int i=0;i<skip;i++) {j=*in++; *out++=j;
    }
     do {
        j=*in++;
        if (j=='"' || j=='<' || j=='>')  { /*assert(p[-1]=='&');*/  *out++='&'; }
        *out++=j;
      }
      while (j!=0);
}

#include "census_transform.h"

static long long revts_key[300000];   // time keys, ascending
static long long revts_val[300000];   // matching rev_id
static int       revts_n = 0;

// time key from "<yy><mt>:<sek>" (phda9 timestamp encoding)
static long long revts_from_timestamp(const char *p) {
  long long yy = (p[0] - '0') * 10 + (p[1] - '0');
  const char *q = p + 2;
  long long mt = 0;
  while (*q >= '0' && *q <= '9') mt = mt * 10 + (*q++ - '0');
  if (*q == ':') ++q;
  long long sec = 0;
  while (*q >= '0' && *q <= '9') sec = sec * 10 + (*q++ - '0');
  return (yy * 366 + mt) * 86400 + sec;
}

static long long revts_predict(long long k) {
  if (revts_n < 2) return 0;
  int lo = 0, hi = revts_n;                    // lower_bound
  while (lo < hi) { int m = (lo + hi) >> 1; if (revts_key[m] < k) lo = m + 1; else hi = m; }
  int a = lo - 1, b = lo;
  if (a < 0) a = 0; if (a > revts_n - 1) a = revts_n - 1;
  if (b < 0) b = 0; if (b > revts_n - 1) b = revts_n - 1;
  if (revts_key[b] == revts_key[a]) return revts_val[a];
  return revts_val[a] + (long long)(((__int128)(k - revts_key[a]) *
         (__int128)(revts_val[b] - revts_val[a])) /
         (__int128)(revts_key[b] - revts_key[a]));
}

static void revts_add(long long k, long long v) {
  if (revts_n >= 300000) return;          // budget: same on both sides
  int lo = 0, hi = revts_n;                    // upper_bound (= insort_right)
  while (lo < hi) { int m = (lo + hi) >> 1; if (revts_key[m] <= k) lo = m + 1; else hi = m; }
  memmove(revts_key + lo + 1, revts_key + lo, (size_t)(revts_n - lo) * sizeof(long long));
  memmove(revts_val + lo + 1, revts_val + lo, (size_t)(revts_n - lo) * sizeof(long long));
  revts_key[lo] = k; revts_val[lo] = v; ++revts_n;
}

static long long zigzag(long long x)   { return x >= 0 ? 2 * x : -2 * x - 1; }
static long long unzigzag(long long u) { return (u & 1) ? -((u + 1) / 2) : u / 2; }

// enc=1: rev_id -> zigzag(rev_id - pred);  enc=0: back again.
// out must hold 2*n+4096 bytes. Returns: length of the output.
static size_t revid_apply(const char *in, size_t n, char *out, int enc) {
  revts_n = 0;
  size_t i = 0, o = 0;
  while (i < n) {
    const char *nl = (const char*)memchr(in + i, '\n', n - i);
    size_t len = nl ? (size_t)(nl - (in + i)) + 1 : n - i;
    if (len == 10 && memcmp(in + i, "revision>\n", 10) == 0) {
      size_t i2 = i + len;
      const char *nl2 = i2 < n ? (const char*)memchr(in + i2, '\n', n - i2) : 0;
      size_t len2 = nl2 ? (size_t)(nl2 - (in + i2)) + 1 : 0;
      size_t i3 = i2 + len2;
      const char *nl3 = i3 < n ? (const char*)memchr(in + i3, '\n', n - i3) : 0;
      size_t len3 = nl3 ? (size_t)(nl3 - (in + i3)) + 1 : 0;
      if (len2 > 4 && memcmp(in + i2, "id>", 3) == 0 &&
          len3 > 11 && memcmp(in + i3, "timestamp>", 10) == 0) {
        long long k = revts_from_timestamp(in + i3 + 10);
        long long pred = revts_predict(k);
        long long val = 0;
        for (size_t t = 3; t + 1 < len2; ++t) val = val * 10 + (in[i2 + t] - '0');
        long long rev, wr;
        if (enc) { rev = val;                   wr = zigzag(rev - pred); }
        else     { rev = unzigzag(val) + pred; wr = rev; }
        memcpy(out + o, in + i, len); o += len;             // "revision>"
        o += (size_t)sprintf(out + o, "id>%lld\n", wr);
        revts_add(k, rev);
        i = i3;                                            // copy timestamp normally
        continue;
      }
    }
    memcpy(out + o, in + i, len); o += len;
    i += len;
  }
  return o;
}

static unsigned long long name_index_hash(const char *p, int len) {
  unsigned long long h = 1469598103934665603ULL;
  for (int i = 0; i < len; ++i) {
    h ^= (unsigned char)p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

// name -> uid line, open addressing, power-of-two slots.  Both sides build it
// in the same order from the same bytes, so both hold the same table.
typedef struct {
  int *slot;                    // -1 = free, else entry index
  int slots, mask, n;
  int *noff, *nlen;             // name in the arena
  int *uoff, *ulen;             // uid LINE (with tag and newline) in the arena
  char *arena; size_t apos;
} name_index_t;

static void name_index_init(name_index_t *t, int cap, size_t acap) {
  int s = 1024;
  while (s < 2 * cap) s <<= 1;
  t->slots = s; t->mask = s - 1; t->n = 0; t->apos = 0;
  t->slot = (int*)malloc((size_t)s * sizeof(int));
  t->noff = (int*)malloc((size_t)(cap + 1) * sizeof(int));
  t->nlen = (int*)malloc((size_t)(cap + 1) * sizeof(int));
  t->uoff = (int*)malloc((size_t)(cap + 1) * sizeof(int));
  t->ulen = (int*)malloc((size_t)(cap + 1) * sizeof(int));
  t->arena = (char*)malloc(acap + 1);
  if (!t->slot || !t->noff || !t->nlen || !t->uoff || !t->ulen || !t->arena) {
    fprintf(stderr, "preprocess: out of memory\n"); exit(1);
  }
  for (int i = 0; i < s; ++i) t->slot[i] = -1;
}

static void name_index_free(name_index_t *t) {
  free(t->slot); free(t->noff); free(t->nlen);
  free(t->uoff); free(t->ulen); free(t->arena);
}

static int name_index_find(const name_index_t *t, const char *p, int len) {
  int i = (int)(name_index_hash(p, len) & (unsigned long long)t->mask);
  while (t->slot[i] != -1) {
    int e = t->slot[i];
    if (t->nlen[e] == len && memcmp(t->arena + t->noff[e], p, (size_t)len) == 0)
      return e;
    i = (i + 1) & t->mask;
  }
  return -1;
}

// first occurrence wins and is never overwritten - identical on both sides.
static void name_index_add(name_index_t *t, const char *p, int len,
                         const char *u, int ulen) {
  int i = (int)(name_index_hash(p, len) & (unsigned long long)t->mask);
  while (t->slot[i] != -1) i = (i + 1) & t->mask;
  int e = t->n++;
  t->noff[e] = (int)t->apos; t->nlen[e] = len;
  memcpy(t->arena + t->apos, p, (size_t)len); t->apos += (size_t)len;
  t->uoff[e] = (int)t->apos; t->ulen[e] = ulen;
  memcpy(t->arena + t->apos, u, (size_t)ulen); t->apos += (size_t)ulen;
  t->slot[i] = e;
}

static int hcf_is_group_start(const char *p, int len) {
  return (len >= 12 && memcmp(p, "contributor>", 12) == 0);
}

// enc=1: moves the contributor group in front of the timestamp line; enc=0:
// moves it back.  The user id at the first mention is handled by the header
// sort of the metadata block.  out must hold 2*n+4096 bytes.  Returns the
// length written.
static size_t hcf_apply(const char *in, size_t n, char *out, int enc) {
  const char *end = in + n;

  size_t i = 0, o = 0;
  while (i < n) {
    int len = phda9_line_len(in + i, end);

    // ---- find the pair (timestamp, group) in the order of the current side
    size_t g0 = 0, g1 = 0, ts = 0; int tslen = 0; int hit = 0;
    if (enc) {
      if (len > 10 && memcmp(in + i, "timestamp>", 10) == 0) {
        size_t k = i + (size_t)len;
        int lk = k < n ? phda9_line_len(in + k, end) : 0;
        if (lk && hcf_is_group_start(in + k, lk)) {
          ts = i; tslen = len; g0 = k; hit = 1;
        }
      }
    } else {
      if (hcf_is_group_start(in + i, len)) {
        size_t k = i;
        while (k < n) {                          // skip to the end of the group
          int lk = phda9_line_len(in + k, end);
          k += (size_t)lk;
          if (phda9_is_contributor_end_line(in + k - lk, lk)) break;
        }
        int lk = k < n ? phda9_line_len(in + k, end) : 0;
        if (lk > 10 && memcmp(in + k, "timestamp>", 10) == 0) {
          g0 = i; g1 = k; ts = k; tslen = lk; hit = 1;
        }
      }
    }
    if (!hit) { memcpy(out + o, in + i, (size_t)len); o += (size_t)len;
                i += (size_t)len; continue; }

    if (enc) {                                   // group end only needed here
      size_t k = g0;
      while (k < n) {
        int lk = phda9_line_len(in + k, end);
        k += (size_t)lk;
        if (phda9_is_contributor_end_line(in + k - lk, lk)) break;
      }
      g1 = k;
    } else {
      memcpy(out + o, in + ts, (size_t)tslen); o += (size_t)tslen;
    }

    // ---- the group itself, with the uid rule
    size_t j = g0;
    while (j < g1) {
      int lj = phda9_line_len(in + j, end);
      memcpy(out + o, in + j, (size_t)lj); o += (size_t)lj; j += (size_t)lj;
    }

    if (enc) { memcpy(out + o, in + ts, (size_t)tslen); o += (size_t)tslen;
               i = g1; }
    else     { i = ts + (size_t)tslen; }
  }
  return o;
}

#include "header_sort.h"
#include "page_form.h"
void decode_txt_wit(FILE*in,  FILE*out1,U64 size){
    char s[8192*8];
    char o[8192*8];
    int i, j, f = 0, lastID = 0,tf=0;
    wfgets(s, 22, in);
    U64 winfo=atoi(&s[0]);
    U64 insize=ftello(in);
    U64 tsize=size;
    U64 bodystart;
    setpos(in,insize+1);
    wfgets(s, 16, in);
    int headerlenght=atoi(&s[0]);
    char *h1=(char*)calloc(headerlenght+1,1);
    char *h1p=h1;
    wfgets(s, 16, in);
    if (atoi(&s[0]) != 0) phda9_check_failed("lang stream not empty");

    if (!headerlenght) phda9_check_failed("metablock empty");
    blockread(in,(U8*)h1,U64(headerlenght));
    bodystart = curpos(in);
    if (bodystart != insize + 1 + winfo)
      phda9_check_failed("metablock end differs from the head line");
    if (bodystart > (U64)size) phda9_check_failed("metablock longer than the stream");
    {
      char *rs = (char*)malloc(5 * (size_t)headerlenght + 4096);
      if (!rs) { fprintf(stderr, "preprocess: out of memory\n"); exit(1); }
      int rslen = (int)header_sort_apply(h1, (size_t)headerlenght, rs, 0);
      free(h1);
      h1 = rs;
      h1p = h1;
      headerlenght = rslen;
    }
    {
      char *cf = (char*)malloc(2 * (size_t)headerlenght + 4096);
      if (!cf) { fprintf(stderr, "preprocess: out of memory\n"); exit(1); }
      int cflen = (int)hcf_apply(h1, (size_t)headerlenght, cf, 0);
      free(h1);
      h1 = cf;
      h1p = h1;
      headerlenght = cflen;
    }

    setpos(in,bodystart);
    int header=0;
    do {
        j = g1_dec_getline(s, in, tsize);
        if (j <= 0 && !g1_dec_offen()) break;

        if (header==1){
            int n=0;
            int cont=0;
            do {
                *(int*)o = 0x20202020;
                j=4;
                //'ider' 'iver' 'tser'
                if (*(int*)&h1p[0]!=0x69646572 &&*(int*)&h1p[0]!=0x69766572&& *(int*)&h1p[0]!=0x74736572 && n!=0) *(int*)(o+j)= 0x20202020,j=j+2;
                if (cont==1) *(int*)(o+j)= 0x20202020,j=j+2;
                if (*(int*)&h1p[0]==0x6E6F632F) cont=0;//'noc/'
                int k=(int)(strchr(h1p,10)+1-(char*)h1p);

                // id
                // The first line of a record is always the page-id delta
                // (243,425 records, no namespace lines); the encoder writes
                // no namespace line, so the decoder has no branch for one.
                if ( n==0){
                    n++;
                    lastID = lastID+ atoi(&h1p[1]);
                    sprintf(o+j, "<id>%d</id>%c",   lastID, 10);
                    wfputs(o,out1);
                }
                else if (*(int*)&h1p[0]==0x656D6974 ){//'emit'
                    char *p = strchr(h1p, ':');
                    int d = atoi(&h1p[19-7]), hms = atoi(p+1), h = hms/3600;
                    o[0]=h1p[17-7],o[1]=h1p[18-7],o[2]=' ';
                    int y=atoi(&o[0]);
                    sprintf(o, "      <timestamp>%d-%02d-%02dT%02d:%02d:%02dZ</timestamp>%c",
                    y + 2001, d/31+1, d%31+1, h, hms/60 - h*60, hms%60, 10);
                    wfputs(o,out1);
                }
                else if (n || cont==1){
                    o[j]='<';
                    memcpy(o+j+1, h1p, k);
                    int e=(strchr(h1p, '>')-h1p)+2;
                    if (e!=k || cont==1) { //end tag
                        o[k+j++]='<';
                        o[k+j++]='/';
                        memcpy(o+j+k, h1p, k);
                        j=j+(strchr(h1p, '>')-h1p)+1;

                    }
                    o[k+j++]=10;
                    o[k+j]=0;
                    wfputs(o,out1);
                    if (*(int*)&h1p[0]==0x746E6F63) cont=1;//'tnoc'
                }
                h1p=h1p+k;
            }
            while (memcmp(&h1p[0],"/contributor>",13)    );
            *(int*)o = 0x20202020;
            j=4;
            *(int*)(o+j)= 0x20202020,j=j+2;
            int k=(int)(strchr(h1p,10)+1-(char*)h1p);
            o[j]='<';
            memcpy(o+j+1, h1p, k);
            o[k+j++]=10;
            o[k+j]=0;
            wfputs(o,out1);
            h1p=h1p+k;
            header=0;
        }

        {
            if (tf==0 && memcmp(&s[j-9],"</title>",8)==0 && *(int*)s==0x20202020)
            header=1;
        }

            skipline(s,o);// remove this
            hent9(o,s);
            skipline(o,s);   // plain copy o -> s

            int skip=0;
            char *p = strstr(s, "<text ");//, *w;
            if (p)  { tf=1, p = strchr(p, '>');   skip= (char*)p+1-(char*)s;
                if(p[-1]=='/' /*|| p[2]==0*/) tf=0;

            }

            if (strstr(s, "</text>") || strstr(s, "</revision>")|| strstr(s, "</page>")/*|| strstr(s, "</sha1>")*/)  tf=0;
            if (tf) {
                restoreamp(s,o,skip);
                skipline(o,s);
            }
            hent6( s,o);
            hent1( o,s);
            hent3( s,o);
            wfputs(o,out1);

    }
    while (g1_dec_offen() || curpos(in) < tsize);
    free(h1);
}

void encode_txt_wit(FILE* in, FILE* out) {
    char s[8192*8];
    char o[8192*8];
    FILE *out3=tmpfile2();
    FILE *body=tmpfile2();

    for (int i = 0; i<21; i++){
        putc(32,out);
    }
    putc('\n',out);
    int i, j, f = 0, lastID = 0,tf=0;

  do {
    j=wfgets(s, 8192*8, in);

    if (f==2) {
        if (*(int*)&s[4]==0x3E736E3C) phda9_check_failed("ns record in the metablock");
        // id
        int curID = atoi(&s[8]);
        if (*(int*)&s[4] != 0x3E64693C){
        phda9_check_failed("page record without <id>");
        }//return 0;// just fail it '>di<'
        //if (curID <= lastID)        return 0;
       sprintf(o,  ">%d%c", curID - lastID, 10);
        wfputs(o,out3);
        lastID = curID;
        f = 1;
        continue;
    }
    if (f) {
        if (*(int*)&s[6]==0x6D69743C) {//'mit<'
            int year   = atoi(&s[17]);
            int month  = atoi(&s[22]);
            int day    = atoi(&s[25]);
            int hour   = atoi(&s[28]);
            int minute = atoi(&s[31]);
            int second = atoi(&s[34]);
            sprintf(o, "timestamp>%02d%d:%d%c",
                    year-2001, month*31+day-32, hour*3600+minute*60+second, 10);
            wfputs(o,out3);
            continue;
        }
         char *p =strchr(s, '>');
        if (p) {
            p = strchr(p+1, '<');
            if (p)  p[0] = 10,  p[1] = 0;
        }
        if (s[0]!=' ' ||s[1]!=' ' ||s[2]!=' ' ||s[3]!=' ') {
          phda9_check_failed("revision line without indentation");
           }
        int s2=0; //lenght
        if (f==3) {
            if (*(int*)&s[6]==0x6F632F3C)  s2=7;//'oc/<'
            else                       s2=9;
        }
        else {
          // On four-space indentation the metablock sees exactly two tags on
          // enwik9, `<revision>` (243,425) and `<restrictions>` (1,013);
          // `<redirect` does not occur.  The guard catches every unexpected
          // four-space tag (at six-space indentation s[4] is a blank).
          if (*(int*)&s[4]==0x7665723C || *(int*)&s[4]==0x7365723C) s2=5;//'ver<' 'ser<'
          else {
            if (s[4]=='<') phda9_check_failed("unexpected element at level 4");
            s2=7;
          }
          if (*(int*)&s[6]==0x6E6F633C) {//'noc<'
           // `<contributor deleted` does not occur in enwik9: loud guard
           // instead of the inherited special case.
           if (*(int*)&s[6+4+4+4]==0x6C656420) phda9_check_failed("contributor deleted");
           f=3;
          }
        }
        // every branch above assigns 5, 7 or 9
        wfputs(s+s2,out3);
    }
    else  {hent(s,o);
        hent2(o,s);
        hent5(s,o);

        int skip=0;
        char *p = strstr(o, "<text ");
        if (p)  { tf=1, p = strchr(p, '>');  skip= (char*)p+1-(char*)o;
        if(p[-1]=='/' /*|| p[2]==0*/) tf=0;

        }

        if (strstr(o, "</text>") || strstr(o, "</revision>")|| strstr(o, "</page>"))  tf=0;
        if (tf) {
        removeamp(o,s,skip);
        skipline(s,o);

        }
        hent9(o,s);
        { g1_add(o);
          if (g1_pn >= 10 &&
              memcmp(g1_pb + g1_pn - 10, "  </page>\n", 10) == 0)
            g1_page_out(body); }
}
    if (tf==0) for(i=0; i<j; i++)
      if (*(int*)&s[i]==0x69742F3C && *(int*)&s[i+4]==0x3E656C74&&*(int*)s==0x20202020) f=2;//'it/<' '>elt'

    for(i=0; i<j; i++)
      if (*(int*)&s[i]==0x6F632F3C && *(int*)&s[i+4]==0x6972746E) f=0;//'oc/<' 'irtn'
  }
  while (!feof(in));
  if (g1_pn) { fprintf(stderr, "page form: %lu bytes without page anchor at stream end\n", (unsigned long)g1_pn); exit(1); }
  g1_bericht();
  // output tail to main file and report tail size as info
  U64 msize=curpos(body);
  int tsize=0;                       // the language-link stream is always empty
  int headersize=curpos(out3);
   setpos(out3,0);
   char *hdr_buf = 0;
   {
     char *raw = (char*)malloc((size_t)headersize + 1);
     hdr_buf = (char*)malloc(2 * (size_t)headersize + 4096);
     if (!raw || !hdr_buf) { fprintf(stderr, "preprocess: out of memory\n"); exit(1); }
     blockread(out3, (U8*)raw, (U64)headersize);
     memcpy(hdr_buf, raw, (size_t)headersize);
     free(raw);
   }
   {
     char *cf = (char*)malloc(2 * (size_t)headersize + 4096);
     if (!cf) { fprintf(stderr, "preprocess: out of memory\n"); exit(1); }
     headersize = (int)hcf_apply(hdr_buf, (size_t)headersize, cf, 1);
     free(hdr_buf);
     hdr_buf = cf;
   }
   {
     char *rs = (char*)malloc(2 * (size_t)headersize + 4096);
     if (!rs) { fprintf(stderr, "preprocess: out of memory\n"); exit(1); }
     headersize = (int)header_sort_apply(hdr_buf, (size_t)headersize, rs, 1);
     free(hdr_buf);
     hdr_buf = rs;
   }
   sprintf(o, "%d%c", headersize, 10);
   j=strlen(o);
   wfputs(o,out); //header
   sprintf(o, "%d%c", tsize, 10);
   j=j+strlen(o);
   wfputs(o,out); //lang

   blockwrite(out, (U8*)hdr_buf, (U64)headersize);
   free(hdr_buf);
   {
     if (curpos(out) != (U64)(22 + j + headersize))
       phda9_check_failed("metablock end differs from the length lines");
     setpos(body, 0);
     const size_t cbn = 1u << 20;
     U8 *cb = (U8*)malloc(cbn);
     if (!cb) { fprintf(stderr, "preprocess: out of memory\n"); exit(1); }
     U64 n, ges = 0;
     while ((n = blockread(body, cb, (U64)cbn)) > 0) { blockwrite(out, cb, n); ges += n; }
     free(cb);
     // `msize` is curpos(body), and with the metadata block at the head `body`
     // is a file of its
     // own that starts at 0 - so it is the page part WITHOUT the head line.
     if (ges != msize)
       phda9_check_failed("page part incomplete");
     fclose(body);
   }

   fclose(out3);
 /* printf("Main size: %d kb\n",U32(msize/1024));
  printf("header size: %d kb\n",U32(headersize/1024));
  printf("Langs size: %d kb\n",U32(tsize/1024));
  */
  tsize=tsize+j+headersize;
  // write out size of tail lenght
  setpos(out,0);
  sprintf(o, "%d%c", tsize, 20);
  for (int i = 0; i<20; i++){
       U8 a=o[i];
       if (a>='0' && a<='9') putc(a,out);
       else break;
  }

}
// end WIT

bool cat(char const * filename_from1, char const * filename_from2, char const * filename_to) {
  FILE* ifile1 = fopen(filename_from1, "rb");
  FILE* ifile2 = fopen(filename_from2, "rb");
  FILE* ofile = fopen(filename_to, "wb");

  do {
    int c=getc(ifile1);
    if (c==EOF) break;
    putc(c,ofile);
  }
  while (!feof(ifile1));
  do {
    int c=getc(ifile2);
    if (c==EOF) break;
    putc(c,ofile);
  }
  while (!feof(ifile2));
  fclose(ifile1);
  fclose(ifile2);
  fclose(ofile);

  return true;
}

int phda9_prepr() {
   // open files
  FILE *in=fopen(".main_reordered", "rb");
  //if (!in)  exit(1);
  FILE *out=fopen(".main_phda9prepr", "wb");
  //if (!out)  exit(1);

  // process file
  encode_txt_wit(in,out);
  fclose(in) ;
  fclose(out);
  return 0;
}

int phda9_resto() {

  FILE *in=fopen(".main_decomp", "rb");
  //if (!in)  exit(1);
  FILE *out=fopen(".main_decomp_restored", "wb");
  //if (!out)  exit(1);

  // get size
  fseek(in, 0, SEEK_END);
  U64 insize=ftell(in);
  fseek(in, 0, SEEK_SET);
  // process file
  decode_txt_wit(in,out,insize);
  fclose(in) ;
  fclose(out);
  return 0;
}
#endif // PREPR_H
