// ppmd is written by Dmitry Shkarin.
// mod_ppmd is adapted from ppmd by Eugene Shelwien.
// This file is adapted from mod_ppmd_v2: http://encode.su/threads/2515-mod_ppmd
//
// Heap: fx2-cmix and cmix-lex kept the PPM heap in a MAP_SHARED mapping of a
// disk file (cmix-lex: stable mapping, MADV_DONTNEED cadence against the RSS
// limit).  This program replaces the mapping by its own paging (the paging
// code next to this file): the heap keeps a reserved address range, so every
// pointer value and index of PPMd stays as in memory, and the data live in a
// pool of 4 KiB frames backed by a sparse working file.  The program sets the
// resident size itself; the model output does not depend on it.

#include "ppmd.h"
#include "../mixer_selectors.h"
#include "../ppmd_meta.h"
#include "../ppmd_order_table.h"

#include "../ppmd_prewarm.h"

#include <cstring>
#include <malloc.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>

#include "ppmd_page.h"
#include "ppmd_page_ptr.h"
#include "ppmd_page.cpp"

namespace PPMD {

template <class T> T Min( T x, T y ) { return (x<y) ? x : y; }
template <class T> T Max( T x, T y ) { return (x>y) ? x : y; }

#pragma pack(1)

typedef unsigned short word;
typedef unsigned int uint;
typedef unsigned char byte;
typedef unsigned long long qword;

const int ORealMAX=256;

static signed char EscCoef[12] = { 16, -10, 1, 51, 14, 89, 23, 35, 64, 26, -42, 43 };

static const byte ExpEscape[16]={ 51,43,18,12,11,9,8,7,6,5,4,3,3,2,2,2 };

struct ppmd_Model {

  typedef unsigned short word;
  typedef unsigned int uint;
  typedef unsigned char byte;
  typedef unsigned long long qword;

  enum{ SCALE=1<<15 };

  enum{
    UNIT_SIZE=12,
    N1=4, N2=4, N3=4, N4=(128+3-1*N1-2*N2-3*N3)/4,
    N_INDEXES=N1+N2+N3+N4
  };

  byte* HeapStart;

  uint Ptr2Indx( void* p ) {
    qword addr = ((byte*)p)-HeapStart;
    uint lim = (uint)(UnitsStart-HeapStart);
    uint indx = (addr>=lim) ? (addr-lim)/UNIT_SIZE+lim : addr;
    return indx;
  }

  void* Indx2Ptr( uint indx ) {
    uint lim = (uint)(UnitsStart-HeapStart);
    qword addr = (indx>=lim) ? qword(indx-lim)*UNIT_SIZE+lim : indx;
    return HeapStart + addr;
  }

struct _MEM_BLK {
  uint Stamp;
  uint NextIndx;
  uint NU;
};

struct BLK_NODE {
  uint Stamp;
  uint NextIndx;
  int avail() const { return (NextIndx!=0); }
};

struct MEM_BLK;   // forward: the proxy typedefs must stand here, the
                  // BLK_NODE helpers below already touch heap blocks.
// Heap pointers are proxies (hzg, see the paging pointer header) that hold the
// canonical address and translate it to a pool frame on every access.
typedef ppmd_page::hzg<BLK_NODE,1> pBLK_NODE;   // may point to BList/s0 (outside the heap)
typedef ppmd_page::hzg<MEM_BLK,1>  pMEM_BLK;   // likewise
typedef ppmd_page::hzg<byte,0>     pBYTE;
#  define PAGE_W(p)     ((p).w())
#  define PAGE_WI(p,i)  ((p).w(i))
#  define PAGE_RAW_W(p)  ((p).w())

pBLK_NODE  getNext( pBLK_NODE  This ) {
  return (pBLK_NODE)Indx2Ptr(This->NextIndx);
}

void setNext( pBLK_NODE  This, pBLK_NODE  p ) {
  PAGE_W(This)->NextIndx = Ptr2Indx(p);
}

void link( pBLK_NODE  This, pBLK_NODE  p ) {
  PAGE_W(p)->NextIndx = This->NextIndx;
  setNext( This, p );
}

void unlink( pBLK_NODE  This ) {
  PAGE_W(This)->NextIndx = getNext(This)->NextIndx;
}

void* remove( pBLK_NODE  This ) {
  pBLK_NODE  p = getNext(This);
  unlink(This);
  PAGE_W(This)->Stamp--;
  return p;
}

void insert( pBLK_NODE  This, void* pv, int NU ) {
  pBLK_NODE  p = (pBLK_NODE)pv;
  link(This,p);
  PAGE_W(p)->Stamp = ~uint(0);
  // _MEM_BLK is 12 B, BLK_NODE only 8: the NU field lies OUTSIDE the length
  // `p` is translated with, so translate the full length explicitly.
  ((_MEM_BLK*)ppmd_page::xl_w((unsigned char*)(void*)p, sizeof(_MEM_BLK)))->NU = NU;
  PAGE_W(This)->Stamp++;
}

struct MEM_BLK : public BLK_NODE {
  uint NU;
};

BLK_NODE BList[N_INDEXES+1];

  uint GlueCount;
  uint GlueCount1;
  qword SubAllocatorSize;
  pBYTE pText;
  pBYTE UnitsStart;
  pBYTE LoUnit;
  pBYTE HiUnit;
  pBYTE AuxUnit;

uint U2B( uint NU ) {
  return 8*NU+4*NU;
}

int StartSubAllocator( qword SASize ) {
  qword t = SASize << 20U;

  {
    const unsigned long long puf =
      (unsigned long long)(2048) * 1024ULL * 1024ULL;
    const int rc = ppmd_page::start(t, puf, "ppm.page");
    if( rc ) {
      fprintf(stderr, "ppmd page: setup failed (rc=%d)\n", rc);
      exit(EXIT_FAILURE);
    }
    HeapStart = (byte*)ppmd_page::V.heap;
    SubAllocatorSize = t;
    return 1;
  }
}

void InitSubAllocator() {
  memset( BList, 0, sizeof(BList) );
  HiUnit = (pText=HeapStart) + SubAllocatorSize;
  qword Diff = SubAllocatorSize/8/UNIT_SIZE*7 *UNIT_SIZE ;
  LoUnit = UnitsStart = HiUnit-Diff;
  ppmd_page::zonen_setze( (unsigned long long)(UnitsStart-HeapStart) );
  GlueCount=GlueCount1=0;

}

void MetaSchnitt( qword modellbytes ) {
  if( MetaGetan || !ppmd_meta::block_fertig ) return;
  MetaGetan = true;
  LtMeta = false;
  (void)modellbytes;
  StartModelRare();
  {
    size_t n = 0;
    unsigned char* m = ppmd_prewarm::Material(&n);
    for( size_t i=0; i<n; i++ ) { ppmd_UpdateByte( m[i] ); ppmd_PrepareByte(); }
    free( m );
  }
  // The metablock's one-shot tables (metablock unit, warm-up
  // material) are freed by now.  Most of them lie below the mmap threshold,
  // so their pages would stay resident in the arena for the whole run
  // (measured 9.1 MiB).  Coder and decoder both pass here once.
  malloc_trim(0);
}

qword GetUsedMemory() {
  int i;
  qword RetVal = SubAllocatorSize - (HiUnit-LoUnit) - (UnitsStart-pText);
  for( i=0; i<N_INDEXES; i++ ) {

    RetVal -= qword(Indx2Units[i]*BList[i].Stamp)*12;
  }
  return RetVal;
}

void StopSubAllocator() {
  if( SubAllocatorSize ) SubAllocatorSize=0, delete[] HeapStart;

}

void ReleaseHeap() {
  if( !SubAllocatorSize ) return;
  SubAllocatorSize = 0;
  ppmd_page::freigeben();
  HeapStart = NULL;
  // The pointers into the heap are dead now.  They are not read any more (see
  // above), but a null pointer dies louder than a dangling one should anyone
  // reach behind this point after all.
  pText = UnitsStart = LoUnit = HiUnit = AuxUnit = NULL;
}

void GlueFreeBlocks() {
  uint i, k, sz;
  MEM_BLK s0;
  pMEM_BLK p, p0=&s0, p1;

  if( LoUnit!=HiUnit ) PAGE_WI(LoUnit,0)=0;

  for( PAGE_W(p0)->NextIndx=0,i=0; i<=N_INDEXES; i++ ) {
     while( BList[i].avail() ) {
       p = (pMEM_BLK)remove(&BList[i]);
       if( p->NU ) {
         while( p1 = p + p->NU, p1->Stamp==~uint(0) ) {
           PAGE_W(p)->NU += p1->NU;
           PAGE_W(p1)->NU = 0;
         }
         link(p0,p); p0=p;
       }
     }
  }

  while( s0.avail() ) {
    p = (pMEM_BLK)remove(&s0);
    sz= p->NU;
    if( sz ) {
      for(; sz>128; sz-=128, p+=128 ) insert(&BList[N_INDEXES-1],p,128);
      i = Units2Indx[sz-1];
      if( Indx2Units[i] != sz ) {
        k = sz - Indx2Units[--i];
        insert( &BList[k-1], p+(sz-k) , k );
      }
      insert( &BList[i], p, Indx2Units[i] );
    }
  }

  GlueCount = 1 << (13+GlueCount1++);
}

void SplitBlock( void* pv, uint OldIndx, uint NewIndx ) {
  uint i, k, UDiff=Indx2Units[OldIndx]-Indx2Units[NewIndx];
  byte* p = ((byte*)pv)+U2B(Indx2Units[NewIndx]);
  i = Units2Indx[UDiff-1];
  if( Indx2Units[i]!=UDiff ) {
    k=Indx2Units[--i];
    insert(&BList[i],p,k);
    p += U2B(k);
    UDiff -= k;
  }
  insert( &BList[Units2Indx[UDiff-1]], p, UDiff );
}

void* AllocUnitsRare( uint indx ) {
  uint i = indx;
  do {
    if( ++i == N_INDEXES ) {
      if( !GlueCount-- ) {
        GlueFreeBlocks();
        if( BList[i=indx].avail() ) return remove(&BList[i]);
      } else {
        i = U2B(Indx2Units[indx]);
        return (UnitsStart-pText>i) ? UnitsStart-=i : NULL;
      }
    }
  } while( !BList[i].avail() );

  void* RetVal=remove(&BList[i]);
  SplitBlock( RetVal, i, indx );

  return RetVal;
}

void* AllocUnits( uint NU ) {
  uint indx = Units2Indx[NU-1];
  if( BList[indx].avail() ) return remove(&BList[indx]);
  void* RetVal=LoUnit;
  LoUnit += U2B(Indx2Units[indx]);
  if( LoUnit<=HiUnit ) return RetVal;
  LoUnit -= U2B(Indx2Units[indx]);
  return AllocUnitsRare(indx);
}

void* AllocContext() {
  if( HiUnit!=LoUnit ) return HiUnit-=UNIT_SIZE;
  return BList->avail() ? remove(BList) : AllocUnitsRare(0);
}

void FreeUnits( void* ptr, uint NU ) {
  uint indx = Units2Indx[NU-1];
  insert( &BList[indx], ptr, Indx2Units[indx] );
}

void FreeUnit( void* ptr ) {
  int i = (byte*)ptr > UnitsStart+128*1024 ? 0 : N_INDEXES;
  insert( &BList[i], ptr, 1 );
}

void UnitsCpy( void* Dest, void* Src, uint NU ) {
  // Source and destination may map to the same frame (frame = page mod N):
  // copy page-wise through a bounce buffer; the destination is marked dirty.
  ppmd_page::kopiere( (unsigned char*)Dest, (const unsigned char*)Src, 12ULL*NU );
}

void* ExpandUnits( void* OldPtr, uint OldNU ) {
  uint i0 = Units2Indx[OldNU-1];
  uint i1 = Units2Indx[OldNU-1+1];
  if( i0==i1 ) return OldPtr;
  void* ptr = AllocUnits(OldNU+1);
  if( ptr ) {
    UnitsCpy( ptr, OldPtr, OldNU );
    insert( &BList[i0], OldPtr, OldNU );
  }
  return ptr;
}

void* ShrinkUnits( void* OldPtr, uint OldNU, uint NewNU ) {
  uint i0 = Units2Indx[OldNU-1];
  uint i1 = Units2Indx[NewNU-1];
  if( i0==i1 ) return OldPtr;
  if( BList[i1].avail() ) {
    void* ptr = remove(&BList[i1]);
    UnitsCpy( ptr, OldPtr, NewNU );
    insert( &BList[i0], OldPtr, Indx2Units[i0] );
    return ptr;
  } else {
    SplitBlock(OldPtr,i0,i1);
    return OldPtr;
  }
}

void* MoveUnitsUp( void* OldPtr, uint NU ) {
  uint indx = Units2Indx[NU-1];
  PrefetchData(OldPtr);
  if( (byte*)OldPtr > UnitsStart+128*1024 ||
      pBLK_NODE(OldPtr) > getNext(&BList[indx]) ) return OldPtr;

  void* ptr = remove(&BList[indx]);
  UnitsCpy( ptr, OldPtr, NU );

  insert( &BList[N_INDEXES], OldPtr, Indx2Units[indx] );

  return ptr;
}

void PrepareTextArea() {
  AuxUnit = (byte*)AllocContext();
  if( !AuxUnit ) {
    AuxUnit = UnitsStart;
  } else {
    if( AuxUnit==UnitsStart) AuxUnit = (UnitsStart+=UNIT_SIZE);
  }
}

void ExpandTextArea() {
  pBLK_NODE  p;
  uint Count[N_INDEXES], i=0;
  memset( Count, 0, sizeof(Count) );

  if( AuxUnit!=UnitsStart ) {
    // four bytes through a byte pointer: translate the length explicitly
    if( *(const uint*)ppmd_page::xl_a((unsigned char*)(void*)AuxUnit, 4) != ~uint(0) )
      UnitsStart += UNIT_SIZE;
    else
      insert( BList, AuxUnit, 1 );
  }

  while( (p=(pBLK_NODE)UnitsStart)->Stamp == ~uint(0) ) {
    pMEM_BLK  pm = (pMEM_BLK)p;
    UnitsStart = (byte*)(pm + pm->NU);
    Count[Units2Indx[pm->NU-1]]++;
    i++;
    PAGE_W(pm)->Stamp = 0;
  }

  if( i ) {

    for( p=BList+N_INDEXES; p->NextIndx; p=getNext(p) ) {
      while( p->NextIndx && !getNext(p)->Stamp ) {
        Count[Units2Indx[((pMEM_BLK)getNext(p))->NU-1]]--;
        unlink(p);
        BList[N_INDEXES].Stamp--;
      }
      if( !p->NextIndx ) break;
    }

    for( i=0; i<N_INDEXES; i++ ) {
      for( p=BList+i; Count[i]!=0; p=getNext(p) ) {
        while( !getNext(p)->Stamp ) {
          unlink(p); BList[i].Stamp--;
          if ( !--Count[i] ) break;
        }
      }
    }

  }

}

static const int MAX_O=ORealMAX;

template <class T>
  T CLAMP( const T& X, const T& LoX, const T& HiX ) { return (X >= LoX)?((X <= HiX)?(X):(HiX)):(LoX); }

template <class T>
  void SWAP( T& t1, T& t2 ) { T tmp=t1; t1=t2; t2=tmp; }

void PrefetchData(void* Addr) {
  // prefetching loads the page into the pool
  *(volatile const byte*)ppmd_page::xl_a((unsigned char*)Addr, 1);
}

enum {
  UP_FREQ = 5
};

byte Indx2Units[N_INDEXES];
byte Units2Indx[128];

byte NS2BSIndx[256];
byte QTable[260];

void PPMD_STARTUP( void ) {
  int i, k, m, Step;

  for( i=0,k=1; i<N1 ;i++,k+=1 ) Indx2Units[i]=k;
  for( k++; i<N1+N2 ;i++,k+=2 ) Indx2Units[i]=k;
  for( k++; i<N1+N2+N3 ;i++,k+=3 ) Indx2Units[i]=k;
  for( k++; i<N1+N2+N3+N4;i++,k+=4 ) Indx2Units[i]=k;

  for( k=0,i=0; k<128; k++ ) {
    i += Indx2Units[i]<k+1;
    Units2Indx[k]=i;
  }

  NS2BSIndx[0] = 2*0;
  NS2BSIndx[1] = 2*1;
  NS2BSIndx[2] = 2*1;
  memset(NS2BSIndx+3, 2*2, 26);
  memset(NS2BSIndx+29, 2*3, 256-29);

  for( i=0; i<UP_FREQ; i++ ) QTable[i]=i;

  for( m=i=UP_FREQ, k=Step=1; i<260; i++ ) {
    QTable[i] = m;
    if( !--k ) k = ++Step, m++;
  }
}

enum {
  MAX_FREQ = 124,
  O_BOUND = 9
};

struct PPM_CONTEXT;

struct STATE {
  byte Symbol;
  byte Freq;
  uint iSuccessor;
};

typedef ppmd_page::hzg<STATE,0>       pSTATE;  // always in the heap
typedef ppmd_page::hzg<PPM_CONTEXT,0> pPPMC;   // always in the heap
// hSTATE/hPPMC: the proxies of the hot functions (processSymbol2_T,
// processSymbol2, UpdateModel).  Form H inlines
// the fast path there and calls it out of line everywhere else.
typedef ppmd_page::hzg<STATE,0,1>       hSTATE;   // inlined
typedef ppmd_page::hzg<PPM_CONTEXT,0,1> hPPMC;
// The eight functions that take a context (rescale, cutOff, processBinSymbol,
// processSymbol1/2 and their _T forms) take the context proxy by value instead
// of the reference parameter of the original code; the body works on the
// pointer q.
#  define CTX_PARAM(q)   pPPMC q
#  define CTX_BIND(q)
#  define CTX_ARG(x)   (x)
#  define PTR_IS_NULL(p)   (!(p))
// The one-symbol state lives INSIDE the context (it overlays SummFreq).
// Expressed as an ADDRESS, so it becomes a CANONICAL proxy and not a frame
// address that the next miss could make stale.
#define ONESTATE(c) ((pSTATE)((byte*)(c) + 2))

pPPMC  getSucc( pSTATE  This ) {
  return (pPPMC)Indx2Ptr( This->iSuccessor );
}

void SWAP( STATE& s1, STATE& s2 ) {
  word t1 = (word&)s1;
  uint t2 = s1.iSuccessor;
  (word&)s1 = (word&)s2;
  s1.iSuccessor = s2.iSuccessor;
  (word&)s2 = t1;
  s2.iSuccessor = t2;
}

struct PPM_CONTEXT {

  byte NumStats;
  byte Flags;
  word SummFreq;
  uint iStats;
  uint iSuffix;

  // const-correct: a read reference yields a read-only state
  const STATE& oneState() const { return (const STATE&) SummFreq; }
  STATE& oneState() { return (STATE&) SummFreq; }
};

pSTATE  getStats( pPPMC  This ) { return (pSTATE)Indx2Ptr(This->iStats); }

pPPMC  suff( pPPMC  This ) { return (pPPMC)Indx2Ptr(This->iSuffix); }

  int _MaxOrder, _CutOff, _MMAX;
  uint _filesize;
  bool MetaGetan;
  int OrderFall;

  pSTATE  FoundState;
  pPPMC  MaxContext;

  uint EscCount;
  uint CharMask[256];

  int BSumm;
  int RunLength;
  int InitRL;
  byte LtC1, LtC2;
  int  LtLim;                     // limit of the current position
  int  LtJ;                       // order of MinContext of this byte
  int  LtE;                       // escapes of this byte
  bool LtTextOpen;                // pText is referenced (deficit at entry > 0)
  uint LtIssued;                  // text index referenced after the last step
  bool LtMeta;                    // before the metablock restart: flat limit
  int  LtLimOf( unsigned c1, unsigned c2 ) {
    if( LtMeta ) {
      (void)c1; (void)c2;
      return ( (14) < _MaxOrder ) ? (14) : _MaxOrder;
    }
    int v = ppmd_order_table::kGrenze2[ ppmd_order_table::kKlasse[c1&255u] ][ ppmd_order_table::kKlasse[c2&255u] ];
    if( v > _MaxOrder ) v = _MaxOrder;
    if( v < 2 ) v = 2;
    return v;
  }
  int LtOrd( pPPMC q ) { int n=0; while( q->iSuffix ) { q=suff(q); ++n; } return n; }

enum {
  INT_BITS = 7,
  PERIOD_BITS = 7,
  TOT_BITS = INT_BITS + PERIOD_BITS,
  INTERVAL = 1 << INT_BITS,
  BIN_SCALE = 1 << TOT_BITS,
  ROUND = 16
};

struct SEE2_CONTEXT {
  word Summ;
  byte Shift;
  byte Count;

  void init( uint InitVal ) {
    Shift = PERIOD_BITS-4;
    Summ = InitVal << Shift;
    Count = 7;
  }

  uint getMean() {
    return Summ >> Shift;
  }

  void update() {
    if( --Count==0 ) setShift_rare();
  }

  void setShift_rare() {
    uint i = Summ >> Shift;
    i = PERIOD_BITS - (i>40) - (i>280) - (i>1020);
    if( i<Shift ) { Summ >>= 1; Shift--; } else
    if( i>Shift ) { Summ <<= 1; Shift++; }
    Count = 5 << Shift;
  }
};

  int NumMasked;

pSTATE  rescale( CTX_PARAM(q), int OrderFall, pSTATE  FoundState ) {
  CTX_BIND(q)
  STATE tmp; pSTATE  p; pSTATE  p1;

  PAGE_W(q)->Flags &= 0x14;

  p1 = getStats(q);
  tmp = FoundState[0];
  for( p=FoundState; p!=p1; p-- ) PAGE_WI(p,0)=p[-1];
  PAGE_WI(p1,0) = tmp;

  int of = (OrderFall != 0);
  int a, i;
  int f0 = p->Freq;
  int sf = q->SummFreq;
  int EscFreq = sf-f0;
  PAGE_W(q)->SummFreq = PAGE_W(p)->Freq = (f0+of)>>1;

  for( i=0; i<q->NumStats; i++ ) {
    p++;
    a = p->Freq;
    EscFreq -= a;
    a = (a+of)>>1;
    PAGE_W(p)->Freq = a;
    PAGE_W(q)->SummFreq += a;
    if( a ) PAGE_W(q)->Flags |= 0x08*(p->Symbol>=0x40);
    if( a > p[-1].Freq ) {
      tmp = p[0];
      for( p1=p; tmp.Freq>p1[-1].Freq; p1-- ) PAGE_WI(p1,0)=p1[-1];
      PAGE_WI(p1,0) = tmp;
    }
  }

  if( p->Freq==0 ) {
    for( i=0; p->Freq==0; i++,p-- );
    EscFreq += i;
    a = (q->NumStats+2) >> 1;
    if( (PAGE_W(q)->NumStats-=i)==0 ) {
      tmp = getStats(q)[0];
      tmp.Freq = Min( MAX_FREQ/3, (2*tmp.Freq+EscFreq-1)/EscFreq );
      PAGE_W(q)->Flags &= 0x18;
      FreeUnits( getStats(q), a );
      PAGE_W(q)->oneState() = tmp;
      FoundState = ONESTATE(q);
      return FoundState;
    }
    PAGE_W(q)->iStats = Ptr2Indx( ShrinkUnits(getStats(q),a,(q->NumStats+2)>>1) );
  }

  PAGE_W(q)->SummFreq += (EscFreq+1) >> 1;
  if( OrderFall || (q->Flags & 0x04)==0 ) {
    a = (sf-=EscFreq) - f0;
    a = CLAMP( uint( ( f0*q->SummFreq - sf*getStats(q)->Freq + a-1 ) / a ), 2U, MAX_FREQ/2U-18U );
  } else {
    a = 2;
  }

  PAGE_W(FoundState=getStats(q))->Freq += a;
  PAGE_W(q)->SummFreq += a;
  PAGE_W(q)->Flags |= 0x04;

  return FoundState;
}

void AuxCutOff( pSTATE  p, int Order, int MaxOrder ) {
  if( Order<MaxOrder ) {
    PrefetchData( getSucc(p) );
    PAGE_W(p)->iSuccessor = cutOff( CTX_ARG(getSucc(p)), Order+1,MaxOrder);
  } else {
    PAGE_W(p)->iSuccessor=0;
  }
}

uint cutOff( CTX_PARAM(q), int Order, int MaxOrder ) {
  CTX_BIND(q)
  int i, tmp, EscFreq, Scale;
  pSTATE  p;
  pSTATE  p0;

  if( q->NumStats==0 ) {

    int flag = 1;
    p = ONESTATE(q);
    if( (byte*)getSucc(p) >= UnitsStart ) {
      AuxCutOff( p, Order, MaxOrder );
      if( p->iSuccessor || Order<O_BOUND ) flag=0;
    }
    if( flag ) {
      FreeUnit( q );
      return 0;
    }

  } else {

    tmp = (q->NumStats+2)>>1;
    p0 = (pSTATE)MoveUnitsUp(getStats(q),tmp);
    PAGE_W(q)->iStats = Ptr2Indx(p0);

    for( i=q->NumStats, p=p0+i; p>=p0; p-- ) {
      if( (byte*)getSucc(p) < UnitsStart ) {
        PAGE_WI(p,0).iSuccessor=0;
        SWAP( PAGE_WI(p,0), PAGE_WI(p0,i--) );
      } else AuxCutOff( p, Order, MaxOrder );
    }

    if( i!=q->NumStats && Order>0 ) {
      PAGE_W(q)->NumStats = i;
      p = p0;
      if( i<0 ) {
        FreeUnits( p, tmp );
        FreeUnit( q );
        return 0;
      }
      if( i==0 ) {
        PAGE_W(q)->Flags = (q->Flags & 0x10) + 0x08*(p[0].Symbol>=0x40);
        PAGE_WI(p,0).Freq = 1+(2*(p[0].Freq-1))/(q->SummFreq-p[0].Freq);
      { STATE state_tmp = p[0]; PAGE_W(q)->oneState() = state_tmp; }
        FreeUnits( p, tmp );
      } else {
        p = (pSTATE)ShrinkUnits( p0, tmp, (i+2)>>1 );
        PAGE_W(q)->iStats = Ptr2Indx(p);
        Scale = (q->SummFreq>16*i);
        PAGE_W(q)->Flags = (q->Flags & (0x10+0x04*Scale));
        if( Scale ) {
          EscFreq = q->SummFreq;
          PAGE_W(q)->SummFreq = 0;
          for( i=0; i<=q->NumStats; i++ ) {
            EscFreq -= p[i].Freq;
            PAGE_WI(p,i).Freq = (p[i].Freq+1)>>1;
            PAGE_W(q)->SummFreq += p[i].Freq;
            PAGE_W(q)->Flags |= 0x08*(p[i].Symbol>=0x40);
          };
          EscFreq = (EscFreq+1)>>1;
          PAGE_W(q)->SummFreq += EscFreq;
        } else {
          for( i=0; i<=q->NumStats; i++ ) PAGE_W(q)->Flags |= 0x08*(p[i].Symbol>=0x40);
        }
      }
    }

  }

  if( (byte*)q==UnitsStart ) {

    UnitsCpy( AuxUnit, q, 1 );
    return Ptr2Indx(AuxUnit);
  } else {

    if( (byte*)suff(q)==UnitsStart ) PAGE_W(q)->iSuffix=Ptr2Indx(AuxUnit);
  }

  return Ptr2Indx(q);
}

void StartModelRare( void ) {
  int i, k, s;
  byte i2f[25];

  memset( CharMask, 0, sizeof(CharMask) );
  EscCount=1;

  if( _MaxOrder<2 ) {
    OrderFall = _MaxOrder;
    for( pPPMC  pc=MaxContext; pc->iSuffix!=0; pc=suff(pc) ) OrderFall--;
    return;
  }

  LtLim = LtLimOf( LtC1, LtC2 );
  OrderFall = LtLim;

  InitSubAllocator();

  InitRL = -( (LtLim<13) ? LtLim : 13 );
  LtIssued = 0;
  RunLength = InitRL;

  MaxContext = (pPPMC)AllocContext();
  PAGE_W(MaxContext)->NumStats = 255;
  PAGE_W(MaxContext)->SummFreq = 255+2;
  PAGE_W(MaxContext)->iStats = Ptr2Indx(AllocUnits(256/2));
  PAGE_W(MaxContext)->Flags = 0;
  PAGE_W(MaxContext)->iSuffix = 0;
  PrevSuccess = 0;

  for( i=0; i<256; i++ ) {
    PAGE_WI(getStats(MaxContext),i).Symbol = i;
    PAGE_WI(getStats(MaxContext),i).Freq = 1;
    PAGE_WI(getStats(MaxContext),i).iSuccessor = 0;
  }

  if( 1 ) {

    for( k=i=0; i<25; i2f[i++]=k+1 ) while( QTable[k]==i ) k++;

    for( k=0; k<64; k++ ) {
      for( s=i=0; i<6; i++ ) s += EscCoef[2*i+((k>>i)&1)];
      s = 128*CLAMP( s, 32, 256-32 );
      for( i=0; i<25; i++ ) BinSumm[i][k] = BIN_SCALE - s/i2f[i];
    }

    for( i=0; i<23; i++ ) for( k=0; k<32; k++ ) SEE2Cont[i][k].init(8*i+5);
  }
}

void RestoreModelRare( void ) {
  pSTATE  p;
  pText = HeapStart;
  pPPMC  pc = saved_pc;

  for(;; MaxContext=suff(MaxContext) ) {
   if( (MaxContext->NumStats==1) && (MaxContext!=pc) ) {
     p = getStats(MaxContext);
     if( (byte*)(getSucc(p+1))>=UnitsStart ) break;
   } else break;

    PAGE_W(MaxContext)->Flags = (MaxContext->Flags & 0x10) + 0x08*(p->Symbol>=0x40);
    PAGE_WI(p,0).Freq = (p[0].Freq+1) >> 1;
    { STATE state_tmp = p[0]; PAGE_W(MaxContext)->oneState() = state_tmp; }
    PAGE_W(MaxContext)->NumStats=0;
    FreeUnits( p, 1 );
  }

  while( MaxContext->iSuffix ) MaxContext=suff(MaxContext);

  AuxUnit = UnitsStart;

  ExpandTextArea();

  do {
    PrepareTextArea();
    cutOff( CTX_ARG(MaxContext), 0, _MaxOrder );
    ExpandTextArea();
  } while( GetUsedMemory()>3*(SubAllocatorSize>>2) );

  GlueCount = GlueCount1 = 0;
  OrderFall = LtLim; LtIssued = 0;
}

pPPMC  saved_pc;

pPPMC  UpdateModel( pPPMC  MinContext ) {
  byte Flag, FSymbol;
  uint ns1, ns, cf, sf, s0, FFreq;
  uint iSuccessor, iFSuccessor;
  hPPMC  pc = MaxContext;
  hSTATE  p = NULL;

  FSymbol = FoundState->Symbol;
  FFreq = FoundState->Freq;
  iFSuccessor = FoundState->iSuccessor;

  if( MinContext->iSuffix ) {
    pc = suff(MinContext);

    if( pc[0].NumStats ) {
      p = getStats(pc);
      if( p[0].Symbol!=FSymbol ) {
        for( p++; p[0].Symbol!=FSymbol; p++ );
        if( p[0].Freq >= p[-1].Freq ) SWAP( PAGE_WI(p,0), PAGE_WI(p,-1) ), p--;
      }
      if( p[0].Freq<MAX_FREQ-3 ) {
        cf = 2 + (FFreq<28);
        PAGE_WI(p,0).Freq += cf;
        PAGE_WI(pc,0).SummFreq += cf;
      }
    } else {
      p = ONESTATE(pc);
      PAGE_WI(p,0).Freq += (p[0].Freq<14);
    }
  }
  // pc = MaxContext;

  if( LtE==0 && !LtTextOpen && OrderFall<=1 && iFSuccessor ) {
    PAGE_W(FoundState)->iSuccessor = LtCreateSuccessors( p, MinContext, LtJ, LtLim );
    OrderFall = 0;
    if( !FoundState->iSuccessor ) { saved_pc=pc; return 0; };
    MaxContext = getSucc(FoundState);
    return MaxContext;
  }

  *PAGE_W(pText++) = FSymbol;
  iSuccessor = Ptr2Indx(pText);
  if( pText>=UnitsStart ) { saved_pc=pc; return 0; };

  if( iFSuccessor ) {
    if( (byte*)Indx2Ptr(iFSuccessor) < UnitsStart )
     iFSuccessor = LtCreateSuccessors( p, MinContext, LtJ, LtLim );
    else
     PrefetchData( Indx2Ptr(iFSuccessor) );
  } else
    iFSuccessor = LtReduceOrder( p, MinContext );

  if( !iFSuccessor ) { saved_pc=pc; return 0; };

  // Order limit: deficit against the NEXT limit; <= 0 is a cut (target
  // below j+1) and counts as "limit reached".  The written byte is taken back
  // only if nobody references it (as without the limit: e > 0 <=> text closed).
  if( --OrderFall <= 0 ) {
    OrderFall = 0;
    iSuccessor = iFSuccessor;
    if( !LtTextOpen ) pText--;
  }

  s0 = MinContext->SummFreq - FFreq;
  ns = MinContext->NumStats;
  Flag = 0x08*(FSymbol>=0x40);
  for( pc=MaxContext; pc!=MinContext; pc=suff(pc) ) {
    ns1 = pc[0].NumStats;

    if( ns1 ) {

      if( ns1&1 ) {
        p = (hSTATE)ExpandUnits( getStats(pc),(ns1+1)>>1 );
        if( !p ) { saved_pc=pc; return 0; };
        PAGE_WI(pc,0).iStats = Ptr2Indx(p);
      }

      PAGE_WI(pc,0).SummFreq += QTable[ns+4] >> 3;
    } else {

      p = (hSTATE)AllocUnits(1);
      if( !p ) { saved_pc=pc; return 0; };
      { STATE state_tmp = pc[0].oneState(); PAGE_WI(p,0) = state_tmp; }
      PAGE_WI(pc,0).iStats = Ptr2Indx(p);
      PAGE_WI(p,0).Freq = (p[0].Freq<=MAX_FREQ/3) ? (2*p[0].Freq-1) : (MAX_FREQ-15);

      PAGE_WI(pc,0).SummFreq = p[0].Freq + (ns>1) + ExpEscape[QTable[BSumm>>8]];
    }

    cf = (FFreq-1)*(5 + pc[0].SummFreq);
    sf = s0 + pc[0].SummFreq;

    if( cf<=3*sf ) {

      cf = 1 + (2*cf>sf) + (2*cf>3*sf);
      PAGE_WI(pc,0).SummFreq += 4;
    } else {
      cf = 5 + (cf>5*sf) + (cf>6*sf) + (cf>8*sf) + (cf>10*sf) + (cf>12*sf);
      PAGE_WI(pc,0).SummFreq += cf;
    }

    p = getStats(pc) + (++PAGE_WI(pc,0).NumStats);
    PAGE_WI(p,0).iSuccessor = iSuccessor;
    PAGE_WI(p,0).Symbol = FSymbol;
    PAGE_WI(p,0).Freq = cf;
    PAGE_WI(pc,0).Flags |= Flag;
  }

  MaxContext = (hPPMC)Indx2Ptr(iFSuccessor);
  return MaxContext;
}

uint CreateSuccessors( uint Skip, pSTATE  p, pPPMC  pc ) {
  byte tmp;
  uint cf, s0;
  pSTATE  ps[MAX_O];
  pSTATE * pps=ps;

  byte sym = FoundState->Symbol;
  uint iUpBranch = FoundState->iSuccessor;

  if( !Skip ) {
    *pps++ = FoundState;
    if( !pc[0].iSuffix ) goto NO_LOOP;
  }

  if( p ) { pc = suff(pc); goto LOOP_ENTRY; }

  do {
    pc = suff(pc);

    if( pc[0].NumStats ) {

      for( p=getStats(pc); p[0].Symbol!=sym; p++ );

      tmp = 2*(p[0].Freq<MAX_FREQ-1);
      PAGE_WI(p,0).Freq += tmp;
      PAGE_WI(pc,0).SummFreq += tmp;
    } else {

      p = ONESTATE(pc);
      PAGE_WI(p,0).Freq += (!suff(pc)->NumStats & (p[0].Freq<16));
    }

LOOP_ENTRY:
    if( p[0].iSuccessor!=iUpBranch ) {
      pc = getSucc(p);
      break;
    }
    *pps++ = p;
  } while ( pc[0].iSuffix );

NO_LOOP:
  if( pps==ps ) return Ptr2Indx(pc);

  PPM_CONTEXT ct;
  ct.NumStats = 0;
  ct.Flags = 0x10*(sym>=0x40);
  sym = *ppmd_page::xl_a((unsigned char*)Indx2Ptr(iUpBranch), 1);
  ct.oneState().iSuccessor = Ptr2Indx((byte*)Indx2Ptr(iUpBranch)+1);
  ct.oneState().Symbol = sym;
  ct.Flags |= 0x08*(sym>=0x40);

  if( pc[0].NumStats ) {
    for( p=getStats(pc); p[0].Symbol!=sym; p++ );
    cf = p[0].Freq - 1;
    s0 = pc[0].SummFreq - pc[0].NumStats - cf;
    cf = 1 + ((2*cf<s0) ? (12*cf>s0) : 2+cf/s0);
    ct.oneState().Freq = Min<uint>( 7, cf );
  } else {
    ct.oneState().Freq = pc[0].oneState().Freq;
  }

  do {
    pPPMC  pc1 = (pPPMC)AllocContext();
    if( !pc1 ) return 0;
    ((uint*)PAGE_RAW_W(pc1))[0] = ((uint*)&ct)[0];
    ((uint*)PAGE_RAW_W(pc1))[1] = ((uint*)&ct)[1];
    PAGE_W(pc1)->iSuffix = Ptr2Indx(pc);
    pc = pc1; pps--;
    PAGE_WI(pps[0],0).iSuccessor = Ptr2Indx(pc);
  } while( pps!=ps );

  return Ptr2Indx(pc);
}

uint ReduceOrder( pSTATE  p, pPPMC  pc ) {
  byte tmp;
  pSTATE  p1;
  pPPMC  pc1=pc;
  PAGE_W(FoundState)->iSuccessor = Ptr2Indx(pText);
  byte sym = FoundState->Symbol;
  uint iUpBranch = FoundState->iSuccessor;
  OrderFall++;

  if( p ) { pc=suff(pc); goto LOOP_ENTRY; }

  while(1) {
    if( !pc->iSuffix ) return Ptr2Indx(pc);
    pc = suff(pc);

    if( pc->NumStats ) {
      for( p=getStats(pc); p[0].Symbol!=sym; p++ );
      tmp = 2*(p->Freq<MAX_FREQ-3);
      PAGE_W(p)->Freq += tmp;
      PAGE_W(pc)->SummFreq += tmp;
    } else {
      p = ONESTATE(pc);
      PAGE_W(p)->Freq += (p->Freq<11);
    }

LOOP_ENTRY:
    if( p->iSuccessor ) break;
    PAGE_W(p)->iSuccessor = iUpBranch;
    OrderFall++;
  }

  if( p->iSuccessor<=iUpBranch ) {
    p1 = FoundState;
    FoundState = p;
    PAGE_W(p)->iSuccessor = CreateSuccessors(0,0,pc);
    FoundState = p1;
  }

  if( OrderFall==1 && pc1==MaxContext ) {
    PAGE_W(FoundState)->iSuccessor = p->iSuccessor;
    pText--;
  }

  return p->iSuccessor;
}

uint LtCreateSuccessors( pSTATE p, pPPMC pc, int jF, int cap ) {
  byte tmp;
  uint cf, s0;
  pSTATE ps[MAX_O];
  pSTATE * pps=ps;
  byte sym = FoundState->Symbol;
  uint iUpBranch = FoundState->iSuccessor;

  *pps++ = FoundState;
  if( !pc[0].iSuffix ) goto NO_LOOP;

  if( p ) { pc = suff(pc); goto LOOP_ENTRY; }

  do {
    pc = suff(pc);

    if( pc[0].NumStats ) {
      for( p=getStats(pc); p[0].Symbol!=sym; p++ );
      tmp = 2*(p[0].Freq<MAX_FREQ-1);
      PAGE_WI(p,0).Freq += tmp;
      PAGE_WI(pc,0).SummFreq += tmp;
    } else {
      p = ONESTATE(pc);
      PAGE_WI(p,0).Freq += (!suff(pc)->NumStats & (p[0].Freq<16));
    }

LOOP_ENTRY:
    if( p[0].iSuccessor!=iUpBranch ) {
      pc = getSucc(p);
      break;
    }
    *pps++ = p;
  } while ( pc[0].iSuffix );

NO_LOOP:
  {
  const int n = (int)(pps-ps);
  const int lowest = jF - n + 1;
  const int topNew = (jF < cap-1) ? jF : cap-1;
  int nNew = topNew - lowest + 1;
  if( nNew < 0 ) nNew = 0;
  if( nNew ) {
    PPM_CONTEXT ct;
    ct.NumStats = 0;
    ct.Flags = 0x10*(sym>=0x40);
    sym = *ppmd_page::xl_a((unsigned char*)Indx2Ptr(iUpBranch), 1);
    ct.oneState().iSuccessor = Ptr2Indx((byte*)Indx2Ptr(iUpBranch)+1);
    ct.oneState().Symbol = sym;
    ct.Flags |= 0x08*(sym>=0x40);

    if( pc[0].NumStats ) {
      for( p=getStats(pc); p[0].Symbol!=sym; p++ );
      cf = p[0].Freq - 1;
      s0 = pc[0].SummFreq - pc[0].NumStats - cf;
      cf = 1 + ((2*cf<s0) ? (12*cf>s0) : 2+cf/s0);
      ct.oneState().Freq = Min<uint>( 7, cf );
    } else {
      ct.oneState().Freq = pc[0].oneState().Freq;
    }

    for( int k=0; k<nNew; k++ ) {
      pPPMC  pc1 = (pPPMC)AllocContext();
      if( !pc1 ) return 0;
      ((uint*)PAGE_RAW_W(pc1))[0] = ((uint*)&ct)[0];
      ((uint*)PAGE_RAW_W(pc1))[1] = ((uint*)&ct)[1];
      PAGE_W(pc1)->iSuffix = Ptr2Indx(pc);
      pc = pc1; pps--;
      PAGE_WI(pps[0],0).iSuccessor = Ptr2Indx(pc);
    }
  }
  while( pps!=ps ) { pps--; PAGE_WI(pps[0],0).iSuccessor = Ptr2Indx(pc); }
  }
  return Ptr2Indx(pc);
}

// ReduceOrder with the cap: the original's special case ("OrderFall==1 &&
// pc1==MaxContext") becomes "deficit after this step <= 0", and then EVERY
// state that got the text anchor here is redirected; the take-back of the
// written byte is done once in UpdateModel.
uint LtReduceOrder( pSTATE  p, pPPMC  pc ) {
  byte tmp;
  pSTATE  p1;
  pSTATE  zs[MAX_O]; int nz = 0;
  PAGE_W(FoundState)->iSuccessor = Ptr2Indx(pText);
  byte sym = FoundState->Symbol;
  uint iUpBranch = FoundState->iSuccessor;
  int jp = LtJ - 1;          // order of the state p (below FoundState)
  OrderFall++;

  if( p ) { pc=suff(pc); goto LOOP_ENTRY; }

  while(1) {
    if( !pc->iSuffix ) return Ptr2Indx(pc);
    pc = suff(pc);

    if( pc->NumStats ) {
      for( p=getStats(pc); p[0].Symbol!=sym; p++ );
      tmp = 2*(p->Freq<MAX_FREQ-3);
      PAGE_W(p)->Freq += tmp;
      PAGE_W(pc)->SummFreq += tmp;
    } else {
      p = ONESTATE(pc);
      PAGE_W(p)->Freq += (p->Freq<11);
    }

LOOP_ENTRY:
    if( p->iSuccessor ) break;
    PAGE_W(p)->iSuccessor = iUpBranch;
    zs[nz++] = p;
    --jp;
    OrderFall++;
  }

  if( p->iSuccessor<=iUpBranch ) {
    p1 = FoundState;
    FoundState = p;
    PAGE_W(p)->iSuccessor = LtCreateSuccessors(0, pc, jp, LtLim);
    FoundState = p1;
  }

  if( OrderFall<=1 ) {
    PAGE_W(FoundState)->iSuccessor = p->iSuccessor;
    for( int k=0; k<nz; k++ ) PAGE_W(zs[k])->iSuccessor = p->iSuccessor;
  }

  return p->iSuccessor;
}

int PrevSuccess;
word BinSumm[25][64];

template< int ProcMode >
void processBinSymbol( CTX_PARAM(q), int symbol ) {
  CTX_BIND(q)
  pSTATE rs = ONESTATE(q);
  int i = NS2BSIndx[suff(q)->NumStats] + PrevSuccess + q->Flags + ((RunLength>>26) & 0x20);
  word& bs = BinSumm[QTable[rs->Freq-1]][i];
  BSumm = bs;
  bs -= (BSumm+64) >> PERIOD_BITS;

  int flag = ProcMode ? 0 : rs->Symbol!=symbol;

  if( flag ) {
    CharMask[rs->Symbol] = EscCount;
    NumMasked = 0;
    PrevSuccess = 0;
    FoundState = 0;
  } else {
    bs += INTERVAL;
    PAGE_W(rs)->Freq += (rs->Freq<196);
    RunLength++;
    PrevSuccess = 1;
    FoundState = rs;
  }
}

template< int ProcMode >
void processSymbol1( CTX_PARAM(q), int symbol ) {
  CTX_BIND(q)
  pSTATE  p = getStats(q);

  int cnum = q->NumStats;
  int i = p[0].Symbol;
  int low = 0;
  int freq = p[0].Freq;
  int total = q->SummFreq;
  int flag;

  int count=0;

  if( ProcMode ) {

    flag = count<freq;
  } else {
    flag = i==symbol;
  }

  if( flag ) {

    PrevSuccess = 0;
    PAGE_WI(p,0).Freq += 4;
    PAGE_W(q)->SummFreq += 4;

  } else {

    PrevSuccess = 0;

    for( low=freq,i=1; i<=cnum; i++ ) {
      freq = p[i].Freq;
      flag = ProcMode ? low+freq>count : p[i].Symbol==symbol;
      if( flag ) break;
      low += freq;
    }

    if( flag ) {
      PAGE_WI(p,i).Freq += 4;
      PAGE_W(q)->SummFreq += 4;
      if( p[i].Freq > p[i-1].Freq ) SWAP( PAGE_WI(p,i), PAGE_WI(p,i-1) ), i--;
      p = p+i;
    } else {
      if( q->iSuffix ) PrefetchData( suff(q) );
      freq = total-low;
      NumMasked = cnum;
      for( i=0; i<=cnum; i++ ) CharMask[p[i].Symbol]=EscCount;
      p = NULL;
    }
  }

  FoundState = p;
  if( p && (p[0].Freq>MAX_FREQ) ) FoundState=rescale(CTX_ARG(q),OrderFall,FoundState);
}

SEE2_CONTEXT SEE2Cont[23][32];
SEE2_CONTEXT DummySEE2Cont;

template< int ProcMode >
void processSymbol2( CTX_PARAM(q), int symbol ) {
  CTX_BIND(q)
  byte px[256];
  hSTATE  p = getStats(q);

  int c;
  int count=0;
  int low;
  int see_freq;
  int freq;
  int cnum = q->NumStats;

  SEE2_CONTEXT* psee2c;
  if( cnum != 0xFF ) {
    psee2c = SEE2Cont[ QTable[cnum+3]-4 ];
    psee2c+= (q->SummFreq > 10*(cnum+1));
    psee2c+= 2*(2*cnum < suff(q)->NumStats+NumMasked) + q->Flags;
    see_freq = psee2c->getMean()+1;

  } else {
    psee2c = &DummySEE2Cont;
    see_freq = 1;
  }

  int flag=0,pl;

  int i,j;
  for( i=0,j=0,low=0; i<=cnum; i++ ) {
    c = p[i].Symbol;
    if( CharMask[c]!=EscCount ) {
      CharMask[c]=EscCount;
      low += p[i].Freq;
      if( ProcMode )
        px[j++] = i;
      else
        if( c==symbol ) flag=1,j=i,pl=low;
    }
  }

  int Total = see_freq + low;

  if( ProcMode ) {

    flag = count<low;
  }

  if( flag ) {
    if( ProcMode ) {
      for( low=0, i=0; (low+=p[j=px[i]].Freq)<=count; i++ );
    } else {
      low = pl;
    }
    p+=j;

    freq = p[0].Freq;

    if( see_freq>2 ) psee2c->Summ -= see_freq;
    psee2c->update();

    FoundState = p;
    PAGE_WI(p,0).Freq += 4;
    PAGE_W(q)->SummFreq += 4; if( p[0].Freq > MAX_FREQ ) FoundState=rescale(CTX_ARG(q),OrderFall,FoundState);
    RunLength = -( (LtLim<13) ? LtLim : 13 );
    EscCount++;
  } else {
    low = Total;
    freq = see_freq;
    NumMasked = cnum;
    psee2c->Summ += Total-see_freq;

  }

}

struct qsym {
  word sym;
  word freq;
  word total;

  void store( uint _sym, uint _freq, uint _total ) {
    sym=_sym; freq=_freq; total=_total;
  }

};

qsym SQ[1024];
uint SQ_ptr;

uint sqp[256];

uint trF[256];
uint trT[256];

void ConvertSQ( void ) {
  uint i,c,j,b,freq,total,prob,cnum;
  uint cum = 0xFFFFFF00;

  cnum=256;
  memset(sqp, 0, sizeof(sqp));
  trF[0]=0; trT[0]=0;
  // for( i=0; i<256; i++ ) sqp[i]=0,trF[i]=0,trT[i]=0;

  for( i=0; i<SQ_ptr; i++ ) {
    c = SQ[i].sym; freq = SQ[i].freq; total = SQ[i].total;
    prob = qword(qword(cum)*freq)/total;
    if( c<256 ) {
      sqp[c] = prob+1; cnum--;
    } else {
      cum = prob;
    }
  }

  {
    uint T[512];
    for( c=0; c<256; c++ ) {
      T[256+c] = sqp[c] ? sqp[c] : 1u;
    }
    for( j=255; j!=0; j-- ) {
      const uint l = T[2*j], r = T[2*j+1];
      T[j]  = l + r;
      trT[j]= l + r;
      trF[j]= l;
    }
  }
  (void)b;

}

void processBinSymbol_T( CTX_PARAM(q), int ) {
  CTX_BIND(q)
  pSTATE rs = ONESTATE(q);
  int i = NS2BSIndx[suff(q)->NumStats] + PrevSuccess + q->Flags + ((RunLength>>26) & 0x20);
  word& bs = BinSumm[QTable[rs->Freq-1]][i];
  BSumm = bs;

  SQ[SQ_ptr++].store(rs->Symbol,BSumm+BSumm,SCALE);
  SQ[SQ_ptr++].store(256,SCALE-BSumm-BSumm,SCALE);

  CharMask[rs->Symbol] = EscCount;
  NumMasked = 0;
}

void processSymbol1_T( CTX_PARAM(q), int ) {
  CTX_BIND(q)
  pSTATE  p = getStats(q);

  int cnum = q->NumStats;
  int low = 0;
  int freq = 0;
  int total = q->SummFreq;
  int i;

  for( i=0,low=0; i<=cnum; i++ ) {
    freq = p[i].Freq;
    SQ[SQ_ptr++].store(p[i].Symbol,freq,total);
    low += freq;
  }

  if( q->iSuffix ) PrefetchData( suff(q) );

  NumMasked = cnum;
  for( i=0; i<=cnum; i++ ) CharMask[p[i].Symbol]=EscCount;

  SQ[SQ_ptr++].store(256,total-low,total);
}

void processSymbol2_T( CTX_PARAM(q), int ) {
  CTX_BIND(q)
  hSTATE  p = getStats(q);

  int c;
  int low;
  int see_freq;
  int cnum = q->NumStats;

  SEE2_CONTEXT* psee2c;
  if( cnum != 0xFF ) {
    psee2c = SEE2Cont[ QTable[cnum+3]-4 ];
    psee2c+= (q->SummFreq > 10*(cnum+1));
    psee2c+= 2*(2*cnum < suff(q)->NumStats+NumMasked) + q->Flags;
    see_freq = psee2c->getMean()+1;

  } else {
    psee2c = &DummySEE2Cont;
    see_freq = 1;
  }

  int i;
  for( i=0,low=0; i<=cnum; i++ ) {
    c = p[i].Symbol;
    if( CharMask[c]!=EscCount ) low += p[i].Freq;
  }
  int Total = see_freq + low;

  for( i=0; i<=cnum; i++ ) {
    c = p[i].Symbol;
    if( CharMask[c]!=EscCount ) {
      SQ[SQ_ptr++].store(c,p[i].Freq,Total);
      CharMask[c]=EscCount;
    }
  }

  SQ[SQ_ptr++].store(256,see_freq,Total);
  NumMasked = cnum;
}

  uint cxt;
  uint y;

  uint Init( uint MaxOrder, uint MMAX, uint CutOff, uint filesize ) {
    _MaxOrder = MaxOrder;
    MetaGetan = false;
    LtC1 = LtC2 = 0;
    LtMeta = true;
    LtLim = LtLimOf( 0, 0 ); LtJ = 0; LtE = 0;
    LtTextOpen = false; LtIssued = 0;
    _CutOff = CutOff;
    _MMAX = MMAX;
    _filesize = filesize;

    PPMD_STARTUP();

    if( !StartSubAllocator( _MMAX ) ) return 1;

    StartModelRare();

    cxt=0; y=1;

    return 0;
  }

  ~ppmd_Model() {
    //StopSubAllocator();
  }

void ppmd_PrepareByte( void ) {
    ppmd_page::dyn_tick();
    if( ppmd_page::V.z_versatz | ppmd_page::V.z_basis ) {
      const unsigned long long g = ppmd_page::V.z_grenze << 12;
      if( (unsigned long long)(LoUnit-HeapStart) + ppmd_page::ZONEN_RAND >= g
          || (unsigned long long)(HiUnit-HeapStart) < g + ppmd_page::ZONEN_RAND )
        ppmd_page::zonen_aus();
    }
  SQ_ptr=0; NumMasked=0;
  int _OrderFall = OrderFall;

  pPPMC  MinContext = MaxContext;
  if( MinContext->NumStats ) {
    processSymbol1_T( CTX_ARG(MinContext), 0 );
  } else {
    processBinSymbol_T( CTX_ARG(MinContext), 0 );
  }

  while(1) {
    do {
      if( !MinContext->iSuffix ) goto Break;
      OrderFall++;
      MinContext = suff(MinContext);
    } while( MinContext->NumStats==NumMasked );
    processSymbol2_T( CTX_ARG(MinContext), 0 );
  }

  Break:
  EscCount++; NumMasked=0; OrderFall=_OrderFall;

  ConvertSQ();
}

void ppmd_UpdateByte( uint c ) {
  const int lt_f0 = OrderFall;
  pPPMC  MinContext = MaxContext;
  if( MinContext->NumStats ) {
    processSymbol1<0>( CTX_ARG(MinContext), c );
  } else {
    processBinSymbol<0>( CTX_ARG(MinContext), c );
  }

  while( !FoundState ) {
    do {

      OrderFall++;
      MinContext = suff(MinContext);
    } while( MinContext->NumStats==NumMasked );
    processSymbol2<0>( CTX_ARG(MinContext), c );
  }

  pPPMC  p;
  {
    const int lt_next = LtLimOf( c, LtC1 );
    LtE = OrderFall - lt_f0;
    LtJ = LtLim - OrderFall;              // order of MinContext
    LtTextOpen = (lt_f0 > 0);
    OrderFall += lt_next - LtLim;         // = next limit - j  (<= 0: cut)
    LtLim = lt_next;
  }
  LtC2 = LtC1; LtC1 = (byte)c;
  if( LtE==0 && !LtTextOpen && OrderFall<=1 && (byte*)getSucc(FoundState)>=UnitsStart ) {
    p = MaxContext = getSucc(FoundState);
    OrderFall = 0;
  } else {
    p = UpdateModel( MinContext );
    if( p ) MaxContext = p;
  }
  LtIssued = (!PTR_IS_NULL(p) && OrderFall>0) ? Ptr2Indx(pText) : 0;

  if( PTR_IS_NULL(p) ) {
    if( _CutOff ) {
      RestoreModelRare();
    } else {
      StartModelRare();
    }
  }
}

};

#pragma pack()

unsigned long long counter_ = 0;

PPMD::PPMD(int order, int memory, const unsigned int& bit_context,
    const std::vector<bool>& vocab) : ByteModel(vocab), byte_(bit_context) {
  tree_zero_.fill(0);
  tree_total_.fill(0);
  vocab_full_ = true;
  for (int i = 0; i < 256; ++i) {
    if (!vocab_[i]) {
      disabled_bytes_.push_back(static_cast<unsigned char>(i));
      vocab_full_ = false;
    }
  }
  ppmd_model_.reset(new ppmd_Model());
  ppmd_model_->Init(order,memory,1,0);
}

PPMD::~PPMD() {
  ppmd_model_->ReleaseHeap();
  ppmd_page::schluss();
  ::remove("ppm.page");
}

std::valarray<float>& PPMD::Predict() {
  const unsigned int total = tree_total_[tree_context_];
  if (total == 0) {
    return ByteModel::Predict();
  }
  outputs_[0] = static_cast<float>(total - tree_zero_[tree_context_]) /
      static_cast<float>(total);
  return outputs_;
}

void PPMD::Perceive(int bit) {
  ByteModel::Perceive(bit);
  tree_context_ = (tree_context_ << 1) | static_cast<unsigned int>(bit);
}

void PPMD::ByteUpdate() {
  ++counter_;
  ppmd_model_->MetaSchnitt(counter_);
  ppmd_model_->ppmd_UpdateByte(byte_);
  ppmd_model_->ppmd_PrepareByte();
  for (int i = 0; i < 256; ++i) {
    tree_zero_[i] = ppmd_model_->trF[i];
    tree_total_[i] = ppmd_model_->trT[i];
  }
  if (!vocab_full_) {
    for (unsigned char c : disabled_bytes_) {
      const unsigned int mass = ppmd_model_->sqp[c] ? ppmd_model_->sqp[c] : 1U;
      for (int bit_index = 8; bit_index != 0; --bit_index) {
        const unsigned int node = (256U + c) >> bit_index;
        const unsigned int bit = (c >> (bit_index - 1)) & 1U;
        if (tree_total_[node] >= mass) tree_total_[node] -= mass;
        else tree_total_[node] = 0;
        if (bit == 0) {
          if (tree_zero_[node] >= mass) tree_zero_[node] -= mass;
          else tree_zero_[node] = 0;
        }
      }
    }
  }
  for (int i = 0; i < 256; ++i) {
    probs_[i] = ppmd_model_->sqp[i];
    if (probs_[i] < 1) probs_[i] = 1;
  }
  ByteModel::ByteUpdate();
  probs_ /= probs_.sum();
  tree_context_ = 1;
}

} // namespace PPMD
