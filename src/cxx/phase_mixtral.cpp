// =============================================================
// PHASE 6.5 — Strategy A + LRU + DONTNEED + PARALLEL I/O + DENSE-WEIGHTS POLICY
//
//   MIXTRAL SUPPORT (this build):
//   - metadata keys read as llama.* with olmoe.* fallback:
//       Mixtral-8x7B : llama.block_count=32, llama.expert_count=8, llama.expert_used_count=2
//       OLMoE-1B-7B  : olmoe.block_count=16, olmoe.expert_count=64, olmoe.expert_used_count=8
//   - streaming path unchanged: fused expert layout (ne[2]=n_expert) is identical
//     to OLMoE; offset = data_offset + eid*bytes_per_expert still holds.
//   - BUILD REQUIREMENT: compile llama.cpp with -DGGML_CPU_REPACK=OFF. The fork's
//     repack tries to allocate a ~21 GB CPU_REPACK buffer for Mixtral Q4_K_M and
//     OOMs on 16 GB; with it off, Q4_K/Q6_K weights are used in-place via mmap
//     (stock llama.cpp behavior — verified byte-identical by the boot memcmp).
//
//   args: model prompt n_pred threads seed cache_mb [lanes] [dontneed] [stream 0/1]
//         [verify 0/1] [dense 0=mmap/1=warm/2=anon] [mlock 0/1] [sensor_interval tok]
//
//   Example (Mixtral smoke test, 64 tok, verify ON, cache 2 GiB, dense mmap):
//     ./build/stream models/mixtral-8x7b-instruct-v0.1-q4_k_m.gguf \
//       "<prompt>" 64 8 42 2048 4 0 1 1 0 0 32
//
//   Extends Phase 5 (all Phase 5 behavior unchanged, same first 10 args):
//
//   A. Durable expert page-cache purity (doc/phase6_5_result.md, design A):
//      A1. verify is a GATE knob only: benchmark runs pass verify=0 (design A1).
//          Correctness gate keeps verify=1 and must stay 0 FAILs.
//      A2. restore_all() REMOVED from the single-token decode path: expert tensors
//          stay rebound to staging across decodes. Restore only around prefill
//          (nothing rebound yet), batched decodes (ne>1) and teardown.
//      A3. Batched decode (ne[1]!=1): restore FIRST and fall back to resident mmap
//          (sizing n_ctx so this never fires is the harness's job — the guard here
//           is the safety net that keeps the batch path correct).
//      A4. Purity is audited by scripts/check_cache_purity.py (mincore) — this file
//          prints the dontneed flags for the report.
//
//   B. Dense (non-ex) weight policy (design B):
//      B1. Discovery on the ASK pass (ask=true runs for EVERY graph node, so this
//          gives guaranteed coverage): weight leaves (op==GGML_OP_NONE, named)
//          with a GGUF tensor match, excluding *_exps. KV-cache tensors and graph
//          inputs fall out via the GGUF intersection.
//      B2. Byte ranges: data_offset + tensor_offset per dense tensor (comp. of the
//          expert spans in the file).
//      B3. Policies (argv[11]): 0=mmap (nothing), 1=warm (WILLNEED once, dense is
//          never DONTNEED'd), 2=anon (default for Phase 8 — gated by dontneed).
//      B4. anon pipeline (boot-time, once, after prefill):
//          1. page-aligned anon buffers (one per dense tensor);
//          2. parallel pread through the existing IoPool lanes;
//          3. memcmp vs mmap ONCE (boot verify — abort on FAIL);
//          4. rebind each dense tensor ->data to its anon buffer; original saved;
//          5. fadvise(DONTNEED) on the dense file ranges (clean by construction:
//             after rebind no compute path touches the dense mmap — unlike the
//             experts there is no double-I/O risk);
//          6. teardown: restore originals BEFORE llama_free.
//      B5. Optional mlock() (argv[12]): best-effort, rlimit-aware, warns on failure.
//      B6. Residency sensor every argv[13] tokens: dense_resident_frac via mincore
//          over anon buffers (anon) or a MAP_SHARED view (mmap/warm page cache),
//          plus major-fault deltas via getrusage. Unmeasured stays -1, never 0.
//
//   DONTNEED defaults OFF (argv[8]=0) so RAM-resident runs keep the mmap pages in the
//   kernel page cache; purity runs pass 1 explicitly.
//
//   FIX 6.5.1 (post-gate crash rc=139, "warm-up: 0/32 experts"):
//     Newer llama.cpp (resolve_fused_ops / refactored MoE graph) no longer exposes the
//     *_exps tensors as srcs of the ffn_moe_topk node, so the observe-pass discovery
//     (which only scans the observed topk's srcs) found 0 experts and the streaming
//     path dereferenced L.gate.t==nullptr -> SIGSEGV. Fix:
//       (a) expert discovery mirrored onto the ASK pass (same guarantee as dense B1):
//           scans t and t->src[i] for EVERY graph node — version-proof;
//       (b) streaming guard: layer with undiscovered experts -> mmap fallback;
//       (c) main abort: stream=1 with 0 experts discovered refuses to start.
//
//   CORRECTIONS APPLIED (compile fixes):
//     1. `struct stat st;` collided with `Streamer st;` in main() — renamed to
//        `filestat`; size read fixed to `filestat.st_size`.
//     2. `IoPool::worker(int)` had an unnamed parameter referenced by the body —
//        parameter named `id`.
//     3. `#include <cerrno>` restored (errno used in strerror(errno) calls).
//     4. All `#ifdef` blocks use `__linux__` (not the markdown-mangled `**linux**`).
//     5. Dense discovery moved to the ASK pass (B1) — guaranteed coverage of every
//        graph node in any llama.cpp version, instead of the observe-pass window.
//     6. init_dense() warns loudly if dense=anon captured 0 tensors (silent no-op).
//     7. Streamer::gg is `gguf_context*` (was `ggml_context*` -> type error with
//        gguf_init_from_file / gguf_find_tensor).
//     8. `int32 eid` -> `int32_t eid` (typo, undefined identifier).
//     9. `#define _GNU_SOURCE` before includes so mincore/mlock/posix_memalign are
//        declared under `-std=c++11` (glibc keeps them behind GNU feature macros).
//   MIXTRAL APPLIED (this file):
//     a. metadata keys: llama.* with olmoe.* fallback (Mixtral 32/8/2).
//     b. default model path: Mixtral Q4_K_M (still overridable via argv[1]).
//     c. sanity check: n_layer/n_expert/n_used==0 -> abort (no silent div-by-zero).
// =============================================================
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cerrno>
#include <unistd.h>
#include <fcntl.h>
#include <vector>
#include <unordered_map>
#include <map>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <string>
#include <algorithm>
#include <cmath>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>

// ---- LRU / Strategy-A state (verbatim from Phase 4/5) -------------------------
struct CacheKey { int layer,kind,eid;
  bool operator==(const CacheKey&o)const{return layer==o.layer&&kind==o.kind&&eid==o.eid;} };
struct CacheKeyHash { size_t operator()(const CacheKey&k)const{
  size_t h=0; h=h*31+(size_t)k.layer; h=h*31+(size_t)k.kind; h=h*31+(size_t)k.eid; return h;} };
struct CacheEntry { std::vector<uint8_t> data; uint64_t last_used; };
struct ExpertCache {
  size_t max_bytes,used_bytes=0; uint64_t clock=0;
  std::unordered_map<CacheKey,CacheEntry,CacheKeyHash> map; uint64_t hits=0,misses=0;
  explicit ExpertCache(size_t m):max_bytes(m){}
  bool lookup(const CacheKey&k,size_t s,uint8_t*d){
    auto it=map.find(k); if(it==map.end()){++misses;return false;}
    ++hits; it->second.last_used=++clock; memcpy(d,it->second.data.data(),s); return true;}
  void insert(const CacheKey&k,const uint8_t*d,size_t s){
    if(map.count(k))return;
    while(used_bytes+s>max_bytes&&!map.empty()){
      CacheKey v; uint64_t mn=UINT64_MAX;
      for(auto&kv:map) if(kv.second.last_used<mn){mn=kv.second.last_used;v=kv.first;}
      used_bytes-=map[v].data.size(); map.erase(v);
    }
    map[k]={std::vector<uint8_t>(d,d+s),++clock}; used_bytes+=s;}
};

// ---- expert tensor bookkeeping (Phase 3/4/5) -----------------------------------
struct ExpertTensor { ggml_tensor*t=nullptr; const uint8_t*original=nullptr; const char*name=nullptr;
  size_t n_bytes=0,bytes_per_expert=0; size_t data_offset=0; };
struct LayerCtx { ExpertTensor gate,up,down; };

// ---- phase 6.5: dense (non-expert) tensor bookkeeping --------------------------
enum DenseMode { DENSE_MMAP=0, DENSE_WARM=1, DENSE_ANON=2 };
struct DenseTensor {
  ggml_tensor* t; const uint8_t* orig; uint8_t* buf;
  size_t size; size_t file_off; bool mlocked;
};

struct Streamer;  // fwd (IoPool holds Streamer*)
// ---- I/O task: expert read (Phase 5) OR dense boot read (6.5) ------------------
struct IoTask {
  int fd;
  ExpertTensor* et; int32_t eid; int kind; uint8_t* dst;  // expert
  uint8_t* d_dst; size_t d_off; size_t d_bytes;            // dense
  bool is_dense;
};

static bool has_sub(const char*n,const char*s){return n&&strstr(n,s);}
static void ensure_buf(uint8_t**b,size_t*c,size_t need){ if(*c<need){delete[]*b;*b=new uint8_t[need];*c=need;} }
static int moe_tensor_kind(const char*n){
  if(has_sub(n,"ffn_gate_exps"))return 0;
  if(has_sub(n,"ffn_up_exps"))return 1;
  if(has_sub(n,"ffn_down_exps"))return 2;
  return -1;
}

#ifdef __linux__
// fadvise(DONTNEED) only drops clean UNMAPPED pages; prefill already mapped all
// tensors via llama's mmap (populated PTEs), so fadvise becomes a no-op.
// madvise(MADV_DONTNEED) over the mapped range actually drops the pages (shared clean).
static size_t drop_mapped(const uint8_t* p, size_t bytes, size_t page){
  if(!p||!bytes) return 0;
  const uintptr_t a0 = (uintptr_t)p & ~(uintptr_t)(page-1);
  const uintptr_t a1 = ((uintptr_t)p + bytes + page - 1) & ~(uintptr_t)(page-1);
  if(a1<=a0) return 0;
  return madvise((void*)a0, (size_t)(a1-a0), MADV_DONTNEED)==0 ? (size_t)(a1-a0) : 0;
}
#endif

// ---- IoPool (Phase 5, extended for dense boot reads) ---------------------------
struct IoPool {
  struct Lane { std::mutex m; std::condition_variable cv; std::vector<IoTask> queue; bool stop=false; };
  std::vector<Lane*> lanes;
  std::vector<std::thread> threads;
  std::mutex done_m; std::condition_variable done_cv;
  int pending=0;
  Streamer* s; int n_lanes; bool verify;

  IoPool(Streamer* s_, int n, bool v): s(s_), n_lanes(n), verify(v) {
    lanes.reserve(n);
    for(int i=0;i<n;i++) lanes.push_back(new Lane());
    for(int i=0;i<n;i++) threads.emplace_back(&IoPool::worker,this,i);
  }
  ~IoPool(){ shutdown(); for(auto*L:lanes) delete L; }

  void worker(int id){
    Lane& L=*lanes[id];
    std::unique_lock<std::mutex> lk(L.m);
    for(;;){
      L.cv.wait(lk,[&]{ return L.stop||!L.queue.empty(); });
      if(L.stop&&L.queue.empty()) break;
      IoTask t=L.queue.back(); L.queue.pop_back();
      lk.unlock();                 // release lane lock during I/O
      do_read(t);
      { std::lock_guard<std::mutex> d(done_m); --pending; if(pending==0) done_cv.notify_all(); }
      lk.lock();
    }
  }

  void do_read(const IoTask&t){
    if(t.is_dense){
      // dense boot read — verify happens ONCE in init_dense (boot memcmp), not per read
      const ssize_t n=pread(t.fd,t.d_dst,t.d_bytes,(off_t)t.d_off);
      if(n!=(ssize_t)t.d_bytes)
        fprintf(stderr,"ERROR dense pread off=%zu got=%zd (%s)\n",t.d_off,(ssize_t)n,strerror(errno));
      return;
    }
    const size_t abs  = t.et->data_offset + (size_t)t.eid * t.et->bytes_per_expert;
    const size_t bytes= t.et->bytes_per_expert;
    const ssize_t n=pread(t.fd,t.dst,bytes,(off_t)abs);
    if(n!=(ssize_t)bytes)
      fprintf(stderr,"ERROR pread eid=%d off=%zu got=%zd (%s)\n",t.eid,abs,(ssize_t)n,strerror(errno));
    if(verify){
      const uint8_t*ref=(const uint8_t*)t.et->original + (size_t)t.eid*bytes;
      if(memcmp(t.dst,ref,bytes)!=0){ fprintf(stderr,"VERIFY FAIL: %s eid=%d\n",t.et->name,t.eid);
                                      n_verify_fail_inc(); }
    }
#ifdef __linux__
        if(dontneed()){
      drop_mapped((const uint8_t*)t.et->original + (size_t)t.eid*bytes, bytes, (size_t)getpagesize());
      posix_fadvise(t.fd,(off_t)abs,(off_t)bytes,POSIX_FADV_DONTNEED);   // now unmapped
    }
#endif
  }

  void submit(std::vector<IoTask>&tasks){
    size_t n=tasks.size(); if(n==0) return;
    { std::lock_guard<std::mutex> d(done_m); pending=(int)n; }
    for(int i=0;i<(int)n;i++){
      Lane& ln=*lanes[i%n_lanes];
      std::lock_guard<std::mutex> lk(ln.m);
      ln.queue.push_back(tasks[i]);
    }
    for(auto*L:lanes) L->cv.notify_all();
    { std::unique_lock<std::mutex> d(done_m);
      done_cv.wait(d,[&]{ return pending==0; }); }
  }
  void shutdown(){
    for(auto*L:lanes){ std::lock_guard<std::mutex> lk(L->m); L->stop=true; }
    for(auto*L:lanes) L->cv.notify_all();
    for(auto&th:threads) if(th.joinable()) th.join();
    threads.clear();
  }

  void n_verify_fail_inc();
  bool dontneed();
};

struct Streamer {
  std::vector<LayerCtx> layers; struct gguf_context*gg=nullptr; int fd=-1;
  size_t n_expert=64,n_expert_used=8; bool warmed_up=false,enable_stream=false;
  uint8_t*gate_full=nullptr; size_t gate_cap=0;   // full-size buffers (strategy A)
  uint8_t*up_full=nullptr;   size_t up_cap=0;
  uint8_t*down_full=nullptr; size_t down_cap=0;
  ExpertCache*cache=nullptr; IoPool*pool=nullptr;
  uint64_t n_expert_reads=0, n_verify_fail=0; bool dontneed=false;
  // ---- phase 6.5 dense state
  std::map<std::string,DenseTensor> dense_map;   // dedup by name (warm-up discovery)
  std::vector<DenseTensor> dense;                // sorted by file_off (boot)
  int dense_mode=DENSE_MMAP; bool mlock_on=false;
  long page=4096; size_t file_size=0;
  int sensor_interval=0; double dense_res_frac=-1.0;
  unsigned long long majflt_prev=0, majflt_last=0;
};
void IoPool::n_verify_fail_inc(){ ++s->n_verify_fail; }
bool IoPool::dontneed(){ return s->dontneed; }

// ---- 6.5 B6: residency sensor --------------------------------------------------
static unsigned long long major_faults_now(){
  struct rusage ru;
  if(getrusage(RUSAGE_SELF,&ru)!=0) return 0;
  return (unsigned long long)ru.ru_majflt;
}

// dense_resident_frac:
//   anon   -> mincore over the anon buffers (physical residency of OUR copies)
//   mmap/w -> mincore over a MAP_SHARED view of the file (page-cache residency)
// Returns -1.0 when unmeasurable (never silently zero).
static double dense_resident_frac(Streamer& s){
  if(s.dense.empty()) return -1.0;
  void* view=MAP_FAILED;
  if(s.dense_mode!=DENSE_ANON){
    view=mmap(nullptr,s.file_size,PROT_READ,MAP_SHARED,s.fd,0);
    if(view==MAP_FAILED){ fprintf(stderr,"sensor: mmap view failed (%s)\n",strerror(errno)); return -1.0; }
  }
  size_t total=0,res=0;
  for(const auto&d:s.dense){
    total+=d.size;
    const uint8_t* base = (s.dense_mode==DENSE_ANON) ? d.buf : (const uint8_t*)view + d.file_off;
    if(!base) continue;
    const uintptr_t off0 = (uintptr_t)base & (uintptr_t)(s.page-1);       // offset within 1st page
    const uint8_t* aligned = (const uint8_t*)((uintptr_t)base & ~(uintptr_t)(s.page-1));
    size_t need = off0 + d.size;
    size_t len  = ((need + s.page - 1) / (size_t)s.page) * (size_t)s.page; // page-rounded
    if(s.dense_mode!=DENSE_ANON){
      size_t rem = s.file_size - ((size_t)(aligned - (const uint8_t*)view));
      len = std::min(len, rem);                                            // don't probe past EOF
    }
    size_t np = len / (size_t)s.page;
    if(np==0) continue;
    std::vector<unsigned char> vec(np);
    if(mincore((void*)aligned,len,vec.data())!=0) continue;                // can EAGAIN; skip sample
        for(size_t p=0;p<np;p++){
      if(!(vec[p]&1)) continue;
      size_t lo=p*(size_t)s.page, hi=(p+1)*(size_t)s.page;
      size_t c0=std::max(lo,off0), c1=std::min(hi,off0+d.size);            // clip to tensor bytes
      if(c1>c0) res+=c1-c0;
    }
  }
  if(view!=MAP_FAILED) munmap(view,s.file_size);
  return total? (double)res/(double)total : -1.0;
}

// ---- 6.5 B4: dense policy boot (runs once, after prefill / before streaming) ---
static bool init_dense(Streamer& s, bool mlock_on){
  s.dense.reserve(s.dense_map.size());
  for(auto&kv:s.dense_map) s.dense.push_back(kv.second);
  std::sort(s.dense.begin(),s.dense.end(),
            [](const DenseTensor&a,const DenseTensor&b){return a.file_off<b.file_off;});
  size_t total=0; for(const auto&d:s.dense) total+=d.size;
  const char* dm = s.dense_mode==DENSE_WARM?"warm" : s.dense_mode==DENSE_ANON?"anon":"mmap";
  fprintf(stderr,"dense: mode=%s tensors=%zu bytes=%zu MiB\n",dm,s.dense.size(),total>>20);
  if(s.dense_mode==DENSE_ANON && s.dense.empty())
    fprintf(stderr,"WARN: dense discovery captured 0 tensors — anon is a no-op; gate 3 will fail\n");

  if(s.dense_mode==DENSE_WARM){
#ifdef __linux__
    for(const auto&d:s.dense) posix_fadvise(s.fd,(off_t)d.file_off,(off_t)d.size,POSIX_FADV_WILLNEED);
#endif
    fprintf(stderr,"dense: WILLNEED issued over %zu MiB (never DONTNEED'd)\n",total>>20);
    return true;
  }
  if(s.dense_mode==DENSE_MMAP) return true;   // nothing to do

  // ---- DENSE_ANON ---------------------------------------------------------
  // 1+2: allocate page-aligned anon buffers, read all in parallel via the lanes
  std::vector<IoTask> tasks; tasks.reserve(s.dense.size());
  for(auto&d:s.dense){
    size_t alloc=((d.size+s.page-1)/(size_t)s.page)*(size_t)s.page;
    if(posix_memalign((void**)&d.buf,(size_t)s.page,alloc)!=0){
      fprintf(stderr,"dense: alloc %zu failed\n",d.size); return false;
    }
    d.buf[0]=0; d.mlocked=false;
    tasks.push_back({s.fd,nullptr,0,0,nullptr,d.buf,d.file_off,d.size,true});
  }
  if(s.pool) s.pool->submit(tasks);
  else{
    for(auto&t:tasks){
      ssize_t n=pread(t.fd,t.d_dst,t.d_bytes,(off_t)t.d_off);
      if(n!=(ssize_t)t.d_bytes)
        fprintf(stderr,"ERROR dense pread off=%zu got=%zd (%s)\n",t.d_off,(ssize_t)n,strerror(errno));
    }
  }

  // 3. boot verify ONCE: anon copies == mmap originals (correctness gate)
  size_t boot_fail=0;
  for(const auto&d:s.dense)
    if(d.orig && memcmp(d.buf,d.orig,d.size)!=0){ ++boot_fail; fprintf(stderr,"DENSE VERIFY FAIL: %s\n",d.t->name); }
  fprintf(stderr,"dense anon boot verify: %zu FAILs (0 = anon copies == mmap originals)\n",boot_fail);
  if(boot_fail>0) return false;   // correctness before speed

  // 4. rebind each dense tensor ->data onto its anon buffer (original saved in orig)
  for(auto&d:s.dense) d.t->data=d.buf;

  // 5. optional mlock (best-effort, rlimit-aware)
  if(mlock_on){
    struct rlimit rl; getrlimit(RLIMIT_MEMLOCK,&rl);
    size_t locked=0;
    for(auto&d:s.dense){
      if(mlock(d.buf,d.size)!=0){
        fprintf(stderr,"mlock FAIL (%s) — raise RLIMIT_MEMLOCK (soft=%llu hard=%llu); continuing unlocked\n",
                strerror(errno),(unsigned long long)rl.rlim_cur,(unsigned long long)rl.rlim_max);
        break; // best-effort: rest of the buffers stay unlocked
      }
      d.mlocked=true; locked+=d.size;
    }
    fprintf(stderr,"dense: mlock %zu MiB across %zu buffers\n",locked>>20,s.dense.size());
  }

  // 6. DONTNEED the dense file ranges — clean by construction: after the rebind no
  //    compute path touches the dense mmap (unlike the experts: no double-I/O risk).
  //    Gated by the global dontneed knob so the bench matrix can A/B it.
    if(s.dontneed){
#ifdef __linux__
    size_t f1=0, md=0, f2=0;
    for(const auto&d:s.dense)
      if(posix_fadvise(s.fd,(off_t)d.file_off,(off_t)d.size,POSIX_FADV_DONTNEED)==0) f1+=d.size;
    for(const auto&d:s.dense) md+=drop_mapped((const uint8_t*)d.orig,d.size,(size_t)s.page);
    // pass 2: madvise zapped the PTEs -> pages are now UNMAPPED -> fadvise invalidates
    for(const auto&d:s.dense)
      if(posix_fadvise(s.fd,(off_t)d.file_off,(off_t)d.size,POSIX_FADV_DONTNEED)==0) f2+=d.size;
    fprintf(stderr,"dense: DONTNEED fadvise=%zuMiB madvise=%zuMiB fadvise_after=%zuMiB\n",
            f1>>20, md>>20, f2>>20);
#endif
  }
  fprintf(stderr,"dense: anon ready — %zu tensors rebound, %zu MiB total\n",s.dense.size(),total>>20);
  return true;
}

// ---- 6.5 teardown: restore dense originals + release buffers -------------------
static void restore_dense(Streamer& s){
  for(auto&d:s.dense){
    if(d.t && d.t->data==d.buf) d.t->data=(void*)d.orig;
    if(d.mlocked && d.buf) munlock(d.buf,d.size);
    free(d.buf); d.buf=nullptr; d.mlocked=false;
  }
}

static void restore_all(Streamer&s);   // fwd decl (used by the callback's ne>1 guard)

// ---- callback ----------------------------------------------------------------
static bool moe_callback(ggml_tensor*t,bool ask,void*ud){
  Streamer* s=(Streamer*)ud;
  if(ask){
    // ---- 6.5 B1: dense discovery on the ASK pass ---------------------------
    // ask=true is called for EVERY node of the graph, so scanning srcs here gives
    // guaranteed coverage of all dense weight leaves in any llama.cpp version,
    // independent of which nodes the scheduler isolates / observes.
    if(!s->warmed_up){
      auto try_dense=[&](ggml_tensor* c){
        if(!c||!c->name||c->name[0]=='\0'||!c->data) return;
        if(c->op!=GGML_OP_NONE) return;                       // weight leaf only
        if(has_sub(c->name,"exps")) return;                   // experts are streamed, not dense
        int ti=gguf_find_tensor(s->gg,c->name);
        if(ti<0) return;                                      // KV-cache / graph inputs drop here
        if(s->dense_map.count(c->name)) return;
        DenseTensor d;
        d.t=c; d.orig=(const uint8_t*)c->data; d.buf=nullptr; d.mlocked=false;
        d.size=ggml_nbytes(c); if(d.size==0) return;
        d.file_off=gguf_get_data_offset(s->gg)+(size_t)gguf_get_tensor_offset(s->gg,ti);
        s->dense_map[c->name]=d;
      };
      try_dense(t);                                           // the node itself can BE a leaf
      for(int i=0;i<GGML_MAX_SRC;i++) try_dense(t->src[i]);

      // ---- FIX 6.5.1: expert discovery on the ASK pass (same guarantee as dense)
      // The observe pass below only sees the srcs of the *observed* topk nodes; on
      // newer llama.cpp the refactored MoE graph no longer lists the *_exps tensors
      // there. The ASK pass visits EVERY node and every src, so it always finds them.
      auto try_expert=[&](ggml_tensor* c){
        if(!c||!c->name||c->name[0]=='\0'||!c->data) return;
        int kind=moe_tensor_kind(c->name); if(kind<0) return;
        int layer=-1; if(sscanf(c->name,"blk.%d.",&layer)!=1) return;
        if(layer<0||layer>=(int)s->layers.size()) return;
        ExpertTensor*target= kind==0?&s->layers[layer].gate:
                             kind==1?&s->layers[layer].up:&s->layers[layer].down;
        if(target->t) return;
        target->t=c; target->original=(const uint8_t*)c->data; target->name=c->name;
        target->n_bytes=ggml_nbytes(c); target->bytes_per_expert=target->n_bytes/s->n_expert;
        int tid=gguf_find_tensor(s->gg,c->name);
        if(tid>=0) target->data_offset=gguf_get_data_offset(s->gg)+(size_t)gguf_get_tensor_offset(s->gg,tid);
      };
      try_expert(t);
      for(int i=0;i<GGML_MAX_SRC;i++) try_expert(t->src[i]);
    }
    return s->warmed_up||has_sub(t->name,"ffn_moe_topk");
  }

  // ---- warm-up: expert discovery (Phase 3/4/5) — observe pass ----------------
  // Kept as a secondary fallback; the ASK-pass discovery above normally fills
  // target->t first, so this loop is a no-op on healthy builds.
  if(!s->warmed_up){
    for(int i=0;i<GGML_MAX_SRC;i++){
      ggml_tensor* src=t->src[i]; if(!src||!src->name) break;
      int kind=moe_tensor_kind(src->name); if(kind<0) continue;
      int layer=-1; if(sscanf(src->name,"blk.%d.",&layer)!=1) continue;
      if(layer<0||layer>=(int)s->layers.size()) continue;
      ExpertTensor*target= kind==0?&s->layers[layer].gate: kind==1?&s->layers[layer].up:&s->layers[layer].down;
      if(target->t) continue;
      target->t=src; target->original=(const uint8_t*)src->data; target->name=src->name;
      target->n_bytes=ggml_nbytes(src); target->bytes_per_expert=target->n_bytes/s->n_expert;
      int tid=gguf_find_tensor(s->gg,src->name);
      if(tid>=0) target->data_offset=gguf_get_data_offset(s->gg)+(size_t)gguf_get_tensor_offset(s->gg,tid);
    }
    return true;
  }

  // ---- active streaming (Phase 5 logic; 6.5 A3 guard added) ------------------
  if(s->enable_stream && has_sub(t->name,"ffn_moe_topk")){
    int layer=-1; if(sscanf(t->name,"ffn_moe_topk-%d",&layer)!=1) return true;
    if(layer<0||layer>=(int)s->layers.size()) return true;
    if(t->ne[1]!=1){
      // 6.5 A3: a batched decode bypasses streaming — restore FIRST so llama reads the
      // resident mmap experts, NOT stale staging from the previous single-token decode.
      restore_all(*s);
      return true;
    }
    LayerCtx& L=s->layers[layer];

    // FIX 6.5.1: never deref null expert tensors (0/32 discovery) — mmap fallback
    if(!L.gate.t||!L.up.t||!L.down.t){
      fprintf(stderr,"stream: experts layer %d undiscovered — mmap fallback\n",layer);
      return true;
    }

    const int32_t*orig_ids=(const int32_t*)t->data;

    ensure_buf(&s->gate_full,&s->gate_cap,s->n_expert*L.gate.bytes_per_expert);
    ensure_buf(&s->up_full,&s->up_cap,s->n_expert*L.up.bytes_per_expert);
    ensure_buf(&s->down_full,&s->down_cap,s->n_expert*L.down.bytes_per_expert);

    std::vector<IoTask> misses; misses.reserve(s->n_expert_used*3);
    auto collect=[&](int kind,int32_t eid,ExpertTensor&et,uint8_t*base){
      size_t sl=(size_t)eid*et.bytes_per_expert; uint8_t*dst=base+sl;
      CacheKey ck{layer,kind,eid};
      if(s->cache&&s->cache->lookup(ck,et.bytes_per_expert,dst)) return; // hit
      misses.push_back({s->fd,&et,eid,kind,dst,nullptr,0,0,false});      // miss
    };
    for(size_t k=0;k<s->n_expert_used;++k){
      int32_t eid=orig_ids[k]; if(eid<0||eid>=(int32_t)s->n_expert) continue;
      collect(0,eid,L.gate,s->gate_full);
      collect(1,eid,L.up,s->up_full);
      collect(2,eid,L.down,s->down_full);
    }

    if(!misses.empty()){
      if(s->pool) s->pool->submit(misses);
      else for(auto&m:misses){
        const size_t abs = m.et->data_offset + (size_t)m.eid*m.et->bytes_per_expert;
        const size_t bytes= m.et->bytes_per_expert;
        ssize_t n=pread(m.fd,m.dst,bytes,(off_t)abs);
        #ifdef __linux__
        if(s->dontneed){
          drop_mapped((const uint8_t*)m.et->original + (size_t)m.eid*bytes, bytes, (size_t)getpagesize());
          posix_fadvise(m.fd,(off_t)abs,(off_t)bytes,POSIX_FADV_DONTNEED);
        }
#endif
      }
      for(auto&m:misses){
        if(s->cache){ CacheKey ck{layer,m.kind,m.eid};
          s->cache->insert(ck,m.dst,m.et->bytes_per_expert); }
        ++s->n_expert_reads;
      }
    }

    // STRATEGY A: keep original ids; ggml reads at offset eid*bytes
    L.gate.t->data=s->gate_full; L.up.t->data=s->up_full; L.down.t->data=s->down_full;
  }
  return true;
}

static void restore_all(Streamer&s){
  for(auto&L:s.layers){ if(L.gate.t)L.gate.t->data=(void*)L.gate.original;
    if(L.up.t)L.up.t->data=(void*)L.up.original; if(L.down.t)L.down.t->data=(void*)L.down.original; }
}

// ---- main ---------------------------------------------------------------------
int main(int argc,char**argv){
  std::string model_path=argc>1?argv[1]:"models/mixtral-8x7b-instruct-v0.1-q4_k_m.gguf";
  std::string prompt=argc>2?argv[2]:"Explain how mixture of experts routing works.";
  int n_pred=argc>3?atoi(argv[3]):128; int n_threads=argc>4?atoi(argv[4]):4;
  uint32_t seed=argc>5?(uint32_t)strtoul(argv[5],nullptr,10):42;
  size_t cache_mb=argc>6?(size_t)atoll(argv[6]):64;
  int lanes=argc>7?atoi(argv[7]):4;
  bool dontneed=argc>8?atoi(argv[8])!=0:false;      // 6.5: default OFF (user decision)
  bool g_stream=argc>9?atoi(argv[9])!=0:true;
  bool g_verify=argc>10?atoi(argv[10])!=0:0;
  int dense_mode=argc>11?atoi(argv[11]):0;          // 0=mmap 1=warm 2=anon
  bool mlock_on=argc>12?atoi(argv[12])!=0:false;
  int sensor_interval=argc>13?atoi(argv[13]):32;    // 0=off

  llama_backend_init();
  struct gguf_init_params ggpar={true,nullptr};
  struct gguf_context*gg=gguf_init_from_file(model_path.c_str(),ggpar);
  if(!gg){fprintf(stderr,"error gguf\n");return 1;}
  auto ggu=[&](const char*k,size_t d){int i=gguf_find_key(gg,k);return i<0?d:(size_t)gguf_get_val_u32(gg,i);};
  // MIXTRAL: llama.* keys (Mixtral 32/8/2) with olmoe.* fallback (OLMoE 16/64/8)
  size_t n_layer=ggu("llama.block_count",ggu("olmoe.block_count",16));
  size_t n_expert=ggu("llama.expert_count",ggu("olmoe.expert_count",64));
  size_t n_used=ggu("llama.expert_used_count",ggu("olmoe.expert_used_count",8));
  if(n_layer==0||n_expert==0||n_used==0){
    fprintf(stderr,"FATAL: model metadata not found (keys llama.* / olmoe.*)\n"); return 1;
  }
  fprintf(stderr,"%zu layers, %zu experts, top-%zu, cache=%zuMiB, lanes=%d, dontneed=%d, stream=%d, verify=%d, dense=%d, mlock=%d, sensor=%d",
          n_layer,n_expert,n_used,cache_mb,lanes,dontneed,g_stream,g_verify,dense_mode,mlock_on,sensor_interval);

  llama_model_params mp=llama_model_default_params();
  mp.use_extra_bufts=false; mp.load_mode=LLAMA_LOAD_MODE_MMAP;
  llama_model*model=llama_model_load_from_file(model_path.c_str(),mp);
  if(!model){fprintf(stderr,"err model\n");return 1;}
  int fd=open(model_path.c_str(),O_RDONLY); if(fd<0){fprintf(stderr,"err open\n");return 1;}

  struct stat filestat;                       // renamed (was `st`, colliding with Streamer)
  if(fstat(fd,&filestat)!=0){fprintf(stderr,"err fstat\n");return 1;}

  Streamer st; st.gg=gg; st.fd=fd; st.n_expert=n_expert; st.n_expert_used=n_used;
  st.layers.resize(n_layer); st.dontneed=dontneed;
  st.dense_mode=dense_mode; st.mlock_on=mlock_on;
  st.page=(long)sysconf(_SC_PAGESIZE); if(st.page<1) st.page=4096;
  st.file_size=(size_t)filestat.st_size;
  st.sensor_interval=sensor_interval;
  ExpertCache cache(cache_mb*(1u<<20)); st.cache=(cache_mb>0?&cache:nullptr);

  llama_context_params cp=llama_context_default_params();
  cp.n_threads=n_threads; cp.n_threads_batch=n_threads;
  cp.cb_eval=moe_callback; cp.cb_eval_user_data=&st;
  llama_context*ctx=llama_init_from_model(model,cp);
  if(!ctx){fprintf(stderr,"err ctx\n");return 1;}
  const llama_vocab*vocab=llama_model_get_vocab(model);

  char tmpl[4096]; int tl=llama_model_meta_val_str(model,"tokenizer.chat_template",tmpl,sizeof tmpl);
  if(tl<0){fprintf(stderr,"no template; raw\n");}
  else{
    llama_chat_message msgs[]={{"user",prompt.c_str()}};
    std::vector<char> fmt(prompt.size()*2+512);
    int fl=llama_chat_apply_template(tmpl,msgs,1,true,fmt.data(),(int)fmt.size());
    if(fl<0){fprintf(stderr,"tmpl err; raw\n");} else {fmt.resize(fl);prompt.assign(fmt.data(),fmt.size());}
  }
  std::vector<llama_token> toks(prompt.size()+64);
  int nt=llama_tokenize(vocab,prompt.c_str(),(int)prompt.size(),toks.data(),(int)toks.size(),true,false);
  if(nt<0)nt=0; toks.resize(nt);
  { llama_batch b=llama_batch_get_one(toks.data(),(int32_t)toks.size());
    if(llama_decode(ctx,b)!=0){fprintf(stderr,"prefill err\n");return 1;} }
  st.warmed_up=true; st.enable_stream=g_stream;
  if(lanes>0) st.pool=new IoPool(&st,lanes,g_verify);   // lanes serve BOTH dense boot + expert misses
  size_t found=0; for(auto&L:st.layers){if(L.gate.t)found++;if(L.down.t)found++;}
  fprintf(stderr,"warm-up: %zu/%zu experts\n",found,n_layer*2);

  // FIX 6.5.1: refuse to stream with zero discovered experts (would SIGSEGV)
  if(g_stream && found==0){
    fprintf(stderr,"FATAL: 0 experts discovered with stream=1 — ASK-pass discovery found nothing; refusing to start\n");
    return 1;
  }

  // 6.5: dense conversion AFTER prefill (prefill runs mmap; conversion is boot-time once)
  if(!init_dense(st,mlock_on)){fprintf(stderr,"dense init failed\n");return 1;}

    if(dontneed){
#ifdef __linux__
    size_t eb=0;
    for(auto&L:st.layers){
      for(ExpertTensor* et : {&L.gate,&L.up,&L.down}){
        if(!et->t) continue;
        drop_mapped((const uint8_t*)et->original, et->n_bytes, (size_t)st.page);
        posix_fadvise(fd,(off_t)et->data_offset,(off_t)et->n_bytes,POSIX_FADV_DONTNEED);
        eb+=et->n_bytes;
      }
    }
    fprintf(stderr,"expert boot DONTNEED: %zu MiB (madvise->fadvise full ranges)\n",eb>>20);
#endif
  }

  llama_sampler*smpl=llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(smpl,llama_sampler_init_temp(0.0f));
  llama_sampler_chain_add(smpl,llama_sampler_init_greedy());

  int64_t t0=llama_time_us(); std::vector<llama_token> gen=toks;
  const unsigned long long f0=major_faults_now();
  st.majflt_prev=f0;
  if(sensor_interval>0) st.dense_res_frac=dense_resident_frac(st);

  for(int i=0;i<n_pred;i++){
    llama_token tok=gen.back(); llama_batch b=llama_batch_get_one(&tok,1);
    if(llama_decode(ctx,b)!=0){fprintf(stderr,"decode err\n");break;}
    // 6.5 A2: NO restore_all() here — expert tensors stay rebound to staging across
    // single-token decodes. Restore happens only via the ne>1 guard and teardown.
    if(sensor_interval>0 && (i+1)%sensor_interval==0){
      st.dense_res_frac=dense_resident_frac(st);
      unsigned long long now=major_faults_now();
      st.majflt_last=now-st.majflt_prev; st.majflt_prev=now;
      fprintf(stderr,"[6.5 sensor] t=%d dense_resident_frac=%.1f%% majflt_delta=%llu\n",
              i+1, st.dense_res_frac<0?-1.0:100.0*st.dense_res_frac, st.majflt_last);
    }
    llama_token nt2=llama_sampler_sample(smpl,ctx,-1); llama_sampler_accept(smpl,nt2);
    char p[256]; int np=llama_token_to_piece(vocab,nt2,p,sizeof p,0,false);
    if(np>0)fwrite(p,1,(size_t)np,stdout); fflush(stdout);
    gen.push_back(nt2); if(llama_vocab_is_eog(vocab,nt2))break;
  }
  int64_t t1=llama_time_us(); printf("\n");
  size_t ng=gen.size()-toks.size(); double dt=(double)(t1-t0)/1e6;
  fprintf(stderr,"Phase6_5: %zu tokens, %.2fs, %.2f tok/s\n",ng,dt,ng/dt);
  if(st.cache){double hr=100.0*(double)st.cache->hits/(double)(st.cache->hits+st.cache->misses+1e-9);
    fprintf(stderr,"Cache: hits=%llu misses=%llu hit_rate=%.1f%% disk_reads=%llu\n",
      (unsigned long long)st.cache->hits,(unsigned long long)st.cache->misses,hr,
      (unsigned long long)st.n_expert_reads);}
  else fprintf(stderr,"Cache: disabled. disk_reads=%llu\n",(unsigned long long)st.n_expert_reads);
  fprintf(stderr,"verify: %llu FAILs (0 = all expert preads == mmap)\n",(unsigned long long)st.n_verify_fail);

  const char* dm = st.dense_mode==DENSE_WARM?"warm" : st.dense_mode==DENSE_ANON?"anon":"mmap";
  size_t dbytes=0; for(const auto&d:st.dense) dbytes+=d.size;
  unsigned long long majflt_total=major_faults_now()-f0;
  st.dense_res_frac=dense_resident_frac(st);                 // final sample
  fprintf(stderr,"Dense: mode=%s tensors=%zu %zuMiB resident_frac=%.1f%% majflt_total=%llu (%.2f/tok)\n",
          dm, st.dense.size(), dbytes>>20,
          st.dense_res_frac<0?-1.0:100.0*st.dense_res_frac, majflt_total, ng?(double)majflt_total/(double)ng:0.0);
  fprintf(stderr,"purity: expert_dontneed=%d dense_dontneed=%d (audit: check_cache_purity.py / check_dense_residency.py)\n",
          st.dontneed?1:0, (st.dense_mode==DENSE_ANON&&st.dontneed)?1:0);

  // teardown (6.5): restore experts AND dense BEFORE freeing the model context
  restore_all(st);
  restore_dense(st);
  delete st.pool;
  llama_sampler_free(smpl); llama_free(ctx); llama_model_free(model);
  llama_backend_free(); gguf_free(gg); if(fd>=0)close(fd);
  delete[] st.gate_full; delete[] st.up_full; delete[] st.down_full;
  return 0;
}