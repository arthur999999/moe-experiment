// =============================================================
// PHASE 5 - Strategy A + LRU + DONTNEED + PARALLEL I/O (lanes)
//   args: model prompt n_pred threads seed cache_mb [lanes] [dontneed] [stream 0/1] [verify 0/1]
//
//   Same math as Phase 4 (Strategy A: full-size buffers, absolute expert
//   offsets data_offset+eid*bytes, ORIGINAL router ids kept). The ONLY change
//   is WHERE miss bytes come from: N concurrent pread lanes instead of serial.
//
//   Gate (self-contained): run stream=1 and stream=0 (resident mmap, equals
//   llama-cli) and `diff` — must be empty. No reliance on stored references.
//
//   Threading model:
//     - 1 thread per lane, each lane owns a mutex/CV + its tiles queue.
//     - Static slice partition of miss tasks: lane i takes [i*n/L,(i+1)*n/L).
//     - shared done_m/done_cv: submit() sets pending=n, workers decrement under
//       done_m, main thread waits pending==0.
//     - cache lookup/insert + scatter stay on the MAIN thread only; lanes never
//       touch the LRU map (no races).
// =============================================================
#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <fcntl.h>
#include <unistd.h>

// ---- LRU / Strategy-A state (verbatim from Phase 4) -------------------------
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
    CacheEntry e; e.data.assign(d,d+s); e.last_used=++clock; map[k]=std::move(e); used_bytes+=s;}
};

struct ExpertTensor { struct ggml_tensor*t=nullptr; void*original=nullptr; const char*name=nullptr;
  size_t data_offset=0,n_bytes=0,bytes_per_expert=0; };
struct LayerCtx { ExpertTensor gate,up,down; };

// ---- forward declaration for the pool (needs Streamer) ----------------------
struct Streamer;

// ---- a single miss read, handed to one lane --------------------------------
struct IoTask {
  int fd; const ExpertTensor* et; int32_t eid; int kind; uint8_t* dst;
};

// ---- parallel read lane pool ------------------------------------------------
struct IoPool {
  struct Lane { std::mutex m; std::condition_variable cv;
                std::vector<IoTask> queue; bool stop=false; };
  std::vector<std::unique_ptr<Lane>> lanes;   // *pointers* = movable
  std::vector<std::thread> threads;
  std::mutex done_m; std::condition_variable done_cv;
  int pending=0;
  Streamer* s; int n_lanes; bool verify;

  IoPool(Streamer* s_, int n, bool v): s(s_), n_lanes(n), verify(v) {
    lanes.reserve(n);
    for(int i=0;i<n;i++) lanes.push_back(std::unique_ptr<Lane>(new Lane()));  // C++11 ok // constrói no lugar
    for(int i=0;i<n;i++) threads.emplace_back(worker_thunk,this,i);
  }
  ~IoPool(){ shutdown(); }
  static void worker_thunk(IoPool*p,int id){ p->worker(id); }

  void worker(int id){
    Lane& L=*lanes[id];
    std::unique_lock<std::mutex> lk(L.m);
    for(;;){
      L.cv.wait(lk,[&]{ return L.stop||!L.queue.empty(); });
      if(L.stop&&L.queue.empty()) break;
      IoTask t=L.queue.back(); L.queue.pop_back();
      lk.unlock();                 // libera a lane durante o I/O
      do_read(t);
      { std::lock_guard<std::mutex> d(done_m); --pending; if(pending==0) done_cv.notify_all(); }
      lk.lock();
    }
  }

  void do_read(const IoTask&t){
    const size_t abs  = t.et->data_offset + (size_t)t.eid * t.et->bytes_per_expert;
    const size_t bytes= t.et->bytes_per_expert;
    const ssize_t n=pread(t.fd,t.dst,bytes,(off_t)abs);
    if(n!=(ssize_t)bytes)
      fprintf(stderr,"ERROR pread eid=%d off=%zu got=%zd (%s)\n",t.eid,abs,n,strerror(errno));
    if(verify){
      const uint8_t*ref=(const uint8_t*)t.et->original + (size_t)t.eid*bytes;
      if(memcmp(t.dst,ref,bytes)!=0){ fprintf(stderr,"VERIFY FAIL: %s eid=%d\n",t.et->name,t.eid);
                                      n_verify_fail_inc(); }
    }
#ifdef __linux__
    if(dontneed())
      posix_fadvise(t.fd,(off_t)abs,(off_t)bytes,POSIX_FADV_DONTNEED);
#endif
  }

  void submit(std::vector<IoTask>&tasks){
    size_t n=tasks.size(); if(n==0) return;
    { std::lock_guard<std::mutex> d(done_m); pending=(int)n; }
    for(int i=0;i<n_lanes;i++){
      size_t lo=(i*n)/n_lanes, hi=((i+1)*n)/n_lanes; if(lo==hi) continue;
      Lane&ln=*lanes[i];
      { std::lock_guard<std::mutex> lk(ln.m);
        for(size_t k=lo;k<hi;k++) ln.queue.push_back(tasks[k]);
        ln.cv.notify_one(); }
    }
    std::unique_lock<std::mutex> d(done_m);
    done_cv.wait(d,[&]{ return pending==0; });
  }
  void shutdown(){
    for(auto&lp:lanes){ std::lock_guard<std::mutex> lk(lp->m); lp->stop=true; }
    for(auto&lp:lanes) lp->cv.notify_all();
    for(auto&th:threads) if(th.joinable()) th.join();
  }

  void n_verify_fail_inc();
  bool dontneed();
};

struct Streamer {
  std::vector<LayerCtx> layers; struct gguf_context*gg=nullptr; int fd=-1;
  size_t n_expert=64,n_expert_used=8; bool warmed_up=false,enable_stream=false;
  uint8_t*gate_full=nullptr; size_t gate_cap=0;   // buffers CHEIOS (strategy A)
  uint8_t*up_full=nullptr;   size_t up_cap=0;
  uint8_t*down_full=nullptr; size_t down_cap=0;
  ExpertCache*cache=nullptr; IoPool*pool=nullptr;
  uint64_t n_expert_reads=0, n_verify_fail=0; bool dontneed=true;
};
// forward-defined helpers for the pool (defined after Streamer fully known)
void IoPool::n_verify_fail_inc(){ ++s->n_verify_fail; }
bool IoPool::dontneed(){ return s->dontneed; }

static bool has_sub(const char*n,const char*s){return n&&strstr(n,s);}
static void ensure_buf(uint8_t**b,size_t*c,size_t need){ if(*c<need){delete[]*b;*b=new uint8_t[need];*c=need;} }

static int moe_tensor_kind(const char*n){
  if(has_sub(n,"ffn_gate_exps"))return 0; if(has_sub(n,"ffn_up_exps"))return 1;
  if(has_sub(n,"ffn_down_exps"))return 2; return -1; }

static bool moe_callback(struct ggml_tensor*t,bool ask,void*ud){
  auto s=(Streamer*)ud;
  if(ask) return !s->warmed_up||has_sub(t->name,"ffn_moe_topk");

  // ---- warm-up: discover expert tensors from mul_mat_id src ----------------
  if(!s->warmed_up){
    for(int i=0;i<GGML_MAX_SRC;++i){
      struct ggml_tensor*src=t->src[i]; if(!src||!src->name) break;
      int kind=moe_tensor_kind(src->name); if(kind<0) continue;
      int layer=-1; if(sscanf(src->name,"blk.%d.",&layer)!=1) continue;
      if(layer<0||layer>=(int)s->layers.size()) continue;
      ExpertTensor*target= kind==0?&s->layers[layer].gate: kind==1?&s->layers[layer].up:&s->layers[layer].down;
      if(target->t) continue;
      target->t=src; target->original=src->data; target->name=src->name;
      target->n_bytes=ggml_nbytes(src); target->bytes_per_expert=target->n_bytes/s->n_expert;
      int tid=gguf_find_tensor(s->gg,src->name);
      if(tid>=0) target->data_offset=gguf_get_data_offset(s->gg)+gguf_get_tensor_offset(s->gg,tid);
    }
    return true;
  }

  // ---- active streaming -----------------------------------------------------
  if(s->enable_stream && has_sub(t->name,"ffn_moe_topk")){
    int layer=-1; if(sscanf(t->name,"ffn_moe_topk-%d",&layer)!=1) return true;
    if(layer<0||layer>=(int)s->layers.size()) return true;
    if(t->ne[1]!=1) return true;
    LayerCtx&L=s->layers[layer];
    const int32_t*orig_ids=(const int32_t*)t->data;

    ensure_buf(&s->gate_full,&s->gate_cap,s->n_expert*L.gate.bytes_per_expert);
    ensure_buf(&s->up_full,&s->up_cap,s->n_expert*L.up.bytes_per_expert);
    ensure_buf(&s->down_full,&s->down_cap,s->n_expert*L.down.bytes_per_expert);

    // collect cache MISSES only (main thread does lookups; lanes never touch cache)
    std::vector<IoTask> misses; misses.reserve(s->n_expert_used*3);
    auto collect=[&](int kind,int32_t eid,ExpertTensor&et,uint8_t*base){
      size_t sl=(size_t)eid*et.bytes_per_expert; uint8_t*dst=base+sl;
      CacheKey ck{layer,kind,eid};
      if(s->cache&&s->cache->lookup(ck,et.bytes_per_expert,dst)) return; // hit
      misses.push_back({s->fd,&et,eid,kind,dst});                       // miss
    };
    for(size_t k=0;k<s->n_expert_used;++k){
      int32_t eid=orig_ids[k]; if(eid<0||eid>=(int32_t)s->n_expert) continue;
      collect(0,eid,L.gate,s->gate_full);
      collect(1,eid,L.up,s->up_full);
      collect(2,eid,L.down,s->down_full);
    }

    // fetch the misses (parallel via lanes, or serial if pool disabled)
    if(!misses.empty()){
      if(s->pool) s->pool->submit(misses);
      else for(auto&m:misses){ /* serial fallback: reuse a one-shot lane-less read */
        const size_t abs = m.et->data_offset + (size_t)m.eid*m.et->bytes_per_expert;
        const size_t bytes= m.et->bytes_per_expert;
        ssize_t n=pread(m.fd,m.dst,bytes,(off_t)abs);
        if(n!=(ssize_t)bytes) fprintf(stderr,"ERROR pread eid=%d off=%zu got=%zd (%s)\n",m.eid,abs,n,strerror(errno));
#ifdef __linux__
        if(s->dontneed) posix_fadvise(m.fd,(off_t)abs,(off_t)bytes,POSIX_FADV_DONTNEED);
#endif
      }
      // cache-insert + count AFTER reads complete (data now in dst buffers)
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
  for(auto&L:s.layers){ if(L.gate.t)L.gate.t->data=L.gate.original;
    if(L.up.t)L.up.t->data=L.up.original; if(L.down.t)L.down.t->data=L.down.original; }
}

int main(int argc,char**argv){
  std::string model_path=argc>1?argv[1]:"models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf";
  std::string prompt=argc>2?argv[2]:"Explain how mixture of experts routing works.";
  int n_pred=argc>3?atoi(argv[3]):128; int n_threads=argc>4?atoi(argv[4]):4;
  uint32_t seed=argc>5?(uint32_t)strtoul(argv[5],nullptr,10):42;
  size_t cache_mb=argc>6?(size_t)atoll(argv[6]):64;
  int lanes=argc>7?atoi(argv[7]):4;
  bool dontneed=argc>8?atoi(argv[8])!=0:true;
  bool g_stream=argc>9?atoi(argv[9])!=0:true;
  bool g_verify=argc>10?atoi(argv[10])!=0:0;

  llama_backend_init();
  struct gguf_init_params ggpar={true,nullptr};
  struct gguf_context*gg=gguf_init_from_file(model_path.c_str(),ggpar);
  if(!gg){fprintf(stderr,"error gguf\n");return 1;}
  auto ggu=[&](const char*k,size_t d){int i=gguf_find_key(gg,k);return i<0?d:(size_t)gguf_get_val_u32(gg,i);};
  size_t n_layer=ggu("olmoe.block_count",16), n_expert=ggu("olmoe.expert_count",64), n_used=ggu("olmoe.expert_used_count",8);
  fprintf(stderr,"%zu layers, %zu experts, top-%zu, cache=%zuMiB, lanes=%d, dontneed=%d, stream=%d, verify=%d\n",
          n_layer,n_expert,n_used,cache_mb,lanes,dontneed,g_stream,g_verify);

  llama_model_params mp=llama_model_default_params();
  mp.use_extra_bufts=false; mp.load_mode=LLAMA_LOAD_MODE_MMAP;
  llama_model*model=llama_model_load_from_file(model_path.c_str(),mp);
  if(!model){fprintf(stderr,"err model\n");return 1;}
  int fd=open(model_path.c_str(),O_RDONLY); if(fd<0){fprintf(stderr,"err open\n");return 1;}

  Streamer st; st.gg=gg; st.fd=fd; st.n_expert=n_expert; st.n_expert_used=n_used;
  st.layers.resize(n_layer); st.dontneed=dontneed;
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
    std::vector<char>fmt(prompt.size()*2+512);
    int fl=llama_chat_apply_template(tmpl,msgs,1,true,fmt.data(),(int)fmt.size());
    if(fl<0){fprintf(stderr,"tmpl err; raw\n");} else {fmt.resize(fl);prompt.assign(fmt.data(),fmt.size());}
  }
  std::vector<llama_token> toks(prompt.size()+64);
  int nt=llama_tokenize(vocab,prompt.c_str(),(int)prompt.size(),toks.data(),(int)toks.size(),true,false);
  if(nt<0)nt=0; toks.resize(nt);
  { llama_batch b=llama_batch_get_one(toks.data(),(int32_t)toks.size());
    if(llama_decode(ctx,b)!=0){fprintf(stderr,"prefill err\n");return 1;} }
  st.warmed_up=true; st.enable_stream=g_stream;
  if(g_stream && lanes>0) st.pool=new IoPool(&st,lanes,g_verify);
  size_t found=0; for(auto&L:st.layers){if(L.gate.t)found++;if(L.down.t)found++;}
  fprintf(stderr,"warm-up: %zu/%zu experts\n",found,n_layer*2);

  llama_sampler*smpl=llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(smpl,llama_sampler_init_temp(0.0f));
  llama_sampler_chain_add(smpl,llama_sampler_init_greedy());

  int64_t t0=llama_time_us(); std::vector<llama_token>gen=toks;
  for(int i=0;i<n_pred;++i){
    if((int)gen.size()>(int)toks.size()){
      llama_token tok=gen.back(); llama_batch b=llama_batch_get_one(&tok,1);
      if(llama_decode(ctx,b)!=0){fprintf(stderr,"decode err\n");break;}
      restore_all(st);
    }
    llama_token nt2=llama_sampler_sample(smpl,ctx,-1); llama_sampler_accept(smpl,nt2);
    char p[256]; int np=llama_token_to_piece(vocab,nt2,p,sizeof p,0,false);
    if(np>0)fwrite(p,1,(size_t)np,stdout); fflush(stdout);
    gen.push_back(nt2); if(llama_vocab_is_eog(vocab,nt2))break;
  }
  int64_t t1=llama_time_us(); printf("\n");
  size_t ng=gen.size()-toks.size(); double dt=(double)(t1-t0)/1e6;
  fprintf(stderr,"Phase5: %zu tokens, %.2fs, %.2f tok/s\n",ng,dt,ng/dt);
  if(st.cache){double hr=100.0*(double)st.cache->hits/(double)(st.cache->hits+st.cache->misses+1e-9);
    fprintf(stderr,"Cache: hits=%llu misses=%llu hit_rate=%.1f%% disk_reads=%llu\n",
      (unsigned long long)st.cache->hits,(unsigned long long)st.cache->misses,hr,
      (unsigned long long)st.n_expert_reads);}
  else fprintf(stderr,"Cache: disabled. disk_reads=%llu\n",(unsigned long long)st.n_expert_reads);
  fprintf(stderr,"verify: %llu FAILs (0 = all preads == mmap)\n",(unsigned long long)st.n_verify_fail);

  delete st.pool;
  llama_sampler_free(smpl); llama_free(ctx); llama_model_free(model);
  llama_backend_free(); gguf_free(gg); if(fd>=0)close(fd);
  delete[] st.gate_full; delete[] st.up_full; delete[] st.down_full;
  return 0;
}