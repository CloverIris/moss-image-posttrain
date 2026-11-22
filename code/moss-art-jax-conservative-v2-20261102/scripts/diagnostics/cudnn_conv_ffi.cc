// Diagnostic cuDNN backend plans + XLA typed FFI. No cudaMalloc; workspace is a JAX result.
#include <cudnn.h>
#include <cuda_runtime_api.h>
#include "xla/ffi/api/ffi.h"
#include <algorithm>
#include <array>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ffi = xla::ffi;
namespace {
void Check(cudnnStatus_t s) { if(s != CUDNN_STATUS_SUCCESS) throw std::runtime_error(cudnnGetErrorString(s)); }
void CheckCuda(cudaError_t s) { if(s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
std::string Quote(const std::string& s) {
  std::ostringstream o; o << '"';
  for(char c:s) { if(c=='"'||c=='\\')o<<'\\'; if(c=='\n')o<<"\\n";else o<<c; } return o.str()+'"';
}
struct Graph {
  cudnnHandle_t handle=nullptr;
  std::vector<cudnnBackendDescriptor_t> descriptors;
  std::mutex mutex;
  int device=0, dtype=0;
  int64_t n=1,h=0,w=0,c=0,k=0,r=0,cap=0;
  std::string fingerprint;
  ~Graph(){for(auto it=descriptors.rbegin();it!=descriptors.rend();++it)cudnnBackendDestroyDescriptor(*it);if(handle)cudnnDestroy(handle);}
  cudnnBackendDescriptor_t New(cudnnBackendDescriptorType_t type) {
    cudnnBackendDescriptor_t x;Check(cudnnBackendCreateDescriptor(type,&x));descriptors.push_back(x);return x;
  }
};
struct Plan { std::shared_ptr<Graph> graph; cudnnBackendDescriptor_t descriptor;int64_t workspace;std::string key;};
std::map<std::string,std::shared_ptr<Plan>> plans;
std::mutex registry_mutex;
std::string returned_json;
template<class T> void Set(cudnnBackendDescriptor_t d,cudnnBackendAttributeName_t name,cudnnBackendAttributeType_t type,int64_t count,const T* value) {
 Check(cudnnBackendSetAttribute(d,name,type,count,value));
}
template<class T> T GetOne(cudnnBackendDescriptor_t d,cudnnBackendAttributeName_t name,cudnnBackendAttributeType_t type) {
 T value{};int64_t count=0;Check(cudnnBackendGetAttribute(d,name,type,1,&count,&value));if(count!=1)throw std::runtime_error("Missing descriptor value");return value;
}
cudnnBackendDescriptor_t Tensor(Graph& g,int64_t uid,std::array<int64_t,4> dims,std::array<int64_t,4> strides,cudnnDataType_t dtype) {
 auto d=g.New(CUDNN_BACKEND_TENSOR_DESCRIPTOR);int64_t alignment=16;
 Set(d,CUDNN_ATTR_TENSOR_DATA_TYPE,CUDNN_TYPE_DATA_TYPE,1,&dtype);
 Set(d,CUDNN_ATTR_TENSOR_DIMENSIONS,CUDNN_TYPE_INT64,4,dims.data());
 Set(d,CUDNN_ATTR_TENSOR_STRIDES,CUDNN_TYPE_INT64,4,strides.data());
 Set(d,CUDNN_ATTR_TENSOR_UNIQUE_ID,CUDNN_TYPE_INT64,1,&uid);
 Set(d,CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,CUDNN_TYPE_INT64,1,&alignment);Check(cudnnBackendFinalize(d));return d;
}
std::string NumericalName(cudnnBackendNumericalNote_t n) {
 switch(n){
 case CUDNN_NUMERICAL_NOTE_TENSOR_CORE:return "TENSOR_CORE";
 case CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS:return "DOWN_CONVERT_INPUTS";
 case CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION:return "REDUCED_PRECISION_REDUCTION";
 case CUDNN_NUMERICAL_NOTE_FFT:return "FFT";
 case CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC:return "NONDETERMINISTIC";
 case CUDNN_NUMERICAL_NOTE_WINOGRAD:return "WINOGRAD";
 default:return "NOTE_"+std::to_string(int(n));}
}
std::string Build(int64_t h,int64_t w,int64_t c,int64_t k,int64_t r,int dtype,int64_t cap) {
 if(h<1||w<1||c<1||k<1||(r!=1&&r!=3)||cap<0||(dtype!=0&&dtype!=1))throw std::runtime_error("Invalid convolution shape/dtype/cap");
 auto g=std::make_shared<Graph>();g->h=h;g->w=w;g->c=c;g->k=k;g->r=r;g->dtype=dtype;g->cap=cap;
 CheckCuda(cudaGetDevice(&g->device));cudaDeviceProp prop;CheckCuda(cudaGetDeviceProperties(&prop,g->device));
 std::ostringstream fp;fp<<"schema1_cudnn"<<cudnnGetVersion()<<"_gpu";
 for(unsigned char b:prop.uuid.bytes)fp<<std::hex<<std::setw(2)<<std::setfill('0')<<int(b);
 fp<<std::dec<<"_n1_h"<<h<<"_w"<<w<<"_c"<<c<<"_k"<<k<<"_r"<<r<<"_s"<<r<<"_pad"<<r/2<<"_nhwc_ohwi_nhwc_dtype"<<dtype<<"_computeF32_cap"<<cap;
 g->fingerprint=fp.str();Check(cudnnCreate(&g->handle));
 cudnnDataType_t dt=dtype==0?CUDNN_DATA_FLOAT:CUDNN_DATA_HALF,comp=CUDNN_DATA_FLOAT;
 auto x=Tensor(*g,1001,{1,c,h,w},{h*w*c,1,w*c,c},dt);
 auto weight=Tensor(*g,1002,{k,c,r,r},{r*r*c,1,r*c,c},dt);
 auto y=Tensor(*g,1003,{1,k,h,w},{h*w*k,1,w*k,k},dt);
 auto conv=g->New(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
 cudnnConvolutionMode_t mode=CUDNN_CROSS_CORRELATION;int64_t spatial=2;
 std::array<int64_t,2> pad={r/2,r/2},one={1,1};
 Set(conv,CUDNN_ATTR_CONVOLUTION_COMP_TYPE,CUDNN_TYPE_DATA_TYPE,1,&comp);
 Set(conv,CUDNN_ATTR_CONVOLUTION_CONV_MODE,CUDNN_TYPE_CONVOLUTION_MODE,1,&mode);
 Set(conv,CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,CUDNN_TYPE_INT64,1,&spatial);
 Set(conv,CUDNN_ATTR_CONVOLUTION_DILATIONS,CUDNN_TYPE_INT64,2,one.data());
 Set(conv,CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,CUDNN_TYPE_INT64,2,one.data());
 Set(conv,CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,CUDNN_TYPE_INT64,2,pad.data());
 Set(conv,CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,CUDNN_TYPE_INT64,2,pad.data());Check(cudnnBackendFinalize(conv));
 auto op=g->New(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
 float alpha=1.f,beta=0.f;
 Set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&x);
 Set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&weight);
 Set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&y);
 Set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&conv);
 Set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,CUDNN_TYPE_FLOAT,1,&alpha);
 Set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,CUDNN_TYPE_FLOAT,1,&beta);Check(cudnnBackendFinalize(op));
 auto graph=g->New(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
 Set(graph,CUDNN_ATTR_OPERATIONGRAPH_HANDLE,CUDNN_TYPE_HANDLE,1,&g->handle);
 Set(graph,CUDNN_ATTR_OPERATIONGRAPH_OPS,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&op);Check(cudnnBackendFinalize(graph));
 std::ostringstream result;result<<"{\"fingerprint\":"<<Quote(g->fingerprint)<<",\"compute\":\"FLOAT\",\"candidates\":[";
 bool first=true;
 for(auto hm:{CUDNN_HEUR_MODE_A,CUDNN_HEUR_MODE_B,CUDNN_HEUR_MODE_FALLBACK}) {
  auto heur=g->New(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
  Set(heur,CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&graph);
  Set(heur,CUDNN_ATTR_ENGINEHEUR_MODE,CUDNN_TYPE_HEUR_MODE,1,&hm);
  auto hs=cudnnBackendFinalize(heur);
  if(hs!=CUDNN_STATUS_SUCCESS)continue;
  int64_t count=0;Check(cudnnBackendGetAttribute(heur,CUDNN_ATTR_ENGINEHEUR_RESULTS,CUDNN_TYPE_BACKEND_DESCRIPTOR,0,&count,nullptr));
  std::vector<cudnnBackendDescriptor_t> cfgs;for(int64_t i=0;i<count;++i)cfgs.push_back(g->New(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
  Check(cudnnBackendGetAttribute(heur,CUDNN_ATTR_ENGINEHEUR_RESULTS,CUDNN_TYPE_BACKEND_DESCRIPTOR,count,&count,cfgs.data()));
  for(int64_t i=0;i<count;++i) {
   if(!first)result<<',';first=false;
   result<<"{\"heuristic\":"<<int(hm)<<",\"rank\":"<<i;
   try {
    auto cfg=cfgs[i];auto engine=g->New(CUDNN_BACKEND_ENGINE_DESCRIPTOR);int64_t onecount=0;
    Check(cudnnBackendGetAttribute(cfg,CUDNN_ATTR_ENGINECFG_ENGINE,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&onecount,&engine));
    auto index=GetOne<int64_t>(engine,CUDNN_ATTR_ENGINE_GLOBAL_INDEX,CUDNN_TYPE_INT64);
    std::array<cudnnBackendNumericalNote_t,32> notes;int64_t note_count=0;
    Check(cudnnBackendGetAttribute(engine,CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,CUDNN_TYPE_NUMERICAL_NOTE,notes.size(),&note_count,notes.data()));
    result<<",\"engine\":"<<index<<",\"numerical_notes\":[";
    std::string reject;
    for(int64_t j=0;j<note_count;++j) {
     if(j)result<<',';result<<Quote(NumericalName(notes[j]));
     if(notes[j]==CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS||notes[j]==CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION)reject="implicit_precision_reduction";
     if(dtype==0 && notes[j]==CUDNN_NUMERICAL_NOTE_TENSOR_CORE)reject="strict_fp32_rejects_tensor_core_tf32";
    }
    result<<"]";
    int64_t knobs_count=0;constexpr int64_t max_knobs=64;
    std::vector<cudnnBackendDescriptor_t> knobs;for(int64_t j=0;j<max_knobs;++j)knobs.push_back(g->New(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR));
    Check(cudnnBackendGetAttribute(cfg,CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,CUDNN_TYPE_BACKEND_DESCRIPTOR,max_knobs,&knobs_count,knobs.data()));
    std::ostringstream knobkey;result<<",\"knobs\":[";
    for(int64_t j=0;j<knobs_count;++j){
     auto type=GetOne<cudnnBackendKnobType_t>(knobs[j],CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,CUDNN_TYPE_KNOB_TYPE);
     auto value=GetOne<int64_t>(knobs[j],CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,CUDNN_TYPE_INT64);
     if(j)result<<',';result<<"["<<int(type)<<","<<value<<"]";knobkey<<"_knob"<<int(type)<<"="<<value;
    }result<<"]";
    if(!reject.empty()){result<<",\"status\":\"rejected\",\"reason\":"<<Quote(reject)<<"}";continue;}
    auto plan=g->New(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    Set(plan,CUDNN_ATTR_EXECUTION_PLAN_HANDLE,CUDNN_TYPE_HANDLE,1,&g->handle);
    Set(plan,CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&cfg);
    auto status=cudnnBackendFinalize(plan);
    if(status!=CUDNN_STATUS_SUCCESS){result<<",\"status\":\"rejected\",\"reason\":"<<Quote(cudnnGetErrorString(status))<<"}";continue;}
    auto workspace=GetOne<int64_t>(plan,CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,CUDNN_TYPE_INT64);
    result<<",\"workspace_bytes\":"<<workspace;
    if(workspace>cap){result<<",\"status\":\"rejected\",\"reason\":\"workspace_cap\"}";continue;}
    std::string key=g->fingerprint+"_engine"+std::to_string(index)+knobkey.str();
    if(!plans.count(key))plans[key]=std::make_shared<Plan>(Plan{g,plan,workspace,key});
    result<<",\"status\":\"ready\",\"key\":"<<Quote(key)<<"}";
   }catch(const std::exception& e){result<<",\"status\":\"rejected\",\"reason\":"<<Quote(e.what())<<"}";}
  }
 }
 result<<"]}";return result.str();
}
ffi::Error Execute(cudaStream_t stream,std::string_view key,ffi::AnyBuffer input,ffi::AnyBuffer weight,ffi::Result<ffi::AnyBuffer> output,ffi::ResultBuffer<ffi::U8> scratch) {
 try{
  std::shared_ptr<Plan> p;
  {std::lock_guard<std::mutex> lock(registry_mutex);auto it=plans.find(std::string(key));if(it==plans.end())return ffi::Error::InvalidArgument("Plan not prebuilt; call prepare before compiling/executing");p=it->second;}
  auto& g=*p->graph;int device;CheckCuda(cudaGetDevice(&device));
  if(device!=g.device)return ffi::Error::InvalidArgument("Plan device mismatch");
  auto dt=g.dtype==0?ffi::DataType::F32:ffi::DataType::F16;
  if(input.element_type()!=dt||weight.element_type()!=dt||output->element_type()!=dt)return ffi::Error::InvalidArgument("Plan dtype mismatch");
  if(input.dimensions().size()!=4||weight.dimensions().size()!=4||output->dimensions().size()!=4)return ffi::Error::InvalidArgument("Rank mismatch");
  std::array<int64_t,4> xs={1,g.h,g.w,g.c},ws={g.k,g.r,g.r,g.c},ys={1,g.h,g.w,g.k};
  if(!std::equal(xs.begin(),xs.end(),input.dimensions().begin())||!std::equal(ws.begin(),ws.end(),weight.dimensions().begin())||!std::equal(ys.begin(),ys.end(),output->dimensions().begin()))return ffi::Error::InvalidArgument("Plan shape mismatch");
  if(p->workspace>g.cap||scratch->size_bytes()<size_t(p->workspace))return ffi::Error::InvalidArgument("Explicit scratch does not satisfy bounded plan workspace");
  std::lock_guard<std::mutex> lock(g.mutex);
  Check(cudnnSetStream(g.handle,stream));
  cudnnBackendDescriptor_t pack=nullptr;Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,&pack));
  struct Cleanup{cudnnBackendDescriptor_t d;~Cleanup(){if(d)cudnnBackendDestroyDescriptor(d);}} cleanup{pack};
  void* pointers[]={input.untyped_data(),weight.untyped_data(),output->untyped_data()};int64_t uids[]={1001,1002,1003};
  void* workspace=p->workspace?scratch->typed_data():nullptr;
  Set(pack,CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,CUDNN_TYPE_VOID_PTR,3,pointers);
  Set(pack,CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,CUDNN_TYPE_INT64,3,uids);
  Set(pack,CUDNN_ATTR_VARIANT_PACK_WORKSPACE,CUDNN_TYPE_VOID_PTR,1,&workspace);Check(cudnnBackendFinalize(pack));
  Check(cudnnBackendExecute(g.handle,p->descriptor,pack));return ffi::Error::Success();
 }catch(const std::exception& e){return ffi::Error::Internal(e.what());}
}
}  // namespace
extern "C" const char* MossBuildPlans(int64_t h,int64_t w,int64_t c,int64_t k,int64_t r,int dtype,int64_t cap) {
 std::lock_guard<std::mutex> lock(registry_mutex);
 try{returned_json=Build(h,w,c,k,r,dtype,cap);}
 catch(const std::exception&e){returned_json="{\"error\":"+Quote(e.what())+"}";}
 return returned_json.c_str();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(MossCudnnConv,Execute,
 ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Attr<std::string_view>("plan_key")
 .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::Buffer<ffi::U8>>());
