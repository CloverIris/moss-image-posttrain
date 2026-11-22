"""Read-only VAE 1024 MAC/shape/HLO audit; standard library only, no GPU."""
from __future__ import annotations
import argparse,csv,hashlib,json,re,struct
from collections import defaultdict
from pathlib import Path

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-dir",default="$MODEL_ROOT/MOSS-Image-SFT")
    ap.add_argument("--output",default="reports/vae_layer_macs.json")
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[2]
    model=Path(args.model_dir)
    wp=model/"vae/diffusion_pytorch_model.safetensors"
    with wp.open("rb") as f:
        size=struct.unpack("<Q",f.read(8))[0]
        header_bytes=f.read(size); weights=json.loads(header_bytes)
    cfg=json.loads((model/"vae/config.json").read_text())
    assert cfg["decoder_block_out_channels"]==[96,192,384,384]
    assert cfg["layers_per_block"]==2 and cfg["latent_channels"]==32
    entries=[]
    def conv(prefix,res,stage,layer):
        co,ci,kh,kw=weights[prefix+".weight"]["shape"]
        mac=res*res*co*ci*kh*kw
        entries.append(dict(kind="conv",prefix=prefix,layer=layer,stage=stage,
            batch=1,height=res,width=res,input_channels=ci,output_channels=co,
            kernel=[kh,kw],weight_shape_oihw=[co,ci,kh,kw],
            input_shape_nchw=[1,ci,res,res],output_shape_nchw=[1,co,res,res],
            mac=mac,gmac=mac/1e9,flop=2*mac,nominal_mac_definition="N*Hout*Wout*Cout*Cin*Kh*Kw, including padded edge taps"))
    def resnet(prefix,res,stage):
        for name in ["conv1","conv2","conv_shortcut"]:
            if prefix+"."+name+".weight" in weights:
                conv(prefix+"."+name,res,stage,prefix)
    conv("post_quant_conv",128,"input","post_quant_conv")
    conv("decoder.conv_in",128,"input","decoder.conv_in")
    resnet("decoder.mid_block.resnets.0",128,"mid_pre")
    for name in ["to_q","to_k","to_v","to_out.0"]:
        prefix="decoder.mid_block.attentions.0."+name
        co,ci=weights[prefix+".weight"]["shape"]; mac=16384*ci*co
        entries.append(dict(kind="linear",prefix=prefix,layer="decoder.mid_block.attentions.0",
            stage="mid_attention",batch=1,tokens=16384,input_channels=ci,
            output_channels=co,weight_shape_oi=[co,ci],mac=mac,gmac=mac/1e9,flop=2*mac))
    for name in ["QK","PV"]:
        mac=16384*16384*384
        entries.append(dict(kind="attention_matmul",prefix="decoder.mid_block.attentions.0."+name,
            layer="decoder.mid_block.attentions.0",stage="mid_attention",batch=1,heads=1,
            query_tokens=16384,key_tokens=16384,head_dim=384,query_chunk=256,
            chunks=64,mac=mac,gmac=mac/1e9,flop=2*mac))
    resnet("decoder.mid_block.resnets.1",128,"mid_post")
    for block,res in enumerate([128,256,512,1024]):
        for ri in range(3):resnet(f"decoder.up_blocks.{block}.resnets.{ri}",res,f"up{block}")
        if block<3:
            prefix=f"decoder.up_blocks.{block}.upsamplers.0.conv"
            conv(prefix,res*2,f"up{block}",f"decoder.up_blocks.{block}.upsamplers.0")
    conv("decoder.conv_out",1024,"output","decoder.conv_out")
    conv_entries=[e for e in entries if e["kind"]=="conv"]
    assert len(conv_entries)==36
    assert set(e["prefix"]+".weight" for e in conv_entries)=={
        k for k,v in weights.items() if k!="__metadata__" and
        k.endswith(".weight") and len(v["shape"])==4 and
        (k.startswith("decoder.") or k.startswith("post_quant_conv."))}
    hlo=[]
    for label,rel in [
        ("fp32","outputs/vae_precision/fp32_reference/optimized.hlo.txt"),
        ("full_fp16","outputs/vae_precision/full_fp16_all/optimized.hlo.txt")]:
        path=root/rel; text=path.read_text(); found=[]
        for lineno,line in enumerate(text.splitlines(),1):
            if 'custom_call_target="__cudnn$conv' not in line:continue
            matches=[e for e in conv_entries if "%p__"+e["prefix"].replace(".","_")+"_bias__." in line]
            assert len(matches)==1,(lineno,matches)
            e=matches[0]
            m=re.search(r"= \((f16|f32|bf16)\[([\d,]+)\]",line)
            shape=list(map(int,m.group(2).split(",")))
            expected=[1,e["height"],e["width"],e["output_channels"]]
            assert shape==expected,(e["prefix"],shape,expected)
            backend=json.loads(line.split("backend_config=",1)[1])
            found.append(dict(prefix=e["prefix"],hlo_line=lineno,result_dtype=m.group(1),
                result_shape_nhwc=shape,custom_call_target=re.search(r'custom_call_target="([^"]+)"',line).group(1),
                scratch_bytes=int(re.search(r"u8\[(\d+)\]",line).group(1)),
                conv_backend_config=backend.get("cudnn_conv_backend_config"),
                instruction_sha256=hashlib.sha256(line.encode()).hexdigest()))
        assert len(found)==36 and len({e["prefix"] for e in found})==36
        hlo.append(dict(label=label,path=str(path),sha256=sha(path),convolutions=found,
            fmha_custom_calls=text.count('custom_call_target="__cudnn$fmha'),
            precision_limitation="Absent algorithm fields in textual proto JSON do not prove no algorithm or no Tensor Cores."))
    total=sum(e["mac"] for e in entries)
    stage_macs=defaultdict(int);layer_macs=defaultdict(int)
    for e in entries: stage_macs[e["stage"]]+=e["mac"];layer_macs[e["layer"]]+=e["mac"]
    old=root/"reports/vae_stage_profile.json"; pd=json.loads(old.read_text())
    med=[s for s in pd["stages"] if s["autotune_level"]==0]
    measured_total=sum(s["warm_median_seconds"] for s in med)
    stage=[]
    for s in med:
        mac=stage_macs[s["name"]]; sec=s["warm_median_seconds"]
        stage.append(dict(name=s["name"],mac=mac,gmac=mac/1e9,nominal_mac_share=mac/total,
            separate_stage_warm_median_seconds=sec,separate_stage_time_share=sec/measured_total,
            equivalent_tflop_per_second=2*mac/sec/1e12))
    data=dict(status="completed",scope="Static 1024 decoder shape/MAC audit plus historical separately compiled FP32 stage timing; no new GPU execution",
        source_hashes={str(p.relative_to(root)):sha(p) for p in [
            Path(__file__).resolve(),root/"vendor/moss_jax/vae.py",root/"scripts/diagnostics/vae_full_fp16.py",old]},
        model_config_sha256=sha(model/"vae/config.json"),
        model_weight_header_sha256=hashlib.sha256(header_bytes).hexdigest(),
        convolution_count=36,linear_count=4,attention_matmul_count=2,
        nominal_total_mac=total,nominal_total_tflop=2*total/1e12,
        conv_mac=sum(e["mac"] for e in entries if e["kind"]=="conv"),
        all_entries=entries,layers=[dict(name=k,mac=v,gmac=v/1e9) for k,v in layer_macs.items()],
        stages=stage,separately_compiled_stage_median_sum_seconds=measured_total,
        late_up1_up2_up3_mac_share=sum(stage_macs[k] for k in ["up1","up2","up3"])/total,
        late_up1_up2_up3_separate_time_share=sum(s["warm_median_seconds"] for s in med if s["name"] in ["up1","up2","up3"])/measured_total,
        compiled_hlo_evidence=hlo,limitations=[
            "MACs are nominal dense work with one multiply-add=2 FLOP; excludes GN, nonlinearities, bias/residual, softmax, reshape/layout and memory traffic.",
            "The MAC count is not a hardware instruction count; padding, fusion and selected algorithms may alter actual work.",
            "Stage times came from separately compiled stage JITs, not whole-graph kernel attribution; do not add atomic and whole-layer times together.",
            "Whole-layer boundaries can change cuDNN bias/residual fusion and layouts; synchronous dispatch adds overhead.",
            "Attention QK/PV covers all16384 queries and keys with one384-dimensional head,64 chunks of256 queries; a single chunk times64 is only an estimate.",
            "No proof about Tensor Core usage can be drawn solely from HLO absence of algorithm fields or FLOP throughput."])
    out=root/args.output;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(data,indent=2)+"\n")
    csvpath=out.with_suffix(".csv")
    cols=["kind","prefix","layer","stage","height","width","input_channels","output_channels","kernel","mac","gmac"]
    with csvpath.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore");w.writeheader();w.writerows(entries)
    print(json.dumps({k:data[k] for k in ["status","convolution_count","nominal_total_tflop","separately_compiled_stage_median_sum_seconds","late_up1_up2_up3_mac_share","late_up1_up2_up3_separate_time_share"]}))
    print("Wrote",out,csvpath)
if __name__=="__main__":main()
