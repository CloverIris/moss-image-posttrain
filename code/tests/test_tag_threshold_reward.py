import json
from pathlib import Path
import numpy as np
import pytest
from moss_art.tag_reward import preprocess_file,vocabulary,hard_scores,clean_prompt,source_records


def small_metadata(tmp_path,per_tag=None):
    c={"tags":["long_hair","black_hair","anime_style"],"tags_split":[["general",2],["style",1]],
       "category_best_threshold":{"general":.2,"style":.7},"tags_best_threshold":per_tag}
    (tmp_path/"config.json").write_text(json.dumps(c));return tmp_path


def test_header_escapes_weighted_tags_dedup_and_unknown_log(tmp_path):
    meta=tmp_path/"metadata";meta.mkdir();small_metadata(meta)
    p=tmp_path/"input.txt";p.write_text(r"prompt:\n(long_hair:1.2), black hair, unknown, long hair"+"\n")
    reports=preprocess_file(p,tmp_path/"out.jsonl",tmp_path/"report.json",meta)
    out=json.loads((tmp_path/"out.jsonl").read_text())
    assert out["prompt"]=="prompt:\n(long_hair:1.2), black hair"
    assert out["metadata"]["target_tags"]==["long_hair","black_hair"]
    assert [x["reason"] for x in reports[0]["dropped"]]==["unknown","duplicate"]
    assert p.read_text()==r"prompt:\n(long_hair:1.2), black hair, unknown, long hair"+"\n"
    with pytest.raises(FileExistsError):preprocess_file(p,tmp_path/"out.jsonl",tmp_path/"newreport.json",meta)


def test_multiple_literal_and_actual_newline_records_and_negative_preserved(tmp_path):
    p=tmp_path/"input.txt";p.write_text("prompt:\nlong hair, unknown\nnegative prompt:\nworst quality\nprompt:\nblack hair")
    rows=source_records(p);assert len(rows)==2
    _,ids,_,_=vocabulary(small_metadata(tmp_path))
    clean,tags,_=clean_prompt(rows[0]["prompt"],ids)
    assert "negative prompt:\nworst quality" in clean and tags==["long_hair"]
    p.write_text(r"prompt:\nlong hair"+"\n"+r"prompt:\nblack hair"+"\n")
    assert len(source_records(p))==2
    with pytest.raises(ValueError,match="No supported"):clean_prompt("prompt:\nunknown",ids)


def test_thresholds_strict_greater_than_and_iou_union(tmp_path):
    meta=small_metadata(tmp_path)
    probs=np.array([[.2,.9,.8],[.9,.9,.8]],np.float32)
    scores=hard_scores(probs,[["long_hair"],["long_hair"]],meta)
    assert scores[0]["coverage"]==scores[0]["reward"]==0
    assert scores[1]["coverage"]==1 and scores[1]["iou"]==.5
    assert scores[1]["predicted_tags"]==["long_hair","black_hair"]
    assert hard_scores(probs,[["long_hair"],["long_hair"]],meta,"iou")[1]["reward"]==.5
    small_metadata(tmp_path,[.1,.95,.9]);scores=hard_scores(probs,[["long_hair","black_hair"]]*2,meta)
    assert scores[0]["coverage"]==.5 and scores[0]["predicted_count"]==1
    with pytest.raises(ValueError):hard_scores(probs,[["unknown"]]*2,meta)


def test_tensorboard_round_trip(tmp_path):
    from moss_art.tensorboard_log import ScalarLogger
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    with ScalarLogger({"logging":{"tensorboard":True,"log_dir":""}},tmp_path) as logger:
        logger.write("eval",0,{"reward_mean":.4})
        logger.write("eval",2,{"reward_mean":.6})
    events=EventAccumulator(str(tmp_path/"tensorboard")).Reload().Scalars("eval/reward_mean")
    assert [x.step for x in events]==[0,2]
    np.testing.assert_allclose([x.value for x in events],[.4,.6])
