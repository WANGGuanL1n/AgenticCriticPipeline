#!/usr/bin/env python3
"""
Batch Generator + Critique — 15 prompts × Image-2(target) + SenseNova-U1(src) → critique → 文档

用法（跳板机上）:
    python3 /tmp/gap_critique/batch_critique.py
输出: /tmp/gap_critique_results/critique_report.md
"""
import sys, os, json, time, yaml
sys.path.insert(0, "/tmp")

from gap_critique import GapPipeline, PipelineState, ImagePairGenerator
from gap_critique.gap_types import GapChannel

# ─── 从 api_keys.yaml 读取配置 ───
API_KEYS_FILE = os.path.expanduser("~/api_keys.yaml")
with open(API_KEYS_FILE) as f:
    keys = yaml.safe_load(f)

IMAGE2_KEY = keys.get("u1_image2", "")
IMAGE2_BASE = keys.get("image2_base_url", "https://api.openai.com/v1")
IMAGE2_MODEL = keys.get("image2_model", "gpt-image-2-pro-all")
SENSENOVA_KEY = keys.get("sensenova_api_key", "")
SENSENOVA_BASE = keys.get("sensenova_base_url", "")

print(f"Image-2: base={IMAGE2_BASE}, model={IMAGE2_MODEL}")
print(f"SenseNova-U1: {'configured' if SENSENOVA_KEY else 'NOT configured (using Gaussian noise placeholder)'}")

# ─── 15 个 Prompt ───
PROMPTS = [
    {"prompt": "Close portrait of an elderly woman by a farmhouse window, textured skin, gentle smile, warm natural light, emotional documentary look. The portrait should feel polished and natural, with sharp eyes, realistic skin texture, accurate facial anatomy, and premium lighting that keeps the face as the main focus.",
     "width": 1536, "height": 2720, "task_type": "portrait"},
    {"prompt": 'A greeting card on a wooden desk with readable Chinese text "生日快乐", flowers beside it, simple celebratory styling. Any text in the image must be rendered exactly as written in quotation marks, with correct spelling, clean typography, and strong readability.',
     "width": 1536, "height": 2720, "task_type": "poster"},
    {"prompt": 'A neon bar sign that clearly reads "OPEN LATE", dark interior, moody reflections, easy text rendering. Any text in the image must be rendered exactly as written in quotation marks, with correct spelling, clean typography, and strong readability.',
     "width": 2720, "height": 1536, "task_type": "poster"},
    {"prompt": "Tight portrait of a surfer with saltwater droplets on tan skin, sunlit face, windblown hair, natural freckles, vivid blue eyes, coastal realism.",
     "width": 2048, "height": 2048, "task_type": "portrait"},
    {"prompt": "Documentary-style portrait of a street boxer in a dim gym, bruised eyebrow, determined look, sweat sheen, precise facial anatomy, gritty realism.",
     "width": 2048, "height": 2048, "task_type": "portrait"},
    {"prompt": "An expressive portrait with mirrored reflections fragmenting the face into geometric shapes, sophisticated editorial art style.",
     "width": 2048, "height": 2048, "task_type": "abstract"},
    {"prompt": "Lavender fields stretching to the horizon under a pastel sunset, a small stone farmhouse, highly detailed flowers, romantic countryside scene.",
     "width": 2048, "height": 2048, "task_type": "landscape"},
    {"prompt": "Stormy seascape with waves crashing against a lighthouse, dramatic sky, realistic water motion, moody coastal photography.",
     "width": 2048, "height": 2048, "task_type": "landscape"},
    {"prompt": "Tropical beach with turquoise water, black volcanic rocks, swaying palms, bright noon sun, ultra-clean travel photography, balanced square crop.",
     "width": 2048, "height": 2048, "task_type": "landscape"},
    {"prompt": "A winter portrait of a traveler in a wool coat and scarf, rosy cheeks, frosty air, bright eyes, elegant vertical framing. The portrait should feel polished and natural, with sharp eyes, realistic skin texture, accurate facial anatomy, and premium lighting that keeps the face as the main focus.",
     "width": 1536, "height": 2720, "task_type": "portrait"},
    {"prompt": "A violinist standing beneath stage lights, elegant bow arm and grounded posture, full body visible.",
     "width": 1536, "height": 2720, "task_type": "portrait"},
    {"prompt": "A woman seen through rain-covered glass from head to waist, elongated reflections, emotional and refined visual poetry.",
     "width": 1536, "height": 2720, "task_type": "abstract"},
    {"prompt": 'A cafe takeaway cup standing on a clean counter, with the sleeve text rendered clearly as "SenseNova-U1", realistic paper texture, morning light, cozy interior blur, and no additional readable menu boards.',
     "width": 1536, "height": 2720, "task_type": "poster"},
    {"prompt": "Cherry blossom trees arching over a temple stairway, petals drifting downward, elegant spring vertical composition.",
     "width": 1536, "height": 2720, "task_type": "landscape"},
    {"prompt": "A quiet church interior with sunlight touching one empty chair, tall architecture, meditative artistic stillness. The final image should feel intentional and refined, with a clear artistic mood, thoughtful composition, nuanced color control, and a gallery-like sense of visual storytelling.",
     "width": 1536, "height": 2720, "task_type": "abstract"},
]

OUTPUT_DIR = "/tmp/gap_critique_results"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)


def main():
    gen = ImagePairGenerator(
        image2_key=IMAGE2_KEY,
        image2_base=IMAGE2_BASE,
        image2_model=IMAGE2_MODEL,
        sensenova_key=SENSENOVA_KEY,
        sensenova_base=SENSENOVA_BASE,
        output_dir=f"{OUTPUT_DIR}/images",
    )

    results = []
    total = len(PROMPTS)

    for i, item in enumerate(PROMPTS):
        prompt = item["prompt"]
        w, h = item["width"], item["height"]
        task_type = item["task_type"]
        short = prompt[:60].replace('"', "'").replace('\n', ' ')

        print(f"\n[{i+1}/{total}] {short}... ({w}x{h}, {task_type})")
        t0 = time.time()

        # Generate target (gpt-image-2)
        print(f"  Generating target...")
        try:
            pair = gen.generate(prompt=prompt, width=w, height=h)
        except Exception as e:
            print(f"  [ERROR] generation failed: {e}")
            results.append({"idx": i+1, "prompt": prompt, "status": "GEN_FAILED", "error": str(e)})
            continue

        gen_time = time.time() - t0
        print(f"  src={os.path.basename(pair.src_path)}  target={os.path.basename(pair.target_path)}  ({gen_time:.0f}s)")

        # Critique
        print(f"  Running critique...")
        t1 = time.time()
        pipeline = GapPipeline(task_type=task_type)
        state = PipelineState()
        result = pipeline.evaluate(
            src_path=pair.src_path,
            target_path=pair.target_path,
            target_descriptor=prompt,
            state=state,
        )
        critique_time = time.time() - t1
        gap = result["gap"]

        print(f"  alignment={gap.overall_alignment:.3f} consensus={gap.artist_consensus:.3f} "
              f"stop={result['should_stop']}")

        results.append({
            "idx": i + 1, "prompt": prompt, "width": w, "height": h,
            "task_type": task_type, "status": "OK",
            "src_path": pair.src_path, "target_path": pair.target_path,
            "alignment": gap.overall_alignment, "confidence": gap.overall_confidence,
            "artist_consensus": gap.artist_consensus, "should_stop": result["should_stop"],
            "routing": {"artists": result["routing"].selected_artist_critics,
                       "on_domain": result["routing"].artist_on_domain},
            "channels": {ch.name: {"score": gs.score, "conf": gs.confidence}
                        for ch, gs in gap.channel_scores.items()},
            "directives": [{"action": d.action.value, "channel": d.channel.name,
                           "priority": d.priority, "desc": d.description}
                          for d in gap.repair_directives[:3]] if gap.repair_directives else [],
            "planner_observation": result["planner_observation"],
            "gen_time_s": gen_time, "critique_time_s": critique_time,
        })

    # ─── Report ───
    report_path = f"{OUTPUT_DIR}/critique_report.md"
    with open(report_path, "w") as f:
        f.write("# Gap Critique 批量评估报告\n\n")
        f.write(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**模型**: target=gpt-image-2-pro-all, src=SenseNova-U1 (高斯噪声占位)\n")
        f.write(f"**样本数**: {total}\n\n")

        ok = [r for r in results if r["status"] == "OK"]
        alignments = [r["alignment"] for r in ok]

        f.write("## 汇总\n\n")
        f.write(f"| 指标 | 值 |\n|---|---|\n")
        f.write(f"| 成功 | {len(ok)}/{total} |\n")
        if alignments:
            f.write(f"| Avg Alignment | {sum(alignments)/len(alignments):.3f} |\n")
            f.write(f"| Range | {min(alignments):.3f} - {max(alignments):.3f} |\n")

        from collections import defaultdict
        by_task = defaultdict(list)
        for r in ok:
            by_task[r.get("task_type", "?")].append(r["alignment"])
        f.write("\n| 任务类型 | 数量 | Avg Alignment |\n|---|---|---|\n")
        for tt, vals in sorted(by_task.items()):
            f.write(f"| {tt} | {len(vals)} | {sum(vals)/len(vals):.3f} |\n")

        f.write("\n---\n\n## 逐项详情\n\n")
        for r in results:
            f.write(f"### {r['idx']}. [{r.get('task_type','?')}] {r['prompt'][:80]}...\n\n")
            if r["status"] != "OK":
                f.write(f"**状态**: {r['status']} — {r.get('error','')}\n\n---\n\n")
                continue
            f.write(f"**尺寸**: {r['width']}×{r['height']}\n\n")
            f.write(f"| 指标 | 值 |\n|---|---|\n")
            f.write(f"| Alignment | {r['alignment']:.3f} |\n")
            f.write(f"| Confidence | {r['confidence']:.3f} |\n")
            f.write(f"| Artist Consensus | {r['artist_consensus']:.3f} |\n")
            f.write(f"| Should Stop | {r['should_stop']} |\n")
            f.write(f"| Artists | {', '.join(r['routing']['artists'])} |\n")
            f.write(f"| 生成耗时 | {r['gen_time_s']:.0f}s | 评估耗时 | {r['critique_time_s']:.0f}s |\n\n")

            f.write("**通道分数**:\n\n| 通道 | Score | Confidence |\n|---|---|---|\n")
            for ch, v in sorted(r["channels"].items()):
                bar = "█" * max(1, int(v["score"])) + "░" * max(0, 10 - int(v["score"]))
                f.write(f"| {ch} | {v['score']:.0f} {bar} | {v['conf']:.2f} |\n")

            if r["directives"]:
                f.write(f"\n**修复指令**:\n\n")
                for d in r["directives"]:
                    f.write(f"- [{d['action']}] `{d['channel']}` prio={d['priority']:.2f}: {d['desc'][:120]}\n")
            f.write(f"\n---\n\n")

    json_path = f"{OUTPUT_DIR}/critique_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"Report: {report_path}")
    print(f"JSON:   {json_path}")
    if alignments:
        print(f"Avg alignment: {sum(alignments)/len(alignments):.3f} "
              f"({min(alignments):.3f}-{max(alignments):.3f})")


if __name__ == "__main__":
    main()
