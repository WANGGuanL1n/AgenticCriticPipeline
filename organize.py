#!/usr/bin/env python3
"""
按任务类型（章节）将生成的图片和 critique 结果整理到 static 目录
章节划分：
  - portrait: 人像 (6张)
  - poster:   文字/海报 (3张)
  - landscape: 风景 (4张)
  - abstract:  抽象/艺术 (2张)
"""
import sys, os, shutil, json, time
sys.path.insert(0, "/tmp")

RESULTS_DIR = "/tmp/gap_critique_results"
STATIC_DIR = f"{RESULTS_DIR}/static"
IMAGES_DIR = f"{RESULTS_DIR}/images"
JSON_PATH = f"{RESULTS_DIR}/critique_results.json"

# ─── 按章节组织 ───
CHAPTERS = {
    "portrait": {
        "title": "人像生成",
        "desc": "人脸、皮肤纹理、解剖学精度、光影质量",
        "indices": [1, 4, 5, 10, 11, 12],
    },
    "poster": {
        "title": "文字与海报",
        "desc": "文字渲染准确度、排版可读性、商业设计",
        "indices": [2, 3, 13],
    },
    "landscape": {
        "title": "风景与自然",
        "desc": "远景细节、气氛、空间关系、色彩控制",
        "indices": [7, 8, 9, 14],
    },
    "abstract": {
        "title": "抽象与艺术表现",
        "desc": "构图创意、几何/反射、艺术叙事",
        "indices": [6, 15],
    },
}


def organize():
    os.makedirs(STATIC_DIR, exist_ok=True)

    # Load JSON results
    if not os.path.exists(JSON_PATH):
        print(f"JSON not found: {JSON_PATH} — batch still running?")
        return

    with open(JSON_PATH) as f:
        results = json.load(f)

    # Build index map
    index_map = {r["idx"]: r for r in results if r.get("status") == "OK"}

    # Generate chapter pages
    chapter_list = []
    for ch_key, ch_info in CHAPTERS.items():
        ch_dir = os.path.join(STATIC_DIR, ch_key)
        os.makedirs(ch_dir, exist_ok=True)

        items = []
        for idx in ch_info["indices"]:
            r = index_map.get(idx)
            if not r:
                continue

            # Copy images to chapter dir
            src_name = os.path.basename(r["src_path"])
            tgt_name = os.path.basename(r["target_path"])

            src_dst = os.path.join(ch_dir, f"{idx:02d}_src_{src_name.split('_',1)[1] if '_' in src_name else src_name}")
            tgt_dst = os.path.join(ch_dir, f"{idx:02d}_target_{tgt_name.split('_',1)[1] if '_' in tgt_name else tgt_name}")

            if os.path.exists(r["src_path"]):
                shutil.copy2(r["src_path"], src_dst)
            if os.path.exists(r["target_path"]):
                shutil.copy2(r["target_path"], tgt_dst)

            items.append({
                "idx": idx,
                "prompt": r["prompt"],
                "src_path": os.path.relpath(src_dst, RESULTS_DIR),
                "target_path": os.path.relpath(tgt_dst, RESULTS_DIR),
                "alignment": r["alignment"],
                "confidence": r["confidence"],
                "artist_consensus": r["artist_consensus"],
                "channels": r.get("channels", {}),
                "directives": r.get("directives", []),
                "artists": r["routing"]["artists"],
            })

        if items:
            chapter_list.append({"key": ch_key, "title": ch_info["title"],
                                "desc": ch_info["desc"], "items": items})

    # ─── Write chapter index ───
    index_path = os.path.join(STATIC_DIR, "index.json")
    with open(index_path, "w") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "chapters": chapter_list}, f,
                  indent=2, ensure_ascii=False)

    # ─── Write per-chapter README.md ───
    for ch in chapter_list:
        readme_path = os.path.join(STATIC_DIR, ch["key"], "README.md")
        with open(readme_path, "w") as f:
            f.write(f"# {ch['title']}\n\n{ch['desc']}\n\n")
            f.write(f"共 {len(ch['items'])} 张\n\n")

            alignments = [it["alignment"] for it in ch["items"]]
            avg = sum(alignments) / len(alignments) if alignments else 0
            f.write(f"| 指标 | 值 |\n|---|---|\n")
            f.write(f"| 数量 | {len(ch['items'])} |\n")
            f.write(f"| 平均 Alignment | {avg:.3f} |\n")
            f.write(f"| 最高 | {max(alignments):.3f} |\n")
            f.write(f"| 最低 | {min(alignments):.3f} |\n\n")

            f.write("---\n\n")
            for it in ch["items"]:
                f.write(f"## {it['idx']}. {it['prompt'][:60]}...\n\n")
                f.write(f"- **Alignment**: {it['alignment']:.3f} (conf={it['confidence']:.3f})\n")
                f.write(f"- **Artists**: {', '.join(it['artists'])}\n")
                f.write(f"- **Target**: `{it['target_path']}`\n")
                f.write(f"- **Source**: `{it['src_path']}`\n\n")

                f.write("**通道分数**:\n\n| 通道 | Score |\n|---|---|\n")
                for ch_name, v in sorted(it["channels"].items()):
                    bar = "█" * max(1, int(v["score"])) + "░" * max(0, 10 - int(v["score"]))
                    f.write(f"| {ch_name} | {v['score']:.0f} {bar} |\n")

                if it["directives"]:
                    f.write(f"\n**修复建议**:\n")
                    for d in it["directives"]:
                        f.write(f"- [{d['action']}] `{d['channel']}`: {d['desc'][:120]}\n")
                f.write("\n---\n\n")

    # ─── Summary ───
    print(f"Organized into {len(chapter_list)} chapters under {STATIC_DIR}/")
    for ch in chapter_list:
        print(f"  {ch['key']}/ ({ch['title']}): {len(ch['items'])} images")
    print(f"\nStatic index: {index_path}")


if __name__ == "__main__":
    organize()
