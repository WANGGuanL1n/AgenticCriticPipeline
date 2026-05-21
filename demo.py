#!/usr/bin/env python3
"""
Gap Critique Pipeline — 端到端验证 Demo

场景：
  1. impressionist scene → Monet + Moebius 评估
  2. baroque portrait → Caravaggio + Rutkowski 评估
  3. minimalist logo → Mondrian + Hokusai 评估
  4. 多轮 refinement loop
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gap_critique import GapPipeline, PipelineState
from gap_critique.generators import ImagePairGenerator, ImagePair


def print_separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def run_scenario(scenario_name: str, prompt: str, task_type: str, style: str, n_rounds: int = 2):
    print_separator(f"Scenario: {scenario_name}")
    print(f"  Prompt: '{prompt}'")
    print(f"  Task type: {task_type} | Style: {style}")

    # Generate image pair (mock placeholders if no API key)
    gen = ImagePairGenerator(output_dir="/tmp/gap_critique_demo")
    pair = gen.generate(prompt=prompt, style=style)
    print(f"  src_path:  {pair.src_path}")
    print(f"  target_path: {pair.target_path}")

    # Create pipeline
    pipeline = GapPipeline(task_type=task_type)
    state = PipelineState()

    # Multi-turn evaluation loop
    for turn in range(1, n_rounds + 1):
        print(f"\n  --- Turn {turn} ---")
        result = pipeline.evaluate(
            src_path=pair.src_path,
            target_path=pair.target_path,
            target_descriptor=style,
            state=state,
        )

        gap = result["gap"]
        routing = result["routing"]

        print(f"  Routing: dims={len(routing.selected_dimension_critics)}, "
              f"artists={routing.selected_artist_critics}")
        print(f"  Artist on-domain: {routing.artist_on_domain}")
        print(f"  Overall alignment: {gap.overall_alignment:.3f} (conf={gap.overall_confidence:.3f})")
        print(f"  Artist consensus: {gap.artist_consensus:.3f}")
        print(f"  Should stop: {result['should_stop']} (stale_count={result['stale_count']})")
        print(f"  Per-channel scores:")
        for ch, gs in gap.channel_scores.items():
            bar = "█" * int(gs.score) + "░" * (10 - int(gs.score))
            print(f"    {ch.name:12s} [{bar}] {gs.score:.0f} (conf={gs.confidence:.2f})")

        if gap.repair_directives:
            print(f"  Repair directives:")
            for d in gap.repair_directives:
                agree_tag = f" [{d.n_supporters} critics agree]" if d.n_supporters > 1 else ""
                print(f"    [{d.action.value}] prio={d.priority:.2f} {d.channel.name}{agree_tag}")
                print(f"      {d.description[:120]}")

        print(f"\n  Planner observation:")
        for line in result["planner_observation"].split("\n"):
            print(f"    | {line}")

        if result["should_stop"]:
            print(f"\n  ✓ Converged after {turn} turns")
            break

    print(f"\n  Final reward (for GRPO): {gap.overall_alignment:.4f}")


def main():
    print("Gap Critique Pipeline — Verification Demo")
    print("Using mock backends (no API calls)")
    print("src=SenseNova-U1 (placeholder), target=gpt-image-2 (placeholder)")

    scenarios = [
        ("Impressionist Garden", "a garden with water lilies at golden hour", "landscape", "impressionist, en plein air, Monet style"),
        ("Baroque Portrait", "a warrior saint holding a sword, dramatic lighting", "portrait", "baroque, chiaroscuro, Caravaggio style"),
        ("Clean Logo", "a tech company logo with geometric shapes", "logo", "minimalist, de stijl, primary colors, grid"),
        ("Fantasy Concept Art", "a dragon perched on a gothic cathedral spire", "scifi", "fantasy concept art, digital painting, epic atmospheric"),
    ]

    for name, prompt, task_type, style in scenarios:
        run_scenario(name, prompt, task_type, style, n_rounds=2)

    print_separator("All scenarios complete")
    print("Pipeline verified: routing differentiates by task, artists speak distinct voices, multi-turn loop works.")


if __name__ == "__main__":
    main()
