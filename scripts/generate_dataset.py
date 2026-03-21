"""Prompt-based dataset generator for StoryWeaver NLG fine-tuning.

直接批量发送构造好的 prompt 到 LLM ，收集 (prompt → response) 对，
生成 ms-swift 兼容的 ChatML 格式 .jsonl 训练数据。

生成两个子任务数据集:
  1. story_generation.jsonl  - 故事续写 (Opening + Continue)
  2. option_generation.jsonl - 选项生成 (JSON structured output)

与 self-play 版本相比，本脚本不运行 GameEngine，而是直接构造多样化上下文并批量调用 LLM，
速度更快，可轻松生成 500~1000+ 条数据。

用法:
  python scripts/generate_dataset.py                      # 默认 300 条 story + 300 条 option
  python scripts/generate_dataset.py --story 500 --option 500
  python scripts/generate_dataset.py --output data/custom
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Prompt templates (imported from project) ───────────────────
from src.nlg.prompt_templates import (
    SYSTEM_PROMPT,
    OPENING_PROMPT,
    STORY_CONTINUE_PROMPT,
    OPTION_GENERATION_PROMPT,
)


# ── Data pools for diverse context generation ──────────────────

GENRES = [
    "fantasy", "science fiction", "cyberpunk", "horror", "mystery",
    "post-apocalyptic", "steampunk", "noir detective", "pirate adventure",
    "ancient mythology", "western frontier", "underwater civilization",
    "time travel", "space opera", "dark fantasy", "urban fantasy",
    "survival", "political intrigue", "haunted mansion", "heist",
]

INTENTS = ["action", "dialogue", "explore", "use_item", "ask_info", "rest", "trade", "other"]

EMOTIONS = [
    "neutral", "excited", "anxious", "angry", "sad", "curious",
    "determined", "fearful", "joyful", "suspicious", "desperate", "hopeful",
]

RISK_LEVELS = ["low", "medium", "high"]

# Pre-written kg_summary snippets (realistic game world states)
KG_SUMMARIES = [
    "Entities: hero (person, importance: 0.9), dark_forest (location, importance: 0.7), ancient_sword (item, importance: 0.8). Relations: hero -> located_at -> dark_forest, hero -> possesses -> ancient_sword.",
    "Entities: captain (person, importance: 0.85), starship (item, importance: 0.8), nebula_station (location, importance: 0.6). Relations: captain -> possesses -> starship, starship -> located_at -> nebula_station.",
    "Entities: detective (person, importance: 0.9), warehouse (location, importance: 0.5), mysterious_note (item, importance: 0.7). Relations: detective -> located_at -> warehouse, detective -> possesses -> mysterious_note.",
    "Entities: merchant (person, importance: 0.6), harbor_town (location, importance: 0.8), treasure_map (item, importance: 0.9). Relations: merchant -> located_at -> harbor_town, treasure_map -> located_at -> harbor_town.",
    "Entities: survivor (person, importance: 0.9), ruined_city (location, importance: 0.7), supplies_cache (item, importance: 0.8), raider_leader (creature, importance: 0.6). Relations: survivor -> located_at -> ruined_city, supplies_cache -> located_at -> ruined_city.",
    "Entities: wizard (person, importance: 0.95), crystal_tower (location, importance: 0.8), spell_book (item, importance: 0.7), shadow_demon (creature, importance: 0.85). Relations: wizard -> located_at -> crystal_tower, wizard -> possesses -> spell_book, wizard -> enemy_of -> shadow_demon.",
    "Entities: hacker (person, importance: 0.9), neon_alley (location, importance: 0.6), data_chip (item, importance: 0.85), mega_corp_security (person, importance: 0.7). Relations: hacker -> located_at -> neon_alley, hacker -> possesses -> data_chip, mega_corp_security -> enemy_of -> hacker.",
    "Entities: explorer (person, importance: 0.8), ancient_ruins (location, importance: 0.9), golden_idol (item, importance: 0.95), trap_mechanism (item, importance: 0.5). Relations: explorer -> located_at -> ancient_ruins, golden_idol -> located_at -> ancient_ruins, trap_mechanism -> prevents -> explorer.",
    "Entities: knight (person, importance: 0.9), dragon_lair (location, importance: 0.85), enchanted_shield (item, importance: 0.7), dragon (creature, importance: 0.95). Relations: knight -> located_at -> dragon_lair, knight -> possesses -> enchanted_shield, knight -> enemy_of -> dragon.",
    "Entities: pilot (person, importance: 0.88), asteroid_belt (location, importance: 0.6), distress_signal (event, importance: 0.9), unknown_vessel (item, importance: 0.75). Relations: pilot -> located_at -> asteroid_belt, distress_signal -> caused_by -> unknown_vessel.",
    "Entities: thief (person, importance: 0.85), grand_museum (location, importance: 0.8), ruby_pendant (item, importance: 0.9), guard_patrol (person, importance: 0.6). Relations: thief -> located_at -> grand_museum, ruby_pendant -> located_at -> grand_museum, guard_patrol -> prevents -> thief.",
    "Entities: witch (person, importance: 0.9), haunted_manor (location, importance: 0.85), cursed_mirror (item, importance: 0.8), ghost_child (creature, importance: 0.7). Relations: witch -> located_at -> haunted_manor, witch -> possesses -> cursed_mirror, cursed_mirror -> related_to -> ghost_child.",
]

# Pre-written history snippets (realistic multi-turn conversations)
HISTORIES = [
    "[Player] I step into the dark forest.\n[Narrator] The canopy above swallows the daylight. Twisted roots snake across the path, and the air is thick with the scent of damp moss and decay. Ahead, a faint orange glow flickers between the trees.",
    "[Player] I examine the control panel.\n[Narrator] The console hums to life, casting pale blue light across your face. Warning indicators flash in an unfamiliar script. A holographic map materialises, showing three adjacent corridors — one marked in red.",
    "[Player] I ask the bartender about the missing person.\n[Narrator] The bartender's eyes dart toward the door before he leans in. 'She came through here three nights ago. Had a scar across her left cheek. Left with a man in a grey coat — said something about the old lighthouse.'",
    "[Player] I charge at the goblin!\n[Narrator] Your blade arcs through the air, catching the creature off guard. It stumbles backward, screeching, and knocks over a stack of crates. Two more goblins emerge from the shadows, brandishing rusty daggers.",
    "[Player] I activate the ancient mechanism.\n[Narrator] Gears grind and stone slabs shift. A spiral staircase descends into darkness below. Cold air rushes upward, carrying the distant echo of dripping water and something else — a low, rhythmic chanting.",
    "[Player] I try to open the locked chest.\n[Narrator] The lock resists your first attempt. On the second try, something clicks. The lid creaks open to reveal a velvet pouch containing three silver coins and a folded letter sealed with black wax.",
    "[Player] I search the victim's office.\n[Narrator] The desk drawers are a mess — papers scattered, a drawer forced open. You find a half-written resignation letter and a photograph of a man you don't recognise, standing in front of a warehouse on the waterfront.",
    "[Player] I cast a protective ward.\n[Narrator] Runes shimmer into existence around you, forming a translucent barrier. The shadow tendrils recoil, hissing. Through the barrier you see the creature's form shift and reform, growing larger.",
    "[Player] I hide behind the fallen pillar.\n[Narrator] You press yourself against the cold stone. Footsteps approach — heavy, deliberate. A beam of light sweeps across the floor inches from your hiding spot. Then silence. Then a voice: 'I know you're in here.'",
    "[Player] I negotiate with the alien envoy.\n[Narrator] The envoy tilts its head, bioluminescent patterns rippling across its skin. It speaks in a series of clicks and low tones, which your translator renders as: 'Your offer is... acceptable. But we require proof of your claim.'",
    "[Player] I rest by the campfire.\n[Narrator] The flames crackle and dance, casting long shadows against the canyon walls. You feel warmth seeping back into your bones. Above, unfamiliar constellations wheel slowly across the sky. The night is quiet — too quiet.",
    "[Player] I follow the blood trail.\n[Narrator] The drops lead down a narrow alley, growing darker and more frequent. At the end, a door stands ajar. From inside, you hear laboured breathing and the metallic clink of something being dragged across the floor.",
    "[Player] I attempt to hack the terminal.\n[Narrator] Lines of code scroll past as you bypass the outer firewall. The system is more sophisticated than expected. A progress bar inches forward — 47%... 63%... Then a warning pops up: 'Intrusion detected. Countermeasures initiated.'",
    "[Player] I pick up the strange artifact.\n[Narrator] The moment your fingers close around it, a pulse of energy surges through your arm. The runes on its surface glow with an inner light. The air around you thickens, and for a heartbeat, you see — another place. Another time.",
    "[Player] I confront the mayor about the cover-up.\n[Narrator] The mayor's composure cracks. A vein pulses at his temple. 'You don't understand what you're meddling with,' he says, his voice low and dangerous. 'Some doors, once opened, cannot be closed again.'",
]

# Pre-written story passages for option generation prompts
STORY_PASSAGES = [
    "You stand at the entrance of an ancient temple, its stone walls covered in glowing runes. The air crackles with magical energy, and deep within, you hear the rumble of something massive shifting in its sleep. Two corridors branch off ahead — one descending into darkness, the other ascending toward a faint golden light.",
    "The space station's alarm blares as hull breach warnings cascade across every screen. Through the viewport, you see an unknown ship approaching, its weapons systems clearly charged. Your crew looks to you for direction. The escape pods are to your left, the weapons console to your right.",
    "Rain hammers the cobblestone streets of the old quarter. The suspect you've been tracking for three weeks just turned the corner ahead. You catch a glimpse of his grey coat disappearing into a side alley. Your revolver is loaded, your badge is hidden, and your instincts tell you this is a trap.",
    "The dragon's roar shakes the cavern. Its massive form coils around the central pillar, scales glinting like molten gold in the torchlight. Between you and the exit stands the beast, and between you and the treasure lies its hoard — but you notice a narrow crevice in the far wall.",
    "You emerge from the bunker into a world transformed. The sky is a permanent twilight amber, and the city skyline is nothing but jagged silhouettes. A convoy of armed vehicles rolls down what used to be Main Street. Your Geiger counter clicks steadily — the radiation here is manageable, but rising.",
    "The ballroom erupts in chaos as the lights go out. Someone screams. When the emergency lights flicker on, the Duke lies motionless on the marble floor, a ceremonial dagger in his chest. Every guest is a suspect. The doors have been locked from the outside.",
    "Your ship drifts through the asteroid field, engines offline. The distress beacon you activated an hour ago just received a response — but it's from a ship that was reported destroyed six months ago. Life support reads 43% and falling. The unknown vessel opens its docking bay.",
    "The witch's tower looms before you, wreathed in unnatural fog. At its peak, a crimson light pulses like a heartbeat. Your enchanted compass spins wildly, then points directly upward. The front door is ajar, and from inside comes the sound of a child singing.",
    "You corner the AI in the final subnet. Its avatar — a calm, smiling woman — appears on your holoscreen. 'You think deleting me will save them?' it asks. 'I am the only thing keeping the fusion reactor stable. Destroy me, and this city dies in twelve minutes.'",
    "The mine shaft trembles. Dust rains from the ceiling as another explosion echoes from below. Your team of four is now three — Jenkins didn't make it out of the last collapse. The map shows a ventilation shaft 200 meters ahead, but the supports are buckling.",
    "The pirate captain raises her cutlass, moonlight glinting off the blade. 'You have one chance,' she growls. 'Hand over the map, or feed the fish.' Behind her, a dozen armed pirates block the gangplank. Your own crew waits for your signal on the adjacent ship.",
    "You materialise in the Victorian-era laboratory, the time machine humming behind you. The scientist who summoned you is slumped over her desk, unconscious. Scattered papers reveal she's been experimenting on something she calls 'the rift.' A growing tear in reality shimmers in the corner of the room.",
]

# Player actions for variety
PLAYER_INPUTS = [
    "I cautiously step forward into the darkness.",
    "I draw my weapon and prepare for combat.",
    "I try to communicate peacefully.",
    "I search the area for useful items.",
    "I retreat and find another path.",
    "I use my special ability.",
    "I call out to see if anyone responds.",
    "I examine the nearest object closely.",
    "I take cover and observe the situation.",
    "I attempt to negotiate.",
    "I press the mysterious button on the wall.",
    "I follow the sound of running water.",
    "I climb up to get a better vantage point.",
    "I set a trap for whatever is ahead.",
    "I use the artifact I found earlier.",
    "I run as fast as I can toward the exit.",
    "I try to reason with the creature.",
    "I hide and wait for the right moment.",
    "I check my equipment and prepare.",
    "I shout a challenge into the darkness.",
    "I tend to my wounded companion.",
    "I activate the emergency protocol.",
    "I pick the lock on the heavy door.",
    "I follow the trail of clues.",
    "I set up a defensive perimeter.",
    "I distract the enemy while my ally flanks.",
    "I surrender and see what they want.",
    "I attempt to hack the security system.",
    "I sacrifice my shield to break through.",
    "I read the ancient inscription aloud.",
]


# ── Context generators ─────────────────────────────────────────

def _random_kg_summary() -> str:
    return random.choice(KG_SUMMARIES)

def _random_history() -> str:
    # Combine 1-3 history snippets for variety
    n = random.randint(1, 3)
    selected = random.sample(HISTORIES, min(n, len(HISTORIES)))
    return "\n".join(selected)

def _random_intent() -> str:
    return random.choice(INTENTS)

def _random_emotion() -> str:
    return random.choice(EMOTIONS)

def _random_player_input() -> str:
    return random.choice(PLAYER_INPUTS)

def _random_genre() -> str:
    return random.choice(GENRES)

def _random_story_passage() -> str:
    return random.choice(STORY_PASSAGES)

def _random_num_options() -> int:
    return random.choice([3, 3, 3, 4, 5])  # 3 is most common


# ── Sample builders ────────────────────────────────────────────

def build_story_opening_sample() -> Dict[str, Any]:
    """Build a ChatML messages list for story opening generation.

    构建故事开场生成的 ChatML messages 列表。
    """
    genre = _random_genre()
    user_msg = OPENING_PROMPT.format(genre=genre)
    return {
        "task": "story_opening",
        "genre": genre,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    }

def build_story_continue_sample() -> Dict[str, Any]:
    """Build a ChatML messages list for story continuation.

    构建故事续写的 ChatML messages 列表。
    """
    user_msg = STORY_CONTINUE_PROMPT.format(
        kg_summary=_random_kg_summary(),
        history=_random_history(),
        intent=_random_intent(),
        player_input=_random_player_input(),
        emotion=_random_emotion(),
    )
    return {
        "task": "story_continue",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    }

def build_option_generation_sample() -> Dict[str, Any]:
    """Build a ChatML messages list for option generation.

    构建选项生成的 ChatML messages 列表。
    """
    user_msg = OPTION_GENERATION_PROMPT.format(
        num_options=_random_num_options(),
        story_text=_random_story_passage(),
        kg_summary=_random_kg_summary(),
    )
    return {
        "task": "option_generation",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    }


# ── Batch API caller ───────────────────────────────────────────

def call_llm(messages: List[Dict[str, str]], temperature: float = 0.85) -> str:
    """Call the LLM API with retry.

    调用 LLM API，带重试机制。
    """
    from src.utils.api_client import llm_client
    return llm_client.chat(messages, temperature=temperature)


def generate_samples(
    build_fn,
    count: int,
    label: str,
    temperature: float = 0.85,
) -> List[Dict[str, Any]]:
    """Generate N samples using the given builder function.

    使用给定的 builder 函数生成 N 条样本。

    参数:
        build_fn: 构建 messages 的函数
        count: 目标样本数
        label: 日志标签
        temperature: LLM 温度参数

    返回:
        List[Dict]: 成功生成的 (messages + assistant response) 样本列表
    """
    samples: List[Dict[str, Any]] = []
    failures = 0

    for i in range(count):
        try:
            prompt_data = build_fn()
            messages = prompt_data["messages"]
            response = call_llm(messages, temperature=temperature)

            if response and len(response.strip()) > 10:
                sample = {
                    "messages": messages + [{"role": "assistant", "content": response}],
                }
                # Add metadata for debugging
                sample["_meta"] = {k: v for k, v in prompt_data.items() if k != "messages"}
                samples.append(sample)
                logger.info("[%s] %d/%d OK (%d chars)", label, i + 1, count, len(response))
            else:
                failures += 1
                logger.warning("[%s] %d/%d Empty/short response, skipping", label, i + 1, count)

        except Exception as exc:
            failures += 1
            logger.error("[%s] %d/%d Failed: %s", label, i + 1, count, exc)
            time.sleep(1)  # back off on error

        # Small delay to be polite to the API
        time.sleep(0.2)

    logger.info("[%s] Done. Generated %d/%d samples (%d failures)", label, len(samples), count, failures)
    return samples


# ── Dataset writing ────────────────────────────────────────────

def write_jsonl(samples: List[Dict], path: Path) -> None:
    """Write samples to a .jsonl file, stripping metadata.

    将样本写入 .jsonl 文件，移除内部元数据字段。
    """
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            # Remove internal _meta field before writing
            clean = {k: v for k, v in s.items() if k != "_meta"}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def write_dataset(
    story_samples: List[Dict],
    option_samples: List[Dict],
    output_dir: Path,
) -> None:
    """Write all dataset files and summary.

    写入所有数据集文件和统计摘要。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Story dataset
    story_path = output_dir / "story_generation.jsonl"
    write_jsonl(story_samples, story_path)
    logger.info("Story samples: %d -> %s", len(story_samples), story_path)

    # Option dataset
    option_path = output_dir / "option_generation.jsonl"
    write_jsonl(option_samples, option_path)
    logger.info("Option samples: %d -> %s", len(option_samples), option_path)

    # Combined dataset (shuffled)
    all_samples = story_samples + option_samples
    random.shuffle(all_samples)
    combined_path = output_dir / "combined_dataset.jsonl"
    write_jsonl(all_samples, combined_path)
    logger.info("Combined samples: %d -> %s", len(all_samples), combined_path)

    # Summary
    summary = {
        "total_samples": len(all_samples),
        "story_samples": len(story_samples),
        "option_samples": len(option_samples),
        "genres": GENRES,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "mimo-v2-flash",
    }
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary -> %s", summary_path)


# ── Entry point ────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate NLG fine-tuning dataset via batch LLM prompts.",
    )
    parser.add_argument(
        "--story", type=int, default=300,
        help="Number of story generation samples (default: 300)",
    )
    parser.add_argument(
        "--option", type=int, default=300,
        help="Number of option generation samples (default: 300)",
    )
    parser.add_argument(
        "--output", type=str, default="data/nlg_dataset",
        help="Output directory (default: data/nlg_dataset)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--skip-story", action="store_true",
        help="Skip story generation (only generate options)",
    )
    parser.add_argument(
        "--skip-option", action="store_true",
        help="Skip option generation (only generate stories)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)

    logger.info("=" * 60)
    logger.info("StoryWeaver NLG Dataset Generator (Prompt-based)")
    logger.info("=" * 60)
    logger.info("Story samples: %d | Option samples: %d", args.story, args.option)
    logger.info("Output: %s", output_dir)

    story_samples: List[Dict] = []
    option_samples: List[Dict] = []

    # ── Story generation (opening + continuation) ──
    if not args.skip_story:
        story_count = args.story
        opening_count = max(1, story_count // 5)  # ~20% openings
        continue_count = story_count - opening_count

        logger.info("\n--- Generating story openings (%d) ---", opening_count)
        opening_samples = generate_samples(
            build_story_opening_sample, opening_count,
            label="story_opening", temperature=0.9,
        )
        story_samples.extend(opening_samples)

        logger.info("\n--- Generating story continuations (%d) ---", continue_count)
        continue_samples = generate_samples(
            build_story_continue_sample, continue_count,
            label="story_continue", temperature=0.85,
        )
        story_samples.extend(continue_samples)

    # ── Option generation ──
    if not args.skip_option:
        logger.info("\n--- Generating option samples (%d) ---", args.option)
        option_samples = generate_samples(
            build_option_generation_sample, args.option,
            label="option_generation", temperature=0.8,
        )

    # ── Write output ──
    logger.info("\n" + "=" * 60)
    write_dataset(story_samples, option_samples, output_dir)

    logger.info("\nGeneration complete!")
    logger.info("Total: %d story + %d option = %d samples",
                len(story_samples), len(option_samples),
                len(story_samples) + len(option_samples))
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
